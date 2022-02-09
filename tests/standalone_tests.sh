#!/bin/bash
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

# this environment variable allows special tests to run
export PL_RUN_STANDALONE_TESTS=1
# python arguments
defaults='-m coverage run --source pi_ml --append -m pytest --capture=no'

# find tests marked as `@RunIf(standalone=True)`. done manually instead of with pytest because it is faster
grep_output=$(grep --recursive --word-regexp 'tests' --regexp 'standalone=True' --include '*.py' --exclude 'tests/conftest.py')

# file paths, remove duplicates
files=$(echo "$grep_output" | cut -f1 -d: | sort | uniq)

# get the list of parametrizations. we need to call them separately. the last two lines are removed.
# note: if there's a syntax error, this will fail with some garbled output
if [[ "$OSTYPE" == "darwin"* ]]; then
  parametrizations=$(pytest $files --collect-only --quiet "$@" | tail -r | sed -e '1,3d' | tail -r)
else
  parametrizations=$(pytest $files --collect-only --quiet "$@" | head -n -2)
fi
parametrizations_arr=($parametrizations)

# tests to skip - space separated
blocklist='tests/profiler/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx'
report=''

for i in "${!parametrizations_arr[@]}"; do
  parametrization=${parametrizations_arr[$i]}

  # check blocklist
  if echo $blocklist | grep -F "${parametrization}"; then
    report+="Skipped\t$parametrization\n"
    continue
  fi

  # run the test
  echo "Running ${parametrization}"
  python ${defaults} "${parametrization}"

  report+="Ran\t$parametrization\n"
done

if nvcc --version; then
    nvprof --profile-from-start off -o trace_name.prof -- python ${defaults} tests/profiler/test_profiler.py::test_pytorch_profiler_nested_emit_nvtx
fi

# needs to run outside of `pytest`
python tests/utilities/test_warnings.py
if [ $? -eq 0 ]; then
    report+="Ran\ttests/utilities/test_warnings.py\n"
fi

# TODO: enable when CI uses torch>=1.9
# test deadlock is properly handled with TorchElastic.
# LOGS=$(PL_RUN_STANDALONE_TESTS=1 PL_RECONCILE_PROCESS=1 python -m torch.distributed.run --nproc_per_node=2 --max_restarts 0 -m coverage run --source pi_ml -a tests/plugins/environments/torch_elastic_deadlock.py | grep "SUCCEEDED")
# if  [ -z "$LOGS" ]; then
#    exit 1
# fi
# report+="Ran\ttests/plugins/environments/torch_elastic_deadlock.py\n"

# test that a user can manually launch individual processes
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
args="--trainer.gpus 2 --trainer.strategy ddp --trainer.max_epochs=1 --trainer.limit_train_batches=1 --trainer.limit_val_batches=1 --trainer.limit_test_batches=1"
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=1 python pl_examples/basic_examples/mnist_examples/image_classifier_5_lightning_datamodule.py ${args} &
MASTER_ADDR="localhost" MASTER_PORT=1234 LOCAL_RANK=0 python pl_examples/basic_examples/mnist_examples/image_classifier_5_lightning_datamodule.py ${args}
report+="Ran\tmanual ddp launch test\n"

# echo test report
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'
