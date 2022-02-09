local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'pl',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 900, # 15 minutes, in seconds.

  image: 'pytorchlightning/pi_ml',
  imageTag: 'base-xla-py{PYTHON_VERSION}-torch{PYTORCH_VERSION}',

  tpuSettings+: {
    softwareVersion: 'pytorch-{PYTORCH_VERSION}',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      source ~/.bashrc
      conda activate lightning
      mkdir -p /home/runner/work/pytorch-lightning && cd /home/runner/work/pytorch-lightning
      git clone https://github.com/PyTorchLightning/pytorch-lightning.git
      cd pytorch-lightning
      echo $PWD
      git ls-remote --refs origin
      git fetch origin "refs/pull/{PR_NUMBER}/head:pr/{PR_NUMBER}" && git checkout "pr/{PR_NUMBER}"
      git checkout {SHA}
      pip install -e .
      echo $KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
      export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"
      coverage run --source=pi_ml -m pytest -v --capture=no \
          tests/strategies/test_tpu_spawn.py \
          tests/profiler/test_xla_profiler.py \
          pi_ml/utilities/xla_device.py \
          tests/accelerators/test_tpu.py \
          tests/callbacks/test_device_stats_monitor.py \
          tests/models/test_tpu.py
      test_exit_code=$?
      echo "\n||| END PYTEST LOGS |||\n"
      coverage xml
      cat coverage.xml | tr -d '\t'
      test $test_exit_code -eq 0
    |||
  ),
};

tputests.oneshotJob
