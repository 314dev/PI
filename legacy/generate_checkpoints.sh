#!/bin/bash
# Sample call:
#  bash generate_checkpoints.sh 1.0.2 1.0.3 1.0.4

set -e

LEGACY_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
FROZEN_MIN_PT_VERSION="1.6"

echo $LEGACY_PATH
# install some PT version here so it does not need to reinstalled for each env
pip install virtualenv "torch==1.6" --quiet

ENV_PATH="$LEGACY_PATH/vEnv"

# iterate over all arguments assuming that each argument is version
for ver in "$@"
do
	echo "processing version: $ver"
	# mkdir "$LEGACY_PATH/$ver"

  # create local env
  echo $ENV_PATH
  virtualenv $ENV_PATH --system-site-packages
  # activate and install PL version
  source "$ENV_PATH/bin/activate"
  # there are problem to load ckpt in older versions since they are saved the newer versions
  pip install "pi_ml==$ver" "torch==$FROZEN_MIN_PT_VERSION" "torchmetrics" "scikit-learn" --quiet

  python --version
  pip --version
  pip list | grep torch

  python "$LEGACY_PATH/simple_classif_training.py"
  cp "$LEGACY_PATH/simple_classif_training.py" ${LEGACY_PATH}/checkpoints/${ver}

  mv ${LEGACY_PATH}/checkpoints/${ver}/lightning_logs/version_0/checkpoints/*.ckpt ${LEGACY_PATH}/checkpoints/${ver}/
  rm -rf ${LEGACY_PATH}/checkpoints/${ver}/lightning_logs

  deactivate
  # clear env
  rm -rf $ENV_PATH

done
