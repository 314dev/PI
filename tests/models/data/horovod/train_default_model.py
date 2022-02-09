"""This script is meant to be executed from `../../test_horovod.py`.

Because Horovod uses a parallel programming model similar to MPI, unit tests for collective
ops like allreduce need to be run in parallel. The most common approach for running parallel
Horovod workers is to launch multiple replicas of the training script via the `horovodrun`
command-line tool:

.. code-block:: bash

    horovodrun -np 2 python train_default_model.py ...

Individual test parameters are configured by the serialized `--trainer-options` JSON object.

An non-zero exit code from this script on any rank will indicate failure, while a zero exit code
across all ranks indicates success.
"""

import argparse
import json
import os
import sys

import torch

# this is needed because Conda does not use `PYTHONPATH` env var while pip and virtualenv do
PYTHONPATH = os.getenv("PYTHONPATH", "")
if ":" in PYTHONPATH:
    sys.path = PYTHONPATH.split(":") + sys.path

from pi_ml import Trainer  # noqa: E402
from pi_ml.callbacks import ModelCheckpoint  # noqa: E402
from pi_ml.utilities import _HOROVOD_AVAILABLE  # noqa: E402

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd
else:
    print("You requested to import Horovod which is missing or not supported for your OS.")

from tests.helpers import BoringModel  # noqa: E402
from tests.helpers.utils import reset_seed, set_random_main_port  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--trainer-options", required=True)
parser.add_argument("--on-gpu", action="store_true", default=False)


def run_test_from_config(trainer_options, on_gpu, check_size=True):
    """Trains the default model with the given config."""
    set_random_main_port()
    reset_seed()

    ckpt_path = trainer_options["weights_save_path"]
    trainer_options.update(callbacks=[ModelCheckpoint(dirpath=ckpt_path)])

    class TestModel(BoringModel):
        def on_train_start(self) -> None:
            expected_device = torch.device("cuda", self.trainer.local_rank) if on_gpu else torch.device("cpu")
            assert self.device == expected_device

        def training_epoch_end(self, outputs) -> None:
            res = self.trainer.strategy.reduce(torch.tensor(1.0, device=self.device), reduce_op="sum")
            assert res.sum() == self.trainer.strategy.world_size

    model = TestModel()
    trainer = Trainer(**trainer_options)

    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    trainer.test(model)

    assert model.device == torch.device("cpu")

    # Horovod should be initialized following training. If not, this will raise an exception.
    if check_size:
        assert hvd.size() == 2

    if trainer.global_rank > 0:
        return

    # test model loading
    pretrained_model = BoringModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # test new model accuracy
    test_loaders = model.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    for dataloader in test_loaders:
        batch = next(iter(dataloader))
        pretrained_model(batch)

    # test HPC saving
    # save logger to make sure we get all the metrics
    if trainer.logger:
        trainer.logger.finalize("finished")
    hpc_save_path = trainer._checkpoint_connector.hpc_save_path(ckpt_path)
    trainer.save_checkpoint(hpc_save_path)
    # test HPC loading
    checkpoint_path = trainer._checkpoint_connector._CheckpointConnector__get_max_ckpt_path_from_folder(ckpt_path)
    trainer._checkpoint_connector.restore(checkpoint_path)

    if on_gpu:
        trainer = Trainer(gpus=1, strategy="horovod", max_epochs=1)
        # Test the root_gpu property
        assert trainer.root_gpu == hvd.local_rank()


if __name__ == "__main__":
    args = parser.parse_args()
    run_test_from_config(json.loads(args.trainer_options), args.on_gpu)
