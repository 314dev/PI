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
import os

import torch

import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pi_ml import Trainer
from pi_ml.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


def test_cpu_slurm_save_load(tmpdir):
    """Verify model save/load/checkpoint on CPU."""
    model = BoringModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)
    version = logger.version

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    trainer.fit(model)
    real_global_step = trainer.global_step

    # traning complete
    assert trainer.state.finished, "cpu model failed to complete"

    # predict with trained model before saving
    # make a prediction
    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        for batch in dataloader:
            break

    model.eval()
    pred_before_saving = model(batch)

    # test HPC saving
    # simulate snapshot on slurm
    # save logger to make sure we get all the metrics
    if logger:
        logger.finalize("finished")
    hpc_save_path = trainer._checkpoint_connector.hpc_save_path(trainer.weights_save_path)
    trainer.save_checkpoint(hpc_save_path)
    assert os.path.exists(hpc_save_path)

    # new logger file to get meta
    logger = tutils.get_default_logger(tmpdir, version=version)

    model = BoringModel()

    class _StartCallback(Callback):
        # set the epoch start hook so we can predict before the model does the full training
        def on_train_epoch_start(self, trainer, model):
            assert trainer.global_step == real_global_step and trainer.global_step > 0
            # predict with loaded model to make sure answers are the same
            mode = model.training
            model.eval()
            new_pred = model(batch)
            assert torch.eq(pred_before_saving, new_pred).all()
            model.train(mode)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        callbacks=[_StartCallback(), ModelCheckpoint(dirpath=tmpdir)],
    )
    # by calling fit again, we trigger training, loading weights from the cluster
    # and our hook to predict using current model before any more weight updates
    trainer.fit(model)


def test_early_stopping_cpu_model(tmpdir):
    class ModelTrainVal(BoringModel):
        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args, **kwargs)
            self.log("val_loss", output["x"])
            return output

    tutils.reset_seed()
    stopping = EarlyStopping(monitor="val_loss", min_delta=0.1)
    trainer_options = dict(
        callbacks=[stopping],
        default_root_dir=tmpdir,
        gradient_clip_val=1.0,
        track_grad_norm=2,
        enable_progress_bar=False,
        accumulate_grad_batches=2,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    model = ModelTrainVal()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


@RunIf(skip_windows=True, skip_49370=True)
def test_multi_cpu_model_ddp(tmpdir):
    """Make sure DDP works."""
    tutils.set_random_main_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        gpus=None,
        num_processes=2,
        strategy="ddp_spawn",
    )

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, data=dm, on_gpu=False)


def test_lbfgs_cpu_model(tmpdir):
    """Test each of the trainer options.

    Testing LBFGS optimizer
    """

    class ModelSpecifiedOptimizer(BoringModel):
        def __init__(self, optimizer_name, learning_rate):
            super().__init__()
            self.optimizer_name = optimizer_name
            self.learning_rate = learning_rate
            self.save_hyperparameters()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        enable_progress_bar=False,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
    )

    model = ModelSpecifiedOptimizer(optimizer_name="LBFGS", learning_rate=0.004)
    tpipes.run_model_test_without_loggers(trainer_options, model, min_acc=0.01)


def test_default_logger_callbacks_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        enable_progress_bar=False,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
    )

    model = BoringModel()
    tpipes.run_model_test_without_loggers(trainer_options, model, min_acc=0.01)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_running_test_after_fitting(tmpdir):
    """Verify test() on fitted model."""

    class ModelTrainValTest(BoringModel):
        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args, **kwargs)
            self.log("val_loss", output["x"])
            return output

        def test_step(self, *args, **kwargs):
            output = super().test_step(*args, **kwargs)
            self.log("test_loss", output["y"])
            return output

    model = ModelTrainValTest()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer, key="test_loss", thr=0.5)


def test_running_test_no_val(tmpdir):
    """Verify `test()` works on a model with no `val_dataloader`.

    It performs train and test only
    """

    class ModelTrainTest(BoringModel):
        def val_dataloader(self):
            pass

        def test_step(self, *args, **kwargs):
            output = super().test_step(*args, **kwargs)
            self.log("test_loss", output["y"])
            return output

    model = ModelTrainTest()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer, key="test_loss")


def test_simple_cpu(tmpdir):
    """Verify continue training session on CPU."""
    model = BoringModel()

    # fit model
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.1, limit_train_batches=20)
    trainer.fit(model)

    # traning complete
    assert trainer.state.finished, "amp + ddp model failed to complete"


def test_cpu_model(tmpdir):
    """Make sure model trains on CPU."""
    trainer_options = dict(
        default_root_dir=tmpdir, enable_progress_bar=False, max_epochs=1, limit_train_batches=4, limit_val_batches=4
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)


def test_all_features_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        track_grad_norm=2,
        enable_progress_bar=False,
        accumulate_grad_batches=2,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model, on_gpu=False, min_acc=0.01)
