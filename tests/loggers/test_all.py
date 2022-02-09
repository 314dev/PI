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
import inspect
import os
import pickle
from unittest import mock
from unittest.mock import ANY

import pytest
import torch

import tests.helpers.utils as tutils
from pi_ml import Callback, Trainer
from pi_ml.loggers import (
    CometLogger,
    CSVLogger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    TestTubeLogger,
    WandbLogger,
)
from pi_ml.loggers.base import DummyExperiment
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf
from tests.loggers.test_comet import _patch_comet_atexit
from tests.loggers.test_mlflow import mock_mlflow_run_creation
from tests.loggers.test_neptune import create_neptune_mock


def _get_logger_args(logger_class, save_dir):
    logger_args = {}
    if "save_dir" in inspect.getfullargspec(logger_class).args:
        logger_args.update(save_dir=str(save_dir))
    if "offline_mode" in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline_mode=True)
    if "offline" in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline=True)
    if issubclass(logger_class, NeptuneLogger):
        logger_args.update(mode="offline")
    return logger_args


def _instantiate_logger(logger_class, save_dir, **override_kwargs):
    args = _get_logger_args(logger_class, save_dir)
    args.update(**override_kwargs)
    logger = logger_class(**args)
    return logger


def test_loggers_fit_test_all(tmpdir, monkeypatch):
    """Verify that basic functionality of all loggers."""

    _test_loggers_fit_test(tmpdir, TensorBoardLogger)

    with mock.patch("pi_ml.loggers.comet.comet_ml"), mock.patch(
        "pi_ml.loggers.comet.CometOfflineExperiment"
    ):
        _patch_comet_atexit(monkeypatch)
        _test_loggers_fit_test(tmpdir, CometLogger)

    with mock.patch("pi_ml.loggers.mlflow.mlflow"), mock.patch(
        "pi_ml.loggers.mlflow.MlflowClient"
    ):
        _test_loggers_fit_test(tmpdir, MLFlowLogger)

    with mock.patch("pi_ml.loggers.neptune.neptune", new_callable=create_neptune_mock):
        _test_loggers_fit_test(tmpdir, NeptuneLogger)

    with mock.patch("pi_ml.loggers.test_tube.Experiment"), pytest.deprecated_call(
        match="TestTubeLogger is deprecated since v1.5"
    ):
        _test_loggers_fit_test(tmpdir, TestTubeLogger)

    with mock.patch("pi_ml.loggers.wandb.wandb") as wandb:
        wandb.run = None
        wandb.init().step = 0
        _test_loggers_fit_test(tmpdir, WandbLogger)


def _test_loggers_fit_test(tmpdir, logger_class):
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("train_some_val", loss)
            return {"loss": loss}

        def validation_epoch_end(self, outputs) -> None:
            avg_val_loss = torch.stack([x["x"] for x in outputs]).mean()
            self.log_dict({"early_stop_on": avg_val_loss, "val_loss": avg_val_loss ** 0.5})

        def test_epoch_end(self, outputs) -> None:
            avg_test_loss = torch.stack([x["y"] for x in outputs]).mean()
            self.log("test_loss", avg_test_loss)

    class StoreHistoryLogger(logger_class):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.history = []

        def log_metrics(self, metrics, step):
            super().log_metrics(metrics, step)
            self.history.append((step, metrics))

    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = StoreHistoryLogger(**logger_args)

    if logger_class == WandbLogger:
        # required mocks for Trainer
        logger.experiment.id = "foo"
        logger.experiment.project_name.return_value = "bar"

    if logger_class == CometLogger:
        logger.experiment.id = "foo"
        logger.experiment.project_name = "bar"

    if logger_class == TestTubeLogger:
        logger.experiment.version = "foo"
        logger.experiment.name = "bar"

    if logger_class == MLFlowLogger:
        logger = mock_mlflow_run_creation(logger, experiment_id="foo", run_id="bar")

    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)
    trainer.test()

    log_metric_names = [(s, sorted(m.keys())) for s, m in logger.history]
    if logger_class == TensorBoardLogger:
        expected = [
            (0, ["epoch", "train_some_val"]),
            (0, ["early_stop_on", "epoch", "val_loss"]),
            (1, ["epoch", "test_loss"]),
        ]
        assert log_metric_names == expected
    else:
        expected = [
            (0, ["epoch", "train_some_val"]),
            (0, ["early_stop_on", "epoch", "val_loss"]),
            (1, ["epoch", "test_loss"]),
        ]
        assert log_metric_names == expected


def test_loggers_save_dir_and_weights_save_path_all(tmpdir, monkeypatch):
    """Test the combinations of save_dir, weights_save_path and default_root_dir."""

    _test_loggers_save_dir_and_weights_save_path(tmpdir, TensorBoardLogger)

    with mock.patch("pi_ml.loggers.comet.comet_ml"), mock.patch(
        "pi_ml.loggers.comet.CometOfflineExperiment"
    ):
        _patch_comet_atexit(monkeypatch)
        _test_loggers_save_dir_and_weights_save_path(tmpdir, CometLogger)

    with mock.patch("pi_ml.loggers.mlflow.mlflow"), mock.patch(
        "pi_ml.loggers.mlflow.MlflowClient"
    ):
        _test_loggers_save_dir_and_weights_save_path(tmpdir, MLFlowLogger)

    with mock.patch("pi_ml.loggers.test_tube.Experiment"), pytest.deprecated_call(
        match="TestTubeLogger is deprecated since v1.5"
    ):
        _test_loggers_save_dir_and_weights_save_path(tmpdir, TestTubeLogger)

    with mock.patch("pi_ml.loggers.wandb.wandb"):
        _test_loggers_save_dir_and_weights_save_path(tmpdir, WandbLogger)


def _test_loggers_save_dir_and_weights_save_path(tmpdir, logger_class):
    class TestLogger(logger_class):
        # for this test it does not matter what these attributes are
        # so we standardize them to make testing easier
        @property
        def version(self):
            return "version"

        @property
        def name(self):
            return "name"

    model = BoringModel()
    trainer_args = dict(default_root_dir=tmpdir, max_steps=1)

    # no weights_save_path given
    save_dir = tmpdir / "logs"
    weights_save_path = None
    logger = TestLogger(**_get_logger_args(TestLogger, save_dir))
    trainer = Trainer(**trainer_args, logger=logger, weights_save_path=weights_save_path)
    trainer.fit(model)
    assert trainer.weights_save_path == trainer.default_root_dir
    assert trainer.checkpoint_callback.dirpath == os.path.join(logger.save_dir, "name", "version", "checkpoints")
    assert trainer.default_root_dir == tmpdir

    # with weights_save_path given, the logger path and checkpoint path should be different
    save_dir = tmpdir / "logs"
    weights_save_path = tmpdir / "weights"
    logger = TestLogger(**_get_logger_args(TestLogger, save_dir))
    trainer = Trainer(**trainer_args, logger=logger, weights_save_path=weights_save_path)
    trainer.fit(model)
    assert trainer.weights_save_path == weights_save_path
    assert trainer.logger.save_dir == save_dir
    assert trainer.checkpoint_callback.dirpath == weights_save_path / "name" / "version" / "checkpoints"
    assert trainer.default_root_dir == tmpdir

    # no logger given
    weights_save_path = tmpdir / "weights"
    trainer = Trainer(**trainer_args, logger=False, weights_save_path=weights_save_path)
    trainer.fit(model)
    assert trainer.weights_save_path == weights_save_path
    assert trainer.checkpoint_callback.dirpath == weights_save_path / "checkpoints"
    assert trainer.default_root_dir == tmpdir


@pytest.mark.parametrize(
    "logger_class",
    [
        CometLogger,
        CSVLogger,
        MLFlowLogger,
        TensorBoardLogger,
        TestTubeLogger,
        # The WandbLogger gets tested for pickling in its own test.
        # The NeptuneLogger gets tested for pickling in its own test.
    ],
)
def test_loggers_pickle_all(tmpdir, monkeypatch, logger_class):
    """Test that the logger objects can be pickled.

    This test only makes sense if the packages are installed.
    """
    _patch_comet_atexit(monkeypatch)
    try:
        if logger_class is TestTubeLogger:
            with pytest.deprecated_call(match="TestTubeLogger is deprecated since v1.5"):
                _test_loggers_pickle(tmpdir, monkeypatch, logger_class)
        else:
            _test_loggers_pickle(tmpdir, monkeypatch, logger_class)
    except (ImportError, ModuleNotFoundError):
        pytest.xfail(f"pickle test requires {logger_class.__class__} dependencies to be installed.")


def _test_loggers_pickle(tmpdir, monkeypatch, logger_class):
    """Verify that pickling trainer with logger works."""
    _patch_comet_atexit(monkeypatch)

    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = logger_class(**logger_args)

    # this can cause pickle error if the experiment object is not picklable
    # the logger needs to remove it from the state before pickle
    _ = logger.experiment

    # logger also has to avoid adding un-picklable attributes to self in .save
    logger.log_metrics({"a": 1})
    logger.save()

    # test pickling loggers
    pickle.dumps(logger)

    trainer = Trainer(max_epochs=1, logger=logger)
    pkl_bytes = pickle.dumps(trainer)

    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})

    # make sure we restord properly
    assert trainer2.logger.name == logger.name
    assert trainer2.logger.save_dir == logger.save_dir


@pytest.mark.parametrize(
    "extra_params",
    [
        pytest.param(dict(max_epochs=1, auto_scale_batch_size=True), id="Batch-size-Finder"),
        pytest.param(dict(max_epochs=3, auto_lr_find=True), id="LR-Finder"),
    ],
)
def test_logger_reset_correctly(tmpdir, extra_params):
    """Test that the tuners do not alter the logger reference."""

    class CustomModel(BoringModel):
        def __init__(self, lr=0.1, batch_size=1):
            super().__init__()
            self.save_hyperparameters()

    tutils.reset_seed()
    model = CustomModel()
    trainer = Trainer(default_root_dir=tmpdir, **extra_params)
    logger1 = trainer.logger
    trainer.tune(model)
    logger2 = trainer.logger
    logger3 = model.logger

    assert logger1 == logger2, "Finder altered the logger of trainer"
    assert logger2 == logger3, "Finder altered the logger of model"


class RankZeroLoggerCheck(Callback):
    # this class has to be defined outside the test function, otherwise we get pickle error
    # due to the way ddp process is launched

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        is_dummy = isinstance(trainer.logger.experiment, DummyExperiment)
        if trainer.is_global_zero:
            assert not is_dummy
        else:
            assert is_dummy
            assert pl_module.logger.experiment.something(foo="bar") is None


@RunIf(skip_windows=True, skip_49370=True, skip_hanging_spawn=True)
@pytest.mark.parametrize(
    "logger_class", [CometLogger, CSVLogger, MLFlowLogger, NeptuneLogger, TensorBoardLogger, TestTubeLogger]
)
def test_logger_created_on_rank_zero_only(tmpdir, monkeypatch, logger_class):
    """Test that loggers get replaced by dummy loggers on global rank > 0."""
    _patch_comet_atexit(monkeypatch)
    try:
        if logger_class is TestTubeLogger:
            with pytest.deprecated_call(match="TestTubeLogger is deprecated since v1.5"):
                _test_logger_created_on_rank_zero_only(tmpdir, logger_class)
        else:
            _test_logger_created_on_rank_zero_only(tmpdir, logger_class)
    except (ImportError, ModuleNotFoundError):
        pytest.xfail(f"multi-process test requires {logger_class.__class__} dependencies to be installed.")


def _test_logger_created_on_rank_zero_only(tmpdir, logger_class):
    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = logger_class(**logger_args)
    model = BoringModel()
    trainer = Trainer(
        logger=logger,
        default_root_dir=tmpdir,
        strategy="ddp_spawn",
        accelerator="cpu",
        devices=2,
        max_steps=1,
        callbacks=[RankZeroLoggerCheck()],
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def test_logger_with_prefix_all(tmpdir, monkeypatch):
    """Test that prefix is added at the beginning of the metric keys."""
    prefix = "tmp"

    # Comet
    with mock.patch("pi_ml.loggers.comet.comet_ml"), mock.patch(
        "pi_ml.loggers.comet.CometOfflineExperiment"
    ):
        _patch_comet_atexit(monkeypatch)
        logger = _instantiate_logger(CometLogger, save_dir=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log_metrics.assert_called_once_with({"tmp-test": 1.0}, epoch=None, step=0)

    # MLflow
    with mock.patch("pi_ml.loggers.mlflow.mlflow"), mock.patch(
        "pi_ml.loggers.mlflow.MlflowClient"
    ):
        logger = _instantiate_logger(MLFlowLogger, save_dir=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log_metric.assert_called_once_with(ANY, "tmp-test", 1.0, ANY, 0)

    # Neptune
    with mock.patch("pi_ml.loggers.neptune.neptune"):
        logger = _instantiate_logger(NeptuneLogger, api_key="test", project="project", save_dir=tmpdir, prefix=prefix)
        assert logger.experiment.__getitem__.call_count == 2
        logger.log_metrics({"test": 1.0}, step=0)
        assert logger.experiment.__getitem__.call_count == 3
        logger.experiment.__getitem__.assert_called_with("tmp/test")
        logger.experiment.__getitem__().log.assert_called_once_with(1.0)

    # TensorBoard
    with mock.patch("pi_ml.loggers.tensorboard.SummaryWriter"):
        logger = _instantiate_logger(TensorBoardLogger, save_dir=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.add_scalar.assert_called_once_with("tmp-test", 1.0, 0)

    # TestTube
    with mock.patch("pi_ml.loggers.test_tube.Experiment"), pytest.deprecated_call(
        match="TestTubeLogger is deprecated since v1.5"
    ):
        logger = _instantiate_logger(TestTubeLogger, save_dir=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log.assert_called_once_with({"tmp-test": 1.0}, global_step=0)

    # WandB
    with mock.patch("pi_ml.loggers.wandb.wandb") as wandb:
        logger = _instantiate_logger(WandbLogger, save_dir=tmpdir, prefix=prefix)
        wandb.run = None
        wandb.init().step = 0
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log.assert_called_once_with({"tmp-test": 1.0, "trainer/global_step": 0})
