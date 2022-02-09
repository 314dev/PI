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
import logging
import pickle
from typing import List, Optional
from unittest import mock
from unittest.mock import Mock

import cloudpickle
import numpy as np
import pytest
import torch

from pi_ml import seed_everything, Trainer
from pi_ml.callbacks import EarlyStopping, ModelCheckpoint
from pi_ml.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel

_logger = logging.getLogger(__name__)


def test_early_stopping_state_key():
    early_stopping = EarlyStopping(monitor="val_loss")
    assert early_stopping.state_key == "EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"


class EarlyStoppingTestRestore(EarlyStopping):
    # this class has to be defined outside the test function, otherwise we get pickle error
    def __init__(self, expected_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_state = expected_state
        # cache the state for each epoch
        self.saved_states = []

    def on_train_start(self, trainer, pl_module):
        if self.expected_state:
            assert self.on_save_checkpoint(trainer, pl_module, {}) == self.expected_state

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        self.saved_states.append(self.on_save_checkpoint(trainer, pl_module, {}).copy())


def test_resume_early_stopping_from_checkpoint(tmpdir):
    """Prevent regressions to bugs:

    https://github.com/PyTorchLightning/pytorch-lightning/issues/1464
    https://github.com/PyTorchLightning/pytorch-lightning/issues/1463
    """
    seed_everything(42)
    model = ClassificationModel()
    dm = ClassifDataModule()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, monitor="train_loss", save_top_k=1)
    early_stop_callback = EarlyStoppingTestRestore(None, monitor="train_loss")
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=0,
        max_epochs=4,
    )
    trainer.fit(model, datamodule=dm)

    assert len(early_stop_callback.saved_states) == 4

    checkpoint_filepath = checkpoint_callback.kth_best_model_path
    # ensure state is persisted properly
    checkpoint = torch.load(checkpoint_filepath)
    # the checkpoint saves "epoch + 1"
    early_stop_callback_state = early_stop_callback.saved_states[checkpoint["epoch"] - 1]
    assert 4 == len(early_stop_callback.saved_states)
    es_name = "EarlyStoppingTestRestore{'monitor': 'train_loss', 'mode': 'min'}"
    assert checkpoint["callbacks"][es_name] == early_stop_callback_state

    # ensure state is reloaded properly (assertion in the callback)
    early_stop_callback = EarlyStoppingTestRestore(early_stop_callback_state, monitor="train_loss")
    new_trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        callbacks=[early_stop_callback],
    )

    with pytest.raises(MisconfigurationException, match=r"You restored a checkpoint with current_epoch"):
        new_trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_filepath)


def test_early_stopping_no_extraneous_invocations(tmpdir):
    """Test to ensure that callback methods aren't being invoked outside of the callback handler."""
    model = ClassificationModel()
    dm = ClassifDataModule()
    early_stop_callback = EarlyStopping(monitor="train_loss")
    early_stop_callback._run_early_stopping_check = Mock()
    expected_count = 4
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        limit_train_batches=4,
        limit_val_batches=4,
        max_epochs=expected_count,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)

    assert trainer.early_stopping_callback == early_stop_callback
    assert trainer.early_stopping_callbacks == [early_stop_callback]
    assert early_stop_callback._run_early_stopping_check.call_count == expected_count


@pytest.mark.parametrize(
    "loss_values, patience, expected_stop_epoch",
    [([6, 5, 5, 5, 5, 5], 3, 4), ([6, 5, 4, 4, 3, 3], 1, 3), ([6, 5, 6, 5, 5, 5], 3, 4)],
)
def test_early_stopping_patience(tmpdir, loss_values: list, patience: int, expected_stop_epoch: int):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideValidationReturn(BoringModel):
        validation_return_values = torch.tensor(loss_values)

        def validation_epoch_end(self, outputs):
            loss = self.validation_return_values[self.current_epoch]
            self.log("test_val_loss", loss)

    model = ModelOverrideValidationReturn()
    early_stop_callback = EarlyStopping(monitor="test_val_loss", patience=patience, verbose=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        num_sanity_val_steps=0,
        max_epochs=10,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    assert trainer.current_epoch == expected_stop_epoch


@pytest.mark.parametrize("validation_step_none", [True, False])
@pytest.mark.parametrize(
    "loss_values, patience, expected_stop_epoch",
    [([6, 5, 5, 5, 5, 5], 3, 4), ([6, 5, 4, 4, 3, 3], 1, 3), ([6, 5, 6, 5, 5, 5], 3, 4)],
)
def test_early_stopping_patience_train(
    tmpdir, validation_step_none: bool, loss_values: list, patience: int, expected_stop_epoch: int
):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideTrainReturn(BoringModel):
        train_return_values = torch.tensor(loss_values)

        def training_epoch_end(self, outputs):
            loss = self.train_return_values[self.current_epoch]
            self.log("train_loss", loss)

    model = ModelOverrideTrainReturn()

    if validation_step_none:
        model.validation_step = None

    early_stop_callback = EarlyStopping(
        monitor="train_loss", patience=patience, verbose=True, check_on_train_epoch_end=True
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        num_sanity_val_steps=0,
        max_epochs=10,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    assert trainer.current_epoch == expected_stop_epoch


def test_pickling(tmpdir):
    early_stopping = EarlyStopping(monitor="foo")

    early_stopping_pickled = pickle.dumps(early_stopping)
    early_stopping_loaded = pickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)

    early_stopping_pickled = cloudpickle.dumps(early_stopping)
    early_stopping_loaded = cloudpickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)


def test_early_stopping_no_val_step(tmpdir):
    """Test that early stopping callback falls back to training metrics when no validation defined."""

    model = ClassificationModel()
    dm = ClassifDataModule()
    model.validation_step = None
    model.val_dataloader = None

    stopping = EarlyStopping(monitor="train_loss", min_delta=0.1, patience=0, check_on_train_epoch_end=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[stopping], overfit_batches=0.20, max_epochs=10)
    trainer.fit(model, datamodule=dm)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch < trainer.max_epochs - 1


@pytest.mark.parametrize(
    "stopping_threshold,divergence_theshold,losses,expected_epoch",
    [
        (None, None, [8, 4, 2, 3, 4, 5, 8, 10], 5),
        (2.9, None, [9, 8, 7, 6, 5, 6, 4, 3, 2, 1], 8),
        (None, 15.9, [9, 4, 2, 16, 32, 64], 3),
    ],
)
def test_early_stopping_thresholds(tmpdir, stopping_threshold, divergence_theshold, losses, expected_epoch):
    class CurrentModel(BoringModel):
        def validation_epoch_end(self, outputs):
            val_loss = losses[self.current_epoch]
            self.log("abc", val_loss)

    model = CurrentModel()
    early_stopping = EarlyStopping(
        monitor="abc", stopping_threshold=stopping_threshold, divergence_threshold=divergence_theshold
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stopping],
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        max_epochs=20,
    )
    trainer.fit(model)
    assert trainer.current_epoch == expected_epoch, "early_stopping failed"


@pytest.mark.parametrize("stop_value", [torch.tensor(np.inf), torch.tensor(np.nan)])
def test_early_stopping_on_non_finite_monitor(tmpdir, stop_value):

    losses = [4, 3, stop_value, 2, 1]
    expected_stop_epoch = 2

    class CurrentModel(BoringModel):
        def validation_epoch_end(self, outputs):
            val_loss = losses[self.current_epoch]
            self.log("val_loss", val_loss)

    model = CurrentModel()
    early_stopping = EarlyStopping(monitor="val_loss", check_finite=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stopping],
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        max_epochs=10,
    )
    trainer.fit(model)
    assert trainer.current_epoch == expected_stop_epoch
    assert early_stopping.stopped_epoch == expected_stop_epoch


@pytest.mark.parametrize("step_freeze, min_steps, min_epochs", [(5, 1, 1), (5, 1, 3), (3, 15, 1)])
def test_min_steps_override_early_stopping_functionality(tmpdir, step_freeze: int, min_steps: int, min_epochs: int):
    """Excepted Behaviour: IF `min_steps` was set to a higher value than the `trainer.global_step` when
    `early_stopping` is being triggered, THEN the trainer should continue until reaching `trainer.global_step` ==
    `min_steps`, and stop.

    IF `min_epochs` resulted in a higher number of steps than the `trainer.global_step`
        when `early_stopping` is being triggered,
    THEN the trainer should continue until reaching
        `trainer.global_step` == `min_epochs * len(train_dataloader)`, and stop.
    This test validate this expected behaviour

    IF both `min_epochs` and `min_steps` are provided and higher than the `trainer.global_step`
        when `early_stopping` is being triggered,
    THEN the highest between `min_epochs * len(train_dataloader)` and `min_steps` would be reached.

    Caveat: IF min_steps is divisible by len(train_dataloader), then it will do min_steps + len(train_dataloader)

    This test validate those expected behaviours
    """

    _logger.disabled = True

    original_loss_value = 10
    limit_train_batches = 3
    patience = 3

    class Model(BoringModel):
        def __init__(self, step_freeze):
            super().__init__()

            self._step_freeze = step_freeze

            self._loss_value = 10.0
            self._eps = 1e-1
            self._count_decrease = 0
            self._values = []

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            return {"test_val_loss": self._loss_value}

        def validation_epoch_end(self, outputs):
            _mean = np.mean([x["test_val_loss"] for x in outputs])
            if self.trainer.global_step <= self._step_freeze:
                self._count_decrease += 1
                self._loss_value -= self._eps
            self._values.append(_mean)
            self.log("test_val_loss", _mean)

    model = Model(step_freeze)
    model.training_step_end = None
    model.test_dataloader = None
    early_stop_callback = EarlyStopping(monitor="test_val_loss", patience=patience, verbose=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        min_steps=min_steps,
        min_epochs=min_epochs,
    )
    trainer.fit(model)

    # Make sure loss was properly decreased
    assert abs(original_loss_value - (model._count_decrease) * model._eps - model._loss_value) < 1e-6

    pos_diff = (np.diff(model._values) == 0).nonzero()[0][0]

    # Compute when the latest validation epoch end happened
    latest_validation_epoch_end = (pos_diff // limit_train_batches) * limit_train_batches
    if pos_diff % limit_train_batches == 0:
        latest_validation_epoch_end += limit_train_batches

    # Compute early stopping latest step
    by_early_stopping = latest_validation_epoch_end + (1 + limit_train_batches) * patience

    # Compute min_epochs latest step
    by_min_epochs = min_epochs * limit_train_batches

    # Make sure the trainer stops for the max of all minimum requirements
    assert trainer.global_step == max(min_steps, by_early_stopping, by_min_epochs), (
        trainer.global_step,
        max(min_steps, by_early_stopping, by_min_epochs),
        step_freeze,
        min_steps,
        min_epochs,
    )

    _logger.disabled = False


def test_early_stopping_mode_options():
    with pytest.raises(MisconfigurationException, match="`mode` can be .* got unknown_option"):
        EarlyStopping(monitor="foo", mode="unknown_option")


class EarlyStoppingModel(BoringModel):
    def __init__(self, expected_end_epoch: int, early_stop_on_train: bool):
        super().__init__()
        self.expected_end_epoch = expected_end_epoch
        self.early_stop_on_train = early_stop_on_train

    def _epoch_end(self) -> None:
        losses = [8, 4, 2, 3, 4, 5, 8, 10]
        loss = losses[self.current_epoch]
        self.log("abc", torch.tensor(loss))
        self.log("cba", torch.tensor(0))

    def training_epoch_end(self, outputs):
        if not self.early_stop_on_train:
            return
        self._epoch_end()

    def validation_epoch_end(self, outputs):
        if self.early_stop_on_train:
            return
        self._epoch_end()

    def on_train_end(self) -> None:
        assert self.trainer.current_epoch == self.expected_end_epoch, "Early Stopping Failed"


_ES_CHECK = dict(check_on_train_epoch_end=True)
_ES_CHECK_P3 = dict(patience=3, check_on_train_epoch_end=True)
_SPAWN_MARK = dict(marks=RunIf(skip_windows=True, skip_49370=True))


@pytest.mark.parametrize(
    "callbacks, expected_stop_epoch, check_on_train_epoch_end, strategy, devices",
    [
        ([EarlyStopping("abc"), EarlyStopping("cba", patience=3)], 3, False, None, 1),
        ([EarlyStopping("cba", patience=3), EarlyStopping("abc")], 3, False, None, 1),
        pytest.param([EarlyStopping("abc"), EarlyStopping("cba", patience=3)], 3, False, "ddp_spawn", 2, **_SPAWN_MARK),
        pytest.param([EarlyStopping("cba", patience=3), EarlyStopping("abc")], 3, False, "ddp_spawn", 2, **_SPAWN_MARK),
        ([EarlyStopping("abc", **_ES_CHECK), EarlyStopping("cba", **_ES_CHECK_P3)], 3, True, None, 1),
        ([EarlyStopping("cba", **_ES_CHECK_P3), EarlyStopping("abc", **_ES_CHECK)], 3, True, None, 1),
        pytest.param(
            [EarlyStopping("abc", **_ES_CHECK), EarlyStopping("cba", **_ES_CHECK_P3)],
            3,
            True,
            "ddp_spawn",
            2,
            **_SPAWN_MARK,
        ),
        pytest.param(
            [EarlyStopping("cba", **_ES_CHECK_P3), EarlyStopping("abc", **_ES_CHECK)],
            3,
            True,
            "ddp_spawn",
            2,
            **_SPAWN_MARK,
        ),
    ],
)
def test_multiple_early_stopping_callbacks(
    tmpdir,
    callbacks: List[EarlyStopping],
    expected_stop_epoch: int,
    check_on_train_epoch_end: bool,
    strategy: Optional[str],
    devices: int,
):
    """Ensure when using multiple early stopping callbacks we stop if any signals we should stop."""

    model = EarlyStoppingModel(expected_stop_epoch, check_on_train_epoch_end)

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=callbacks,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        max_epochs=20,
        strategy=strategy,
        accelerator="cpu",
        devices=devices,
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    "case",
    {
        "val_check_interval": {"val_check_interval": 0.3, "limit_train_batches": 10, "max_epochs": 10},
        "check_val_every_n_epoch": {"check_val_every_n_epoch": 2, "max_epochs": 5},
    }.items(),
)
def test_check_on_train_epoch_end_smart_handling(tmpdir, case):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            self.log("foo", 1)
            return super().validation_step(batch, batch_idx)

    case, kwargs = case
    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_val_batches=1,
        callbacks=EarlyStopping(monitor="foo"),
        enable_progress_bar=False,
        **kwargs,
    )

    side_effect = [(False, "A"), (True, "B")]
    with mock.patch(
        "pi_ml.callbacks.EarlyStopping._evaluate_stopping_criteria", side_effect=side_effect
    ) as es_mock:
        trainer.fit(model)

    assert es_mock.call_count == len(side_effect)
    if case == "val_check_interval":
        assert trainer.global_step == len(side_effect) * int(trainer.limit_train_batches * trainer.val_check_interval)
    else:
        assert trainer.current_epoch == len(side_effect) * trainer.check_val_every_n_epoch - 1


def test_early_stopping_squeezes():
    early_stopping = EarlyStopping(monitor="foo")
    trainer = Trainer()
    trainer.callback_metrics["foo"] = torch.tensor([[[0]]])

    with mock.patch(
        "pi_ml.callbacks.EarlyStopping._evaluate_stopping_criteria", return_value=(False, "")
    ) as es_mock:
        early_stopping._run_early_stopping_check(trainer)

    es_mock.assert_called_once_with(torch.tensor(0))
