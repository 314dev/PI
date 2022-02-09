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

import torch

from pi_ml import Callback, LightningModule, Trainer
from pi_ml.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    ProgressBarBase,
    TQDMProgressBar,
)
from pi_ml.trainer.connectors.callback_connector import CallbackConnector
from tests.helpers import BoringModel


def test_checkpoint_callbacks_are_last(tmpdir):
    """Test that checkpoint callbacks always get moved to the end of the list, with preserved order."""
    checkpoint1 = ModelCheckpoint(tmpdir)
    checkpoint2 = ModelCheckpoint(tmpdir)
    model_summary = ModelSummary()
    early_stopping = EarlyStopping(monitor="foo")
    lr_monitor = LearningRateMonitor()
    progress_bar = TQDMProgressBar()

    # no model reference
    trainer = Trainer(callbacks=[checkpoint1, progress_bar, lr_monitor, model_summary, checkpoint2])
    assert trainer.callbacks == [
        progress_bar,
        lr_monitor,
        model_summary,
        trainer.accumulation_scheduler,
        checkpoint1,
        checkpoint2,
    ]

    # no model callbacks
    model = LightningModule()
    model.configure_callbacks = lambda: []
    trainer.model = model
    cb_connector = CallbackConnector(trainer)
    cb_connector._attach_model_callbacks()
    assert trainer.callbacks == [
        progress_bar,
        lr_monitor,
        model_summary,
        trainer.accumulation_scheduler,
        checkpoint1,
        checkpoint2,
    ]

    # with model-specific callbacks that substitute ones in Trainer
    model = LightningModule()
    model.configure_callbacks = lambda: [checkpoint1, early_stopping, model_summary, checkpoint2]
    trainer = Trainer(callbacks=[progress_bar, lr_monitor, ModelCheckpoint(tmpdir)])
    trainer.model = model
    cb_connector = CallbackConnector(trainer)
    cb_connector._attach_model_callbacks()
    assert trainer.callbacks == [
        progress_bar,
        lr_monitor,
        trainer.accumulation_scheduler,
        early_stopping,
        model_summary,
        checkpoint1,
        checkpoint2,
    ]


class StatefulCallback0(Callback):
    def on_save_checkpoint(self, *args):
        return {"content0": 0}


class StatefulCallback1(Callback):
    def __init__(self, unique=None, other=None):
        self._unique = unique
        self._other = other

    @property
    def state_key(self):
        return self._generate_state_key(unique=self._unique)

    def on_save_checkpoint(self, *args):
        return {"content1": self._unique}


def test_all_callback_states_saved_before_checkpoint_callback(tmpdir):
    """Test that all callback states get saved even if the ModelCheckpoint is not given as last and when there are
    multiple callbacks of the same type."""

    callback0 = StatefulCallback0()
    callback1 = StatefulCallback1(unique="one")
    callback2 = StatefulCallback1(unique="two", other=2)
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename="all_states")
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
        limit_val_batches=1,
        callbacks=[
            callback0,
            # checkpoint callback does not have to be at the end
            checkpoint_callback,
            # callback2 and callback3 have the same type
            callback1,
            callback2,
        ],
    )
    trainer.fit(model)

    ckpt = torch.load(str(tmpdir / "all_states.ckpt"))
    state0 = ckpt["callbacks"]["StatefulCallback0"]
    state1 = ckpt["callbacks"]["StatefulCallback1{'unique': 'one'}"]
    state2 = ckpt["callbacks"]["StatefulCallback1{'unique': 'two'}"]
    assert "content0" in state0 and state0["content0"] == 0
    assert "content1" in state1 and state1["content1"] == "one"
    assert "content1" in state2 and state2["content1"] == "two"
    assert (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None, 'save_on_train_epoch_end': True}" in ckpt["callbacks"]
    )


def test_attach_model_callbacks():
    """Test that the callbacks defined in the model and through Trainer get merged correctly."""

    def _attach_callbacks(trainer_callbacks, model_callbacks):
        model = LightningModule()
        model.configure_callbacks = lambda: model_callbacks
        has_progress_bar = any(isinstance(cb, ProgressBarBase) for cb in trainer_callbacks + model_callbacks)
        trainer = Trainer(
            enable_checkpointing=False,
            enable_progress_bar=has_progress_bar,
            enable_model_summary=False,
            callbacks=trainer_callbacks,
        )
        trainer.model = model
        cb_connector = CallbackConnector(trainer)
        cb_connector._attach_model_callbacks()
        return trainer

    early_stopping = EarlyStopping(monitor="foo")
    progress_bar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor()
    grad_accumulation = GradientAccumulationScheduler({1: 1})

    # no callbacks
    trainer = _attach_callbacks(trainer_callbacks=[], model_callbacks=[])
    assert trainer.callbacks == [trainer.accumulation_scheduler]

    # callbacks of different types
    trainer = _attach_callbacks(trainer_callbacks=[early_stopping], model_callbacks=[progress_bar])
    assert trainer.callbacks == [early_stopping, trainer.accumulation_scheduler, progress_bar]

    # same callback type twice, different instance
    trainer = _attach_callbacks(
        trainer_callbacks=[progress_bar, EarlyStopping(monitor="foo")],
        model_callbacks=[early_stopping],
    )
    assert trainer.callbacks == [progress_bar, trainer.accumulation_scheduler, early_stopping]

    # multiple callbacks of the same type in trainer
    trainer = _attach_callbacks(
        trainer_callbacks=[
            LearningRateMonitor(),
            EarlyStopping(monitor="foo"),
            LearningRateMonitor(),
            EarlyStopping(monitor="foo"),
        ],
        model_callbacks=[early_stopping, lr_monitor],
    )
    assert trainer.callbacks == [trainer.accumulation_scheduler, early_stopping, lr_monitor]

    # multiple callbacks of the same type, in both trainer and model
    trainer = _attach_callbacks(
        trainer_callbacks=[
            LearningRateMonitor(),
            progress_bar,
            EarlyStopping(monitor="foo"),
            LearningRateMonitor(),
            EarlyStopping(monitor="foo"),
        ],
        model_callbacks=[early_stopping, lr_monitor, grad_accumulation, early_stopping],
    )
    assert trainer.callbacks == [progress_bar, early_stopping, lr_monitor, grad_accumulation, early_stopping]


def test_attach_model_callbacks_override_info(caplog):
    """Test that the logs contain the info about overriding callbacks returned by configure_callbacks."""
    model = LightningModule()
    model.configure_callbacks = lambda: [LearningRateMonitor(), EarlyStopping(monitor="foo")]
    trainer = Trainer(
        enable_checkpointing=False, callbacks=[EarlyStopping(monitor="foo"), LearningRateMonitor(), TQDMProgressBar()]
    )
    trainer.model = model
    cb_connector = CallbackConnector(trainer)
    with caplog.at_level(logging.INFO):
        cb_connector._attach_model_callbacks()

    assert "existing callbacks passed to Trainer: EarlyStopping, LearningRateMonitor" in caplog.text
