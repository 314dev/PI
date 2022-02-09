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

from pi_ml import Trainer
from pi_ml.callbacks import ModelCheckpoint
from pi_ml.loggers import LoggerCollection, TensorBoardLogger
from tests.helpers.boring_model import BoringModel


class TestModel(BoringModel):
    def __init__(self, expected_log_dir):
        super().__init__()
        self.expected_log_dir = expected_log_dir

    def training_step(self, *args, **kwargs):
        assert self.trainer.log_dir == self.expected_log_dir
        return super().training_step(*args, **kwargs)


def test_logdir(tmpdir):
    """Tests that the path is correct when checkpoint and loggers are used."""
    expected = os.path.join(tmpdir, "lightning_logs", "version_0")

    model = TestModel(expected)

    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, callbacks=[ModelCheckpoint(dirpath=tmpdir)])

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_no_checkpoint_cb(tmpdir):
    """Tests that the path is correct with no checkpoint."""
    expected = os.path.join(tmpdir, "lightning_logs", "version_0")
    model = TestModel(expected)

    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, enable_checkpointing=False)

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_no_logger(tmpdir):
    """Tests that the path is correct even when there is no logger."""
    expected = os.path.join(tmpdir)
    model = TestModel(expected)

    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, logger=False, callbacks=[ModelCheckpoint(dirpath=tmpdir)])

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_no_logger_no_checkpoint(tmpdir):
    """Tests that the path is correct even when there is no logger."""
    expected = os.path.join(tmpdir)
    model = TestModel(expected)

    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, logger=False, enable_checkpointing=False)

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_custom_callback(tmpdir):
    """Tests that the path is correct even when there is a custom callback."""
    expected = os.path.join(tmpdir, "lightning_logs", "version_0")
    model = TestModel(expected)

    trainer = Trainer(
        default_root_dir=tmpdir, max_steps=2, callbacks=[ModelCheckpoint(dirpath=os.path.join(tmpdir, "ckpts"))]
    )

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_custom_logger(tmpdir):
    """Tests that the path is correct even when there is a custom logger."""
    expected = os.path.join(tmpdir, "custom_logs", "version_0")
    model = TestModel(expected)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=2,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
        logger=TensorBoardLogger(save_dir=tmpdir, name="custom_logs"),
    )

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_logger_collection(tmpdir):
    """Tests that the logdir equals the default_root_dir when the logger is a LoggerCollection."""
    default_root_dir = tmpdir / "default_root_dir"
    save_dir = tmpdir / "save_dir"
    model = TestModel(default_root_dir)
    trainer = Trainer(
        default_root_dir=default_root_dir,
        max_steps=2,
        logger=[TensorBoardLogger(save_dir=save_dir, name="custom_logs")],
    )
    assert isinstance(trainer.logger, LoggerCollection)
    assert trainer.log_dir == default_root_dir

    trainer.fit(model)
    assert trainer.log_dir == default_root_dir
