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
from unittest import mock
from unittest.mock import Mock

import pytest
import torch

from pi_ml import Trainer
from pi_ml.callbacks import ModelCheckpoint
from pi_ml.plugins.environments import SLURMEnvironment
from pi_ml.trainer.states import TrainerFn
from tests.helpers import BoringModel


# TODO: remove HPCHookedModel in v1.8
class HPCHookedModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.hpc_save_called = 0
        self.hpc_load_called = 0

    def on_hpc_save(self, checkpoint):
        assert "state_dict" in checkpoint
        self.hpc_save_called += 1

    def on_hpc_load(self, checkpoint):
        assert "state_dict" in checkpoint
        self.hpc_load_called += 1


# TODO: remove test_hpc_hook_calls in v1.8
@mock.patch(
    "pi_ml.trainer.connectors.accelerator_connector.AcceleratorConnector._is_slurm_managing_tasks",
    return_value=True,
)
def test_hpc_hook_calls(mock_slurm_env, tmpdir):
    model = HPCHookedModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, enable_checkpointing=False, logger=False)
    environment = trainer._accelerator_connector.cluster_environment
    assert isinstance(environment, SLURMEnvironment)
    assert environment.auto_requeue
    with pytest.deprecated_call(
        match=r"Method `LightningModule.on_hpc_save` is deprecated in v1.6 and will be removed in v1.8."
    ):
        trainer.fit(model)

    # simulate snapshot on slurm
    hpc_save_path = trainer._checkpoint_connector.hpc_save_path(tmpdir)
    trainer.save_checkpoint(hpc_save_path)
    assert model.hpc_save_called == 1
    assert model.hpc_load_called == 0

    # new training run, restore from hpc checkpoint file automatically
    assert set(os.listdir(tmpdir)) == {"hpc_ckpt_1.ckpt"}
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, enable_checkpointing=False, logger=False)
    with pytest.deprecated_call(
        match=r"Method `LightningModule.on_hpc_save` is deprecated in v1.6 and will be removed in v1.8."
    ):
        trainer.fit(model)
    assert model.hpc_save_called == 1
    assert model.hpc_load_called == 1


def test_preloaded_checkpoint_lifecycle(tmpdir):
    """Tests that the preloaded checkpoint contents gets cleared from memory when it is not required anymore."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)

    connector = trainer._checkpoint_connector

    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint

    connector.resume_start()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint
    connector.resume_end()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint

    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2)
    connector = trainer._checkpoint_connector
    connector.resume_start(ckpt_path)
    assert connector.resume_checkpoint_path == ckpt_path
    assert connector._loaded_checkpoint
    assert isinstance(connector._loaded_checkpoint, dict)
    trainer.state.fn = TrainerFn.FITTING
    connector.resume_end()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint


def test_hpc_restore_attempt(tmpdir):
    """Test that restore() attempts to restore the hpc_ckpt with highest priority."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    hpc_ckpt_path = tmpdir / "hpc_ckpt_3.ckpt"
    trainer.save_checkpoint(hpc_ckpt_path)
    assert os.listdir(tmpdir) == ["hpc_ckpt_3.ckpt"]

    # set weights to zero
    for param in model.parameters():
        torch.nn.init.constant_(param, 0)

    # case 1: restore hpc first, no explicit resume path provided
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0
        torch.nn.init.constant_(param, 0)

    # case 2: explicit resume path provided, restore hpc anyway
    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    trainer.fit(model, ckpt_path="not existing")

    for param in model.parameters():
        assert param.abs().sum() > 0


def test_hpc_max_ckpt_version(tmpdir):
    """Test that the CheckpointConnector is able to find the hpc checkpoint file with the highest version."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmpdir / "hpc_ckpt.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_0.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_3.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_33.ckpt")

    assert trainer._checkpoint_connector._hpc_resume_path == str(tmpdir / "hpc_ckpt_33.ckpt")
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmpdir) == 33
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmpdir / "not" / "existing")
        is None
    )


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_loops_restore(tmpdir):
    """Test that required loop state_dict is loaded correctly by checkpoint connector."""
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    trainer_args = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )
    trainer = Trainer(**trainer_args)
    trainer.fit(model)

    ckpt_path = str(tmpdir / "last.ckpt")

    trainer = Trainer(**trainer_args)
    for fn in TrainerFn:
        if fn != TrainerFn.TUNING:
            trainer_fn = getattr(trainer, f"{fn}_loop")
            trainer_fn.load_state_dict = Mock()

    for fn in TrainerFn:
        if fn != TrainerFn.TUNING:
            trainer.state.fn = fn
            trainer._checkpoint_connector.resume_start(ckpt_path)
            trainer._checkpoint_connector.restore_loops()

            trainer_loop = getattr(trainer, f"{fn}_loop")
            trainer_loop.load_state_dict.assert_called()
            trainer_loop.load_state_dict.reset_mock()

        for fn2 in TrainerFn:
            if fn2 not in (fn, TrainerFn.TUNING):
                trainer_loop2 = getattr(trainer, f"{fn2}_loop")
                trainer_loop2.load_state_dict.assert_not_called()
