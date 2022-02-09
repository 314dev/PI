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
import os
import re
from typing import Any, Dict, Optional

import torch
from torchmetrics import Metric

import pi_ml as pl
from pi_ml.loops.utilities import _is_max_limit_reached
from pi_ml.plugins.environments import SLURMEnvironment
from pi_ml.trainer.states import TrainerFn
from pi_ml.utilities import _OMEGACONF_AVAILABLE
from pi_ml.utilities.cloud_io import get_filesystem
from pi_ml.utilities.exceptions import MisconfigurationException
from pi_ml.utilities.imports import _fault_tolerant_training
from pi_ml.utilities.migration import pl_legacy_patch
from pi_ml.utilities.rank_zero import rank_zero_deprecation, rank_zero_info
from pi_ml.utilities.types import _PATH
from pi_ml.utilities.upgrade_checkpoint import KEYS_MAPPING as DEPRECATED_CHECKPOINT_KEYS

if _OMEGACONF_AVAILABLE:
    from omegaconf import Container


log: logging.Logger = logging.getLogger(__name__)


class CheckpointConnector:
    def __init__(self, trainer: "pl.Trainer", resume_from_checkpoint: Optional[_PATH] = None) -> None:
        self.trainer = trainer
        self.resume_checkpoint_path: Optional[_PATH] = None
        # TODO: remove resume_from_checkpoint_fit_path in v2.0
        self.resume_from_checkpoint_fit_path: Optional[_PATH] = resume_from_checkpoint
        if resume_from_checkpoint is not None:
            rank_zero_deprecation(
                "Setting `Trainer(resume_from_checkpoint=)` is deprecated in v1.5 and"
                " will be removed in v1.7. Please pass `Trainer.fit(ckpt_path=)` directly instead."
            )
        self._loaded_checkpoint: Dict[str, Any] = {}

    @property
    def _hpc_resume_path(self) -> Optional[str]:
        weights_save_path = self.trainer.weights_save_path
        fs = get_filesystem(weights_save_path)
        if not fs.isdir(weights_save_path):
            return None
        dir_path_hpc = str(weights_save_path)
        max_version = self.__max_ckpt_version_in_folder(dir_path_hpc, "hpc_ckpt_")
        if max_version is not None:
            return os.path.join(dir_path_hpc, f"hpc_ckpt_{max_version}.ckpt")

    @property
    def _fault_tolerant_auto_resume_path(self) -> Optional[str]:
        auto_saved_path = os.path.join(str(self.trainer.weights_save_path), ".pl_auto_save.ckpt")
        fs = get_filesystem(auto_saved_path)
        return auto_saved_path if fs.exists(auto_saved_path) else None

    def resume_start(self, checkpoint_path: Optional[_PATH] = None) -> None:
        """Attempts to pre-load the checkpoint file to memory, with the source path determined in this priority:

        1. from HPC weights if found
        2. from fault-tolerant auto-saved checkpoint if found
        3. from `checkpoint_path` file if provided
        4. don't restore
        """
        self.resume_checkpoint_path = self._hpc_resume_path or self._fault_tolerant_auto_resume_path or checkpoint_path
        checkpoint_path = self.resume_checkpoint_path
        if not checkpoint_path:
            log.detail("`checkpoint_path` not specified. Skipping checkpoint loading.")
            return

        rank_zero_info(f"Restoring states from the checkpoint path at {checkpoint_path}")
        self._loaded_checkpoint = self._load_and_validate_checkpoint(checkpoint_path)

    def _load_and_validate_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        with pl_legacy_patch():
            loaded_checkpoint = self.trainer.strategy.load_checkpoint(checkpoint_path)
        if any(key in loaded_checkpoint for key in DEPRECATED_CHECKPOINT_KEYS):
            raise ValueError(
                "The checkpoint you're attempting to load follows an"
                " outdated schema. You can upgrade to the current schema by running"
                " `python -m pi_ml.utilities.upgrade_checkpoint --file model.ckpt`"
                " where `model.ckpt` is your checkpoint file."
            )
        return loaded_checkpoint

    def resume_end(self) -> None:
        """Signal the connector that all states have resumed and memory for the checkpoint object can be
        released."""
        assert self.trainer.state.fn is not None
        if self.resume_checkpoint_path:
            if self.trainer.state.fn == TrainerFn.FITTING:
                rank_zero_info(f"Restored all states from the checkpoint file at {self.resume_checkpoint_path}")
            elif self.trainer.state.fn in (TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING):
                rank_zero_info(f"Loaded model weights from checkpoint at {self.resume_checkpoint_path}")
        # TODO: remove resume_from_checkpoint_fit_path in v2.0
        if (
            self.trainer.state.fn == TrainerFn.FITTING
            and self.resume_checkpoint_path == self.resume_from_checkpoint_fit_path
        ):
            self.resume_from_checkpoint_fit_path = None
        self.resume_checkpoint_path = None
        self._loaded_checkpoint = {}

        # clear cache after restore
        torch.cuda.empty_cache()

        # wait for all to catch up
        self.trainer.strategy.barrier("CheckpointConnector.resume_end")

    def restore(self, checkpoint_path: Optional[_PATH] = None) -> None:
        """Attempt to restore everything at once from a 'PyTorch-Lightning checkpoint' file through file-read and
        state-restore, in this priority:

        1. from HPC weights if found
        2. from `checkpoint_path` file if provided
        3. don't restore

        All restored states are listed in return value description of `dump_checkpoint`.

        Args:
            checkpoint_path: Path to a PyTorch Lightning checkpoint file.
        """
        self.resume_start(checkpoint_path)

        # restore module states
        self.restore_datamodule()
        self.restore_model()

        # restore callback states
        self.restore_callbacks()

        # restore training state
        self.restore_training_state()
        self.resume_end()

    def restore_datamodule(self) -> None:
        """Calls hooks on the datamodule to give it a chance to restore its state from the checkpoint."""
        if not self._loaded_checkpoint:
            return

        datamodule = self.trainer.datamodule
        if datamodule is not None:
            datamodule.on_load_checkpoint(self._loaded_checkpoint)
            if datamodule.__class__.__qualname__ in self._loaded_checkpoint:
                datamodule.load_state_dict(self._loaded_checkpoint[datamodule.__class__.__qualname__])

    def restore_model(self) -> None:
        """Restores a model's weights from a PyTorch Lightning checkpoint.

        Hooks are called first to give the LightningModule a chance to modify the contents, then finally the model gets
        updated with the loaded weights.
        """
        if not self._loaded_checkpoint:
            return

        model = self.trainer.lightning_module

        # hook: give user access to checkpoint if needed.
        model.on_load_checkpoint(self._loaded_checkpoint)

        # TODO: remove this in v1.8.
        # call hpc specific hook
        if self._hpc_resume_path is not None:
            model.on_hpc_load(self._loaded_checkpoint)

        # restore model state_dict
        self.trainer.strategy.load_model_state_dict(self._loaded_checkpoint)

        # reset metrics states on non-rank 0 as all states have been accumulated on rank 0 via syncing on checkpointing.
        if not self.trainer.is_global_zero:
            for module in self.trainer.lightning_module.modules():
                if isinstance(module, Metric):
                    module.reset()

    def restore_training_state(self) -> None:
        """Restore the trainer state from the pre-loaded checkpoint.

        This includes the precision settings, loop progress, optimizer states and learning rate scheduler states.
        """
        if not self._loaded_checkpoint:
            return

        # restore precision plugin (scaler etc.)
        self.trainer.precision_plugin.on_load_checkpoint(self._loaded_checkpoint)

        # restore loops and their progress
        self.restore_loops()

        assert self.trainer.state.fn is not None
        if self.trainer.state.fn == TrainerFn.FITTING:
            # restore optimizers and schedulers state
            self.restore_optimizers_and_schedulers()

    def restore_callbacks(self) -> None:
        """Restores all callbacks from the pre-loaded checkpoint."""
        if not self._loaded_checkpoint:
            return

        self.trainer._call_callbacks_on_load_checkpoint(self._loaded_checkpoint)

    def restore_loops(self) -> None:
        """Restores the loop progress from the pre-loaded checkpoint.

        Calls hooks on the loops to give it a chance to restore its state from the checkpoint.
        """
        if not self._loaded_checkpoint:
            return

        self.trainer.fit_loop.global_step = self._loaded_checkpoint["global_step"]
        # set the `current_epoch` value for old checkpoints without the progress tracking state.
        # it will be overwritten by the loop's state if it was also saved
        self.trainer.fit_loop.epoch_progress.current.completed = self._loaded_checkpoint["epoch"]

        assert self.trainer.state.fn is not None
        state_dict = self._loaded_checkpoint.get("loops")
        if state_dict is not None:
            if self.trainer.state.fn == TrainerFn.FITTING:
                self.trainer.fit_loop.load_state_dict(state_dict["fit_loop"])
            elif self.trainer.state.fn == TrainerFn.VALIDATING:
                self.trainer.validate_loop.load_state_dict(state_dict["validate_loop"])
            elif self.trainer.state.fn == TrainerFn.TESTING:
                self.trainer.test_loop.load_state_dict(state_dict["test_loop"])
            elif self.trainer.state.fn == TrainerFn.PREDICTING:
                self.trainer.predict_loop.load_state_dict(state_dict["predict_loop"])

        if self.trainer.state.fn != TrainerFn.FITTING:
            return

        # crash if max_epochs is lower then the current epoch from the checkpoint
        if (
            self.trainer.max_epochs != -1
            and self.trainer.max_epochs is not None
            and self.trainer.current_epoch > self.trainer.max_epochs
        ):
            raise MisconfigurationException(
                f"You restored a checkpoint with current_epoch={self.trainer.current_epoch},"
                f" but you have set Trainer(max_epochs={self.trainer.max_epochs})."
            )

    def restore_optimizers_and_schedulers(self) -> None:
        """Restores the optimizers and learning rate scheduler states from the pre-loaded checkpoint."""
        if not self._loaded_checkpoint:
            return

        if self.trainer.strategy.lightning_restore_optimizer:
            # validation
            if "optimizer_states" not in self._loaded_checkpoint:
                raise KeyError(
                    "Trying to restore optimizer state but checkpoint contains only the model."
                    " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
                )
            self.restore_optimizers()

        if "lr_schedulers" not in self._loaded_checkpoint:
            raise KeyError(
                "Trying to restore learning rate scheduler state but checkpoint contains only the model."
                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
            )
        self.restore_lr_schedulers()

    def restore_optimizers(self) -> None:
        """Restores the optimizer states from the pre-loaded checkpoint."""
        if not self._loaded_checkpoint:
            return

        # restore the optimizers
        self.trainer.strategy.load_optimizer_state_dict(self._loaded_checkpoint)
        for optimizer in self.trainer.optimizers:
            # move optimizer to GPU 1 weight at a time
            # avoids OOM
            if self.trainer.root_gpu is not None:
                for param, state in optimizer.state.items():
                    if isinstance(state, dict):
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.trainer.root_gpu)
                    elif isinstance(state, torch.Tensor):
                        optimizer.state[param] = state.cuda(self.trainer.root_gpu)

    def restore_lr_schedulers(self) -> None:
        """Restores the learning rate scheduler states from the pre-loaded checkpoint."""
        if not self._loaded_checkpoint:
            return

        # restore the lr schedulers
        lr_schedulers = self._loaded_checkpoint["lr_schedulers"]
        for config, lrs_state in zip(self.trainer.lr_scheduler_configs, lr_schedulers):
            config.scheduler.load_state_dict(lrs_state)

    # ----------------------------------
    # PRIVATE OPS
    # ----------------------------------

    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        """Creating a model checkpoint dictionary object from various component states.
        Args:
            weights_only: saving model weights only
        Return:
            structured dictionary: {
                'epoch':                     training epoch
                'global_step':               training global step
                'pytorch-lightning_version': The version of PyTorch Lightning that produced this checkpoint
                'callbacks':                 "callback specific state"[] # if not weights_only
                'optimizer_states':          "PT optim's state_dict"[]   # if not weights_only
                'lr_schedulers':             "PT sched's state_dict"[]   # if not weights_only
                'native_amp_scaling_state':  PT amp's state_dict         # if not weights_only and use native amp
                'amp_scaling_state':         Apex's state_dict           # if not weights_only and use apex amp
                'state_dict':                Model's state_dict (e.g. network weights)
                CHECKPOINT_HYPER_PARAMS_NAME:
                CHECKPOINT_HYPER_PARAMS_KEY:
                CHECKPOINT_HYPER_PARAMS_TYPE:
                something_cool_i_want_to_save: anything you define through model.on_save_checkpoint
                LightningDataModule.__class__.__qualname__: pl DataModule's state
            }
        """

        # dump epoch/global_step/pytorch-lightning_version
        current_epoch = self.trainer.current_epoch
        global_step = self.trainer.global_step
        has_reached_max_steps = _is_max_limit_reached(global_step, self.trainer.max_steps)

        global_step += 1
        if not has_reached_max_steps:
            current_epoch += 1

        model = self.trainer.lightning_module

        checkpoint = {
            "epoch": current_epoch,
            "global_step": global_step,
            "pytorch-lightning_version": pl.__version__,
            "state_dict": self._get_lightning_module_state_dict(),
            "loops": self._get_loops_state_dict(),
        }

        if not weights_only:
            # dump callbacks
            checkpoint["callbacks"] = self.trainer._call_callbacks_on_save_checkpoint(checkpoint)

            optimizer_states = []
            for i, optimizer in enumerate(self.trainer.optimizers):
                # Rely on accelerator to dump optimizer state
                optimizer_state = self.trainer.strategy.optimizer_state(optimizer)
                optimizer_states.append(optimizer_state)

            checkpoint["optimizer_states"] = optimizer_states

            # dump lr schedulers
            lr_schedulers = []
            for config in self.trainer.lr_scheduler_configs:
                lr_schedulers.append(config.scheduler.state_dict())
            checkpoint["lr_schedulers"] = lr_schedulers

            self.trainer.precision_plugin.on_save_checkpoint(checkpoint)

        # dump hyper-parameters
        if model.hparams:
            if hasattr(model, "_hparams_name"):
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_NAME] = model._hparams_name
            # dump arguments
            if _OMEGACONF_AVAILABLE and isinstance(model.hparams, Container):
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY] = model.hparams
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_TYPE] = type(model.hparams)
            else:
                checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY] = dict(model.hparams)

        # dump stateful datamodule
        datamodule = self.trainer.datamodule
        if datamodule is not None:
            datamodule_state_dict = datamodule.state_dict()
            if datamodule_state_dict:
                checkpoint[datamodule.__class__.__qualname__] = datamodule_state_dict

        # on_save_checkpoint hooks
        model.on_save_checkpoint(checkpoint)
        if datamodule is not None:
            datamodule.on_save_checkpoint(checkpoint)

        # TODO: remove this in v1.8.
        environment = self.trainer._accelerator_connector.cluster_environment
        if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
            model.on_hpc_save(checkpoint)

        return checkpoint

    def save_checkpoint(self, filepath: _PATH, weights_only: bool = False) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            filepath: write-target file's path
            weights_only: saving model weights only
        """
        _checkpoint = self.dump_checkpoint(weights_only)
        self.trainer.strategy.save_checkpoint(_checkpoint, filepath)

    def _get_lightning_module_state_dict(self) -> Dict[str, torch.Tensor]:
        metrics = (
            [m for m in self.trainer.lightning_module.modules() if isinstance(m, Metric)]
            if _fault_tolerant_training()
            else []
        )

        for metric in metrics:
            metric.persistent(True)
            metric.sync()

        state_dict = self.trainer.strategy.lightning_module_state_dict()

        for metric in metrics:
            # sync can be a no-op (e.g. on cpu) so `unsync` would raise a user error exception if we don't check
            if metric._is_synced:
                metric.unsync()

        return state_dict

    def _get_loops_state_dict(self) -> Dict[str, Any]:
        return {
            "fit_loop": self.trainer.fit_loop.state_dict(),
            "validate_loop": self.trainer.validate_loop.state_dict(),
            "test_loop": self.trainer.test_loop.state_dict(),
            "predict_loop": self.trainer.predict_loop.state_dict(),
        }

    @staticmethod
    def __max_ckpt_version_in_folder(dir_path: _PATH, name_key: str = "ckpt_") -> Optional[int]:
        """List up files in `dir_path` with `name_key`, then yield maximum suffix number.

        Args:
            dir_path: path of directory which may contain files whose name include `name_key`
            name_key: file name prefix
        Returns:
            None if no-corresponding-file else maximum suffix number
        """

        # check directory existence
        fs = get_filesystem(dir_path)
        if not fs.exists(dir_path):
            return None

        # check corresponding file existence
        files = [os.path.basename(f["name"]) for f in fs.listdir(dir_path)]
        files = [x for x in files if name_key in x]
        if len(files) == 0:
            return None

        # extract suffix number
        ckpt_vs = []
        for name in files:
            name = name.split(name_key)[-1]
            name = re.sub("[^0-9]", "", name)
            ckpt_vs.append(int(name))

        return max(ckpt_vs)

    @staticmethod
    def __get_max_ckpt_path_from_folder(folder_path: _PATH) -> str:
        """Get path of maximum-epoch checkpoint in the folder."""

        max_suffix = CheckpointConnector.__max_ckpt_version_in_folder(folder_path)
        ckpt_number = max_suffix if max_suffix is not None else 0
        return f"{folder_path}/hpc_ckpt_{ckpt_number}.ckpt"

    @staticmethod
    def hpc_save_path(folderpath: _PATH) -> str:
        max_suffix = CheckpointConnector.__max_ckpt_version_in_folder(folderpath)
        ckpt_number = (max_suffix if max_suffix is not None else 0) + 1
        filepath = os.path.join(folderpath, f"hpc_ckpt_{ckpt_number}.ckpt")
        return filepath
