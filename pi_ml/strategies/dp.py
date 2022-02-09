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
from typing import Any, List, Optional

import torch
from torch.nn import DataParallel, Module

import pi_ml as pl
from pi_ml.overrides.data_parallel import LightningParallelModule
from pi_ml.plugins.io.checkpoint_plugin import CheckpointIO
from pi_ml.plugins.precision import PrecisionPlugin
from pi_ml.strategies.parallel import ParallelStrategy
from pi_ml.utilities.apply_func import apply_to_collection
from pi_ml.utilities.enums import _StrategyType
from pi_ml.utilities.model_helpers import is_overridden
from pi_ml.utilities.types import _METRIC_COLLECTION, STEP_OUTPUT


class DataParallelStrategy(ParallelStrategy):
    """Implements data-parallel training in a single process, i.e., the model gets replicated to each device and
    each gets a split of the data."""

    distributed_backend = _StrategyType.DP

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=None,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def node_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def setup(self, trainer: "pl.Trainer") -> None:
        # model needs to be moved to the device before it is wrapped
        self.model_to_device()
        self.model = self._setup_model(LightningParallelModule(self.model))
        super().setup(trainer)

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        """Moves the batch to the correct device.

        The input and the output is the same type.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
            dataloader_idx: The index of the dataloader to which the batch belongs.
        """
        # DataParallel handles the transfer of batch to the device
        return batch

    def _setup_model(self, model: Module) -> DataParallel:
        """Wraps the given model into a :class:`~torch.nn.parallel.DataParallel` module."""
        return DataParallel(module=model, device_ids=self.parallel_devices)

    def reduce(self, collection: _METRIC_COLLECTION, *args, **kwargs) -> _METRIC_COLLECTION:
        """Reduces a collection of tensors from all processes. It can be applied to just a single tensor.

        Args:
            collection: The collection of tensors to sync and reduce.
            *args: ignored for DP
            **kwargs: ignored for DP

        Return:
            Reduced tensor values or the same value if it was not or did not contain a tensor.
        """

        def mean(t: torch.Tensor) -> torch.Tensor:
            original_dtype = t.dtype
            return t.float().mean().to(original_dtype)

        return apply_to_collection(collection, torch.Tensor, mean)

    @property
    def root_device(self):
        return self.parallel_devices[0]

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    def barrier(self, *args, **kwargs):
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj

    def reduce_boolean_decision(self, decision: bool) -> bool:
        return decision

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self.model(*args, **kwargs)

    def training_step_end(self, output):
        if is_overridden("training_step_end", self.lightning_module):
            return output

        if isinstance(output, dict) and "loss" in output:
            output["loss"] = self.reduce(output["loss"])

        elif isinstance(output, torch.Tensor):
            output = self.reduce(output)

        return output

    def teardown(self) -> None:
        super().teardown()
        if self.root_device.type == "cuda":
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
