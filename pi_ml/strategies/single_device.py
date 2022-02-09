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
from __future__ import annotations

from typing import Any

import torch

import pi_ml as pl
from pi_ml.plugins.io.checkpoint_plugin import CheckpointIO
from pi_ml.plugins.precision import PrecisionPlugin
from pi_ml.strategies.strategy import Strategy
from pi_ml.utilities.types import _DEVICE


class SingleDeviceStrategy(Strategy):
    """Strategy that handles communication on a single device."""

    def __init__(
        self,
        device: _DEVICE,
        accelerator: pl.accelerators.accelerator.Accelerator | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: PrecisionPlugin | None = None,
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        self._root_device = torch.device(device)
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    def reduce(self, tensor: Any | torch.Tensor, *args: Any, **kwargs: Any) -> Any | torch.Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor. As this plugin only
        operates with a single device, the reduction is simply the identity.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation
        """
        return tensor

    def all_gather(self, tensor: torch.Tensor, group: Any | None = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes."""
        return tensor

    @property
    def root_device(self) -> torch.device:
        return self._root_device

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    def setup(self, trainer: pl.Trainer) -> None:
        self.model_to_device()
        super().setup(trainer)

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args, **kwargs) -> None:
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj

    def teardown(self) -> None:
        super().teardown()
        if self.root_device.type == "cuda":
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
