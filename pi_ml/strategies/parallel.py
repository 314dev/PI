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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel

import pi_ml as pl
from pi_ml.overrides.base import unwrap_lightning_module
from pi_ml.plugins.environments.cluster_environment import ClusterEnvironment
from pi_ml.plugins.io.checkpoint_plugin import CheckpointIO
from pi_ml.plugins.precision import PrecisionPlugin
from pi_ml.strategies.strategy import Strategy
from pi_ml.utilities.distributed import all_gather_ddp_if_available, ReduceOp


class ParallelStrategy(Strategy, ABC):
    """Plugin for training with multiple processes in parallel."""

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        self.parallel_devices = parallel_devices
        self.cluster_environment = cluster_environment

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        """Return the root device."""

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        return unwrap_lightning_module(self.model) if self.model is not None else None

    @property
    def global_rank(self) -> int:
        return self.cluster_environment.global_rank() if self.cluster_environment is not None else 0

    @property
    def local_rank(self) -> int:
        return self.cluster_environment.local_rank() if self.cluster_environment is not None else 0

    @property
    def node_rank(self) -> int:
        return self.cluster_environment.node_rank() if self.cluster_environment is not None else 0

    @property
    def world_size(self) -> int:
        return self.cluster_environment.world_size() if self.cluster_environment is not None else 1

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def parallel_devices(self):
        return self._parallel_devices

    @parallel_devices.setter
    def parallel_devices(self, parallel_devices):
        self._parallel_devices = parallel_devices

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=len(self.parallel_devices), rank=self.global_rank)
        return distributed_sampler_kwargs

    def reconciliate_processes(self, trace: str):
        """Function to re-conciliate processes on failure."""

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes."""
        return all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    def reduce_boolean_decision(self, decision: bool) -> bool:
        decision = torch.tensor(int(decision), device=self.root_device)
        decision = self.reduce(decision, reduce_op=ReduceOp.SUM)
        decision = bool(decision == self.world_size)
        return decision

    @property
    def torch_distributed_backend(self):
        torch_backend = os.getenv("PL_TORCH_DISTRIBUTED_BACKEND")
        if torch_backend is None:
            torch_backend = "nccl" if self.root_device.type == "cuda" else "gloo"
        return torch_backend

    @staticmethod
    def configure_sync_batchnorm(model: "pl.LightningModule") -> "pl.LightningModule":
        """Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    @contextmanager
    def block_backward_sync(self):
        """Blocks ddp sync gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, DistributedDataParallel):
            with self.model.no_sync():
                yield None
        else:
            yield None

    def teardown(self) -> None:
        self.cluster_environment.teardown()
        super().teardown()
