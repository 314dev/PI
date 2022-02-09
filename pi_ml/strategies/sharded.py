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
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

import pi_ml as pl
from pi_ml.core.optimizer import LightningOptimizer
from pi_ml.strategies.ddp import DDPStrategy
from pi_ml.trainer.states import TrainerFn
from pi_ml.utilities.enums import _StrategyType, PrecisionType
from pi_ml.utilities.exceptions import MisconfigurationException
from pi_ml.utilities.imports import _FAIRSCALE_AVAILABLE, _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE
from pi_ml.utilities.rank_zero import rank_zero_only

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS

    from pi_ml.overrides.fairscale import LightningShardedDataParallel, unwrap_lightning_module_sharded


class DDPShardedStrategy(DDPStrategy):
    """Optimizer and gradient sharded training provided by FairScale."""

    distributed_backend = _StrategyType.DDP_SHARDED
    _REDUCE_BUFFER_SIZE_DEFAULT: int = 2 ** 23  # 8M

    def configure_ddp(self) -> None:
        trainer = self.lightning_module.trainer
        if "reduce_buffer_size" not in self._ddp_kwargs:
            # For multi-node training, enabling bucketing will improve performance.
            self._ddp_kwargs["reduce_buffer_size"] = self._REDUCE_BUFFER_SIZE_DEFAULT if self.num_nodes > 1 else 0

        self.model, self.optimizers = self._setup_model_and_optimizers(
            model=LightningShardedDataParallel(self.model),
            optimizers=trainer.optimizers,
        )

    def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = self._wrap_optimizers(optimizers)
        model = ShardedDataParallel(model, sharded_optimizer=optimizers, **self._ddp_kwargs)
        return model, optimizers

    def _reinit_optimizers_with_oss(self, optimizers: List[Union[Optimizer, LightningOptimizer]]) -> List["OSS"]:
        for x, optimizer in enumerate(optimizers):
            if isinstance(optimizer, LightningOptimizer):
                optimizer = optimizer._optimizer
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                if _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE:
                    is_fp16 = self.precision_plugin.precision in (PrecisionType.MIXED, PrecisionType.HALF)
                    # For multi-node training, compressing the model shards in fp16 before broadcasting
                    # improves performance. When using PyTorch AMP, it will not degrade
                    # the model performance.
                    zero_optimizer.broadcast_fp16 = is_fp16 and self.num_nodes > 1
                optimizers[x] = zero_optimizer
                del optimizer
        return optimizers

    def _wrap_optimizers(self, optimizers: List[Optimizer]) -> List["OSS"]:
        if self.model is not None and self.model.trainer.state.fn != TrainerFn.FITTING:
            return optimizers

        return self._reinit_optimizers_with_oss(optimizers)

    def optimizer_state(self, optimizer: "OSS") -> Optional[dict]:
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer
        optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        """
        Retrieves state dict only on rank 0, which contains the entire optimizer state after calling
        :meth:`consolidate_state_dict`.
        """
        return optimizer.state_dict()

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        if not _FAIRSCALE_AVAILABLE:  # pragma: no cover
            raise MisconfigurationException(
                "`DDPShardedStrategy` requires `fairscale` to be installed."
                " Install it by running `pip install fairscale`."
            )
        return unwrap_lightning_module_sharded(self.model) if self.model is not None else None

    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        pass

    @contextmanager
    def block_backward_sync(self) -> Generator:
        """Blocks syncing gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, ShardedDataParallel):
            with self.model.no_sync():
                yield None
        else:
            yield None

    def post_training_step(self):
        pass

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sharded_find_unused_parameters_false",
            cls,
            description="DDP Sharded Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
