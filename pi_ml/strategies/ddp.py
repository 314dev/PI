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
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Union

import __main__
import numpy as np
import torch
import torch.distributed
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

import pi_ml as pl
from pi_ml.core.optimizer import LightningOptimizer
from pi_ml.overrides import LightningDistributedModule
from pi_ml.overrides.distributed import prepare_for_backward
from pi_ml.plugins.environments.cluster_environment import ClusterEnvironment
from pi_ml.plugins.io.checkpoint_plugin import CheckpointIO
from pi_ml.plugins.precision import PrecisionPlugin
from pi_ml.strategies.parallel import ParallelStrategy
from pi_ml.trainer.states import TrainerFn
from pi_ml.utilities import (
    _FAIRSCALE_AVAILABLE,
    _HYDRA_AVAILABLE,
    _IS_WINDOWS,
    _TORCH_GREATER_EQUAL_1_8,
    _TORCH_GREATER_EQUAL_1_9,
    _TORCH_GREATER_EQUAL_1_10,
)
from pi_ml.utilities.distributed import _revert_sync_batchnorm, distributed_available
from pi_ml.utilities.distributed import group as _group
from pi_ml.utilities.distributed import init_dist_connection, ReduceOp, sync_ddp_if_available
from pi_ml.utilities.enums import _StrategyType
from pi_ml.utilities.exceptions import DeadlockDetectedException
from pi_ml.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pi_ml.utilities.seed import reset_seed
from pi_ml.utilities.types import STEP_OUTPUT

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS
if _HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path
if _TORCH_GREATER_EQUAL_1_8:
    from pi_ml.utilities.distributed import register_ddp_comm_hook


log = logging.getLogger(__name__)


class DDPStrategy(ParallelStrategy):
    """Plugin for multi-process single-device training on one or multiple nodes.

    The main process in each node spawns N-1 child processes via :func:`subprocess.Popen`, where N is the number of
    devices (e.g. GPU) per node. It is very similar to how :mod:`torch.distributed.launch` launches processes.
    """

    distributed_backend = _StrategyType.DDP

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        model_averaging_period: Optional[int] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        log.detail(f"{self.__class__.__name__}: initializing DDP plugin")
        self.interactive_ddp_procs = []
        self._num_nodes = 1
        self.sync_batchnorm = False
        self._ddp_kwargs = kwargs
        self._ddp_comm_state = ddp_comm_state
        self._ddp_comm_hook = ddp_comm_hook
        self._ddp_comm_wrapper = ddp_comm_wrapper
        self._model_averaging_period = model_averaging_period
        self._pids: Optional[List[int]] = None
        self._sync_dir: Optional[str] = None
        self._rank_0_has_called_call_children_scripts: bool = False
        self.set_world_ranks()

    @property
    def is_distributed(self) -> bool:
        return True

    @property
    def root_device(self) -> torch.device:
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes
        self.set_world_ranks()

    @property
    def num_processes(self):
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self) -> bool:
        return True

    def setup_environment(self) -> None:
        # start the other scripts
        if not self.cluster_environment.creates_processes_externally:
            self._call_children_scripts()

        self.setup_distributed()
        super().setup_environment()

    def setup(self, trainer: "pl.Trainer") -> None:
        super().setup(trainer)
        # share ddp pids to all processes
        self._rank_0_has_called_call_children_scripts = self.broadcast(self._rank_0_has_called_call_children_scripts)
        if self._should_run_deadlock_detection():
            self._share_information_to_prevent_deadlock()

        # move the model to the correct device
        self.model_to_device()

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = self.lightning_module.trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            self.configure_ddp()

    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        device_ids = self.determine_ddp_device_ids()
        log.detail(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
        return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)

    def _call_children_scripts(self):
        # bookkeeping of spawned processes
        self._check_can_spawn_children()

        # DDP Environment variables
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)

        # allow the user to pass the node rank
        os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
        os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())

        # Check if the current calling command looked like `python a/b/c.py` or `python -m a.b.c`
        # See https://docs.python.org/3/reference/import.html#main-spec
        if __main__.__spec__ is None:  # pragma: no-cover
            # Script called as `python a/b/c.py`
            # when user is using hydra find the absolute path
            path_lib = os.path.abspath if not _HYDRA_AVAILABLE else to_absolute_path

            # pull out the commands used to run the script and resolve the abs file path
            command = sys.argv
            try:
                full_path = path_lib(command[0])
            except Exception:
                full_path = os.path.abspath(command[0])

            command[0] = full_path
            # use the same python interpreter and actually running
            command = [sys.executable] + command
        else:  # Script called as `python -m a.b.c`
            command = [sys.executable, "-m", __main__.__spec__.name] + sys.argv[1:]

        os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

        self.interactive_ddp_procs = []

        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if os.environ.get("PL_GLOBAL_SEED") is None and "PL_GLOBAL_SEED" in env_copy:
                del env_copy["PL_GLOBAL_SEED"]

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if _HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
                    os_cwd = f'"{os.getcwd()}"'
                    command += [f"hydra.run.dir={os_cwd}", f"hydra.job.name=train_ddp_process_{local_rank}"]
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

        self._rank_0_has_called_call_children_scripts = True

    def setup_distributed(self):
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        init_dist_connection(self.cluster_environment, self.torch_distributed_backend)

    def _check_can_spawn_children(self):
        if self.local_rank != 0:
            raise RuntimeError(
                "Lightning attempted to launch new distributed processes with `local_rank > 0`. This should not happen."
                " Possible reasons: 1) LOCAL_RANK environment variable was incorrectly modified by the user,"
                " 2) `ClusterEnvironment.creates_processes_externally` incorrectly implemented."
            )

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def pre_configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        if not self.lightning_module.automatic_optimization and not self._ddp_kwargs.get(
            "find_unused_parameters", False
        ):
            # TODO: PyTorch 1.7.0 DDP introduces `self.reducer._rebuild_buckets()` breaking manual_optimization
            rank_zero_warn(
                "From PyTorch 1.7.0, Lightning `manual_optimization` needs to set `find_unused_parameters=True` to"
                " properly work with DDP. Using `find_unused_parameters=True`."
            )
            self._ddp_kwargs["find_unused_parameters"] = True

    def _register_ddp_hooks(self) -> None:
        log.detail(f"{self.__class__.__name__}: registering ddp hooks")
        # In 1.8, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # Since 1.9, DDP communication hooks can work on all backends.
        if _TORCH_GREATER_EQUAL_1_9 or (
            _TORCH_GREATER_EQUAL_1_8 and self.root_device.type == "cuda" and self._is_single_process_single_device
        ):
            register_ddp_comm_hook(
                model=self.model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

            if _TORCH_GREATER_EQUAL_1_10 and self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
                import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

                if isinstance(self._ddp_comm_state, post_localSGD.PostLocalSGDState):
                    self._reinit_optimizers_with_post_localSGD(self._ddp_comm_state.start_localSGD_iter)

    def _reinit_optimizers_with_post_localSGD(self, warmup_steps: int):
        log.detail(f"{self.__class__.__name__}: reinitializing optimizers with post localSGD")
        optimizers = self.optimizers
        if self._model_averaging_period is None:
            raise ValueError(
                "Post-localSGD algorithm is used, but model averaging period is not provided to DDP strategy."
            )
        if _TORCH_GREATER_EQUAL_1_10:
            if not _IS_WINDOWS:
                from torch.distributed.optim import DistributedOptimizer
            import torch.distributed.algorithms.model_averaging.averagers as averagers
            from torch.distributed.optim import PostLocalSGDOptimizer, ZeroRedundancyOptimizer

        averager = averagers.PeriodicModelAverager(period=self._model_averaging_period, warmup_steps=warmup_steps)
        for x, optimizer in enumerate(optimizers):
            if isinstance(optimizer, LightningOptimizer):
                optimizer = optimizer._optimizer

            is_distributed_optimizer = isinstance(optimizer, DistributedOptimizer) if not _IS_WINDOWS else False
            if (
                is_distributed_optimizer
                or isinstance(optimizer, ZeroRedundancyOptimizer)
                or (_FAIRSCALE_AVAILABLE and isinstance(optimizer, OSS))
            ):
                raise ValueError(
                    f"Cannot wrap a distributed optimizer of type {optimizer.__name__} by PostLocalSGDOptimizer."
                )

            if isinstance(optimizer, PostLocalSGDOptimizer):
                continue

            optim_class = type(optimizer)
            post_localSGD_optimizer = PostLocalSGDOptimizer(
                params=optimizer.param_groups,
                optimizer_class=optim_class,
                averager=averager,
                **optimizer.defaults,
            )
            optimizers[x] = post_localSGD_optimizer
            del optimizer
        self.optimizers = optimizers

    def configure_ddp(self) -> None:
        log.detail(f"{self.__class__.__name__}: configuring DistributedDataParallel")
        self.pre_configure_ddp()
        self.model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def barrier(self, *args, **kwargs) -> None:
        if not distributed_available():
            return
        if _TORCH_GREATER_EQUAL_1_8 and torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        obj = [obj]
        if self.global_rank != src:
            obj = [None]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        """Run before precision plugin executes backward."""
        if not self.lightning_module.automatic_optimization:
            prepare_for_backward(self.model, closure_loss)

    def model_to_device(self):
        log.detail(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        self.model.to(self.root_device)

    def reduce(self, tensor, group: Optional[Any] = None, reduce_op: Union[ReduceOp, str] = "mean") -> torch.Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if isinstance(tensor, torch.Tensor):
            tensor = sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            if isinstance(self.model, DistributedDataParallel):
                # used when calling `trainer.fit`
                return self.model(*args, **kwargs)
            else:
                # used when calling `trainer.validate`
                return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self.lightning_module.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self.lightning_module.predict_step(*args, **kwargs)

    def post_training_step(self):
        if not self.lightning_module.automatic_optimization:
            self.model.require_backward_grad_sync = True

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_find_unused_parameters_false",
            cls,
            description="DDP Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )

    def _should_run_deadlock_detection(self) -> bool:
        """Determines whether the plugin will perform process reconciliation in case of errors.

        If the environment variable `PL_RECONCILE_PROCESS` is set, run detection regardless of the cluster environment.
        By default this is disabled. Otherwise, if the cluster environment creates the processes, allow the scheduler /
        parent process to perform the process termination, external to Lightning.
        """
        return os.getenv("PL_RECONCILE_PROCESS", "0") == "1" or self._rank_0_has_called_call_children_scripts

    def _share_information_to_prevent_deadlock(self) -> None:
        self._share_pids()

        # there should be a unique sync_dir per nodes.
        if self.local_rank == 0:
            # create a temporary directory used to synchronize processes on deadlock.
            self._sync_dir = tempfile.mkdtemp()

        sync_dirs = []
        global_node_rank_zero = 0
        for _ in range(self.num_nodes):
            sync_dirs.append(self.broadcast(self._sync_dir, global_node_rank_zero))
            global_node_rank_zero += self.world_size // self.num_nodes

        self._sync_dir = sync_dirs[self.node_rank]

    def _share_pids(self) -> None:
        """Make all DDP processes aware of all processes pids."""
        self.barrier()
        pids = self.all_gather(torch.tensor(os.getpid(), device=self.root_device))
        pids = pids.cpu().numpy().tolist()
        self._pids = pids if isinstance(pids, list) else [pids]

    def reconciliate_processes(self, trace: str) -> None:
        if self.world_size < 2:
            return

        if not self._should_run_deadlock_detection():
            return

        sync_dir = self._sync_dir

        if not sync_dir:
            rank_zero_warn("Error handling mechanism for deadlock detection is uninitialized. Skipping check.")
            return

        # The cluster may be configured to periodically purge the `/tmp`
        # directory, in which case `sync_dir` may not exist anymore at this
        # point. Idempotently create it to ensure its existence.
        Path(sync_dir).mkdir(parents=True, exist_ok=True)

        # save a file locally.
        torch.save(True, os.path.join(sync_dir, f"{self.global_rank}.pl"))

        # sleep for a short time
        time.sleep(3)

        # return if all processes wrote a file in the `sync_dir`.
        # todo (tchaton) Add support for non-shared file-system which will fail.
        if len(os.listdir(sync_dir)) == (self.world_size // self.num_nodes):
            return

        for pid in self._pids:
            if pid != os.getpid():
                os.kill(pid, signal.SIGKILL)
        shutil.rmtree(sync_dir)
        raise DeadlockDetectedException(f"DeadLock detected from rank: {self.global_rank} \n {trace}")

    def teardown(self) -> None:
        log.detail(f"{self.__class__.__name__}: tearing down DDP plugin")
        super().teardown()
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.lightning_module

        if self.sync_batchnorm:
            self.model = _revert_sync_batchnorm(self.model)

        if self.root_device.type == "cuda":
            # GPU teardown
            log.detail(f"{self.__class__.__name__}: moving model to CPU")
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
