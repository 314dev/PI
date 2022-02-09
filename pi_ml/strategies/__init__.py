from pathlib import Path

from pi_ml.strategies.bagua import BaguaStrategy  # noqa: F401
from pi_ml.strategies.ddp import DDPStrategy  # noqa: F401
from pi_ml.strategies.ddp2 import DDP2Strategy  # noqa: F401
from pi_ml.strategies.ddp_spawn import DDPSpawnStrategy  # noqa: F401
from pi_ml.strategies.deepspeed import DeepSpeedStrategy  # noqa: F401
from pi_ml.strategies.dp import DataParallelStrategy  # noqa: F401
from pi_ml.strategies.fully_sharded import DDPFullyShardedStrategy  # noqa: F401
from pi_ml.strategies.horovod import HorovodStrategy  # noqa: F401
from pi_ml.strategies.ipu import IPUStrategy  # noqa: F401
from pi_ml.strategies.parallel import ParallelStrategy  # noqa: F401
from pi_ml.strategies.sharded import DDPShardedStrategy  # noqa: F401
from pi_ml.strategies.sharded_spawn import DDPSpawnShardedStrategy  # noqa: F401
from pi_ml.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from pi_ml.strategies.single_tpu import SingleTPUStrategy  # noqa: F401
from pi_ml.strategies.strategy import Strategy  # noqa: F401
from pi_ml.strategies.strategy_registry import call_register_strategies, StrategyRegistry  # noqa: F401
from pi_ml.strategies.tpu_spawn import TPUSpawnStrategy  # noqa: F401

FILE_ROOT = Path(__file__).parent
STRATEGIES_BASE_MODULE = "pi_ml.strategies"

call_register_strategies(FILE_ROOT, STRATEGIES_BASE_MODULE)
