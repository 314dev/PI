from typing import Union

from pi_ml.plugins.environments import ClusterEnvironment
from pi_ml.plugins.io.checkpoint_plugin import CheckpointIO
from pi_ml.plugins.io.torch_plugin import TorchCheckpointIO
from pi_ml.plugins.io.xla_plugin import XLACheckpointIO
from pi_ml.plugins.precision.apex_amp import ApexMixedPrecisionPlugin
from pi_ml.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from pi_ml.plugins.precision.double import DoublePrecisionPlugin
from pi_ml.plugins.precision.fully_sharded_native_amp import FullyShardedNativeMixedPrecisionPlugin
from pi_ml.plugins.precision.ipu import IPUPrecisionPlugin
from pi_ml.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pi_ml.plugins.precision.precision_plugin import PrecisionPlugin
from pi_ml.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from pi_ml.plugins.precision.tpu import TPUPrecisionPlugin
from pi_ml.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin
from pi_ml.plugins.training_type.ddp import DDPPlugin
from pi_ml.plugins.training_type.ddp2 import DDP2Plugin
from pi_ml.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pi_ml.plugins.training_type.deepspeed import DeepSpeedPlugin
from pi_ml.plugins.training_type.dp import DataParallelPlugin
from pi_ml.plugins.training_type.fully_sharded import DDPFullyShardedPlugin
from pi_ml.plugins.training_type.horovod import HorovodPlugin
from pi_ml.plugins.training_type.ipu import IPUPlugin
from pi_ml.plugins.training_type.parallel import ParallelPlugin
from pi_ml.plugins.training_type.sharded import DDPShardedPlugin
from pi_ml.plugins.training_type.sharded_spawn import DDPSpawnShardedPlugin
from pi_ml.plugins.training_type.single_device import SingleDevicePlugin
from pi_ml.plugins.training_type.single_tpu import SingleTPUPlugin
from pi_ml.plugins.training_type.tpu_spawn import TPUSpawnPlugin
from pi_ml.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pi_ml.strategies import Strategy

PLUGIN = Union[Strategy, PrecisionPlugin, ClusterEnvironment, CheckpointIO]
PLUGIN_INPUT = Union[PLUGIN, str]

__all__ = [
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "ApexMixedPrecisionPlugin",
    "DataParallelPlugin",
    "DDP2Plugin",
    "DDPPlugin",
    "DDPSpawnPlugin",
    "DDPFullyShardedPlugin",
    "DeepSpeedPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "HorovodPlugin",
    "IPUPlugin",
    "IPUPrecisionPlugin",
    "NativeMixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "FullyShardedNativeMixedPrecisionPlugin",
    "SingleDevicePlugin",
    "SingleTPUPlugin",
    "TPUPrecisionPlugin",
    "TPUBf16PrecisionPlugin",
    "TPUSpawnPlugin",
    "TrainingTypePlugin",
    "ParallelPlugin",
    "DDPShardedPlugin",
    "DDPSpawnShardedPlugin",
]
