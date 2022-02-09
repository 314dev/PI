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
"""General utilities."""

import numpy

from pi_ml.utilities.apply_func import move_data_to_device  # noqa: F401
from pi_ml.utilities.distributed import AllGatherGrad  # noqa: F401
from pi_ml.utilities.enums import (  # noqa: F401
    _AcceleratorType,
    _StrategyType,
    AMPType,
    DistributedType,
    GradClipAlgorithmType,
    LightningEnum,
    ModelSummaryMode,
)
from pi_ml.utilities.grads import grad_norm  # noqa: F401
from pi_ml.utilities.imports import (  # noqa: F401
    _APEX_AVAILABLE,
    _BAGUA_AVAILABLE,
    _DEEPSPEED_AVAILABLE,
    _FAIRSCALE_AVAILABLE,
    _FAIRSCALE_FULLY_SHARDED_AVAILABLE,
    _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE,
    _GROUP_AVAILABLE,
    _HOROVOD_AVAILABLE,
    _HYDRA_AVAILABLE,
    _HYDRA_EXPERIMENTAL_AVAILABLE,
    _IPU_AVAILABLE,
    _IS_INTERACTIVE,
    _IS_WINDOWS,
    _JSONARGPARSE_AVAILABLE,
    _module_available,
    _OMEGACONF_AVAILABLE,
    _POPTORCH_AVAILABLE,
    _RICH_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_8,
    _TORCH_GREATER_EQUAL_1_9,
    _TORCH_GREATER_EQUAL_1_10,
    _TORCH_QUANTIZE_AVAILABLE,
    _TORCHTEXT_AVAILABLE,
    _TORCHVISION_AVAILABLE,
    _TPU_AVAILABLE,
    _XLA_AVAILABLE,
)
from pi_ml.utilities.parameter_tying import find_shared_parameters, set_shared_parameters  # noqa: F401
from pi_ml.utilities.parsing import AttributeDict, flatten_dict, is_picklable  # noqa: F401
from pi_ml.utilities.rank_zero import (  # noqa: F401
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)

FLOAT16_EPSILON = numpy.finfo(numpy.float16).eps
FLOAT32_EPSILON = numpy.finfo(numpy.float32).eps
FLOAT64_EPSILON = numpy.finfo(numpy.float64).eps
