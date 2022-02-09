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
from pi_ml.utilities import rank_zero_deprecation

rank_zero_deprecation(
    "`pi_ml.core.memory.get_memory_profile` and"
    " `pi_ml.core.memory.get_gpu_memory_map` have been moved"
    " to `pi_ml.utilities.memory` since v1.5 and will be removed in v1.7."
)

# To support backward compatibility as get_memory_profile and get_gpu_memory_map have been moved
from pi_ml.utilities.memory import get_gpu_memory_map, get_memory_profile  # noqa: E402, F401 # isort: skip

rank_zero_deprecation(
    "`pi_ml.core.memory.LayerSummary` and"
    " `pi_ml.core.memory.ModelSummary` have been moved"
    " to `pi_ml.utilities.model_summary` since v1.5 and will be removed in v1.7."
)

# To support backward compatibility as LayerSummary and ModelSummary have been moved
from pi_ml.utilities.model_summary import LayerSummary, ModelSummary  # noqa: E402, F401 # isort: skip
