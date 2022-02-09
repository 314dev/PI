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
import re

import pytest
import torch

from pi_ml import Trainer
from pi_ml.tuner.auto_gpu_select import pick_multiple_gpus
from pi_ml.utilities.exceptions import MisconfigurationException
from tests.helpers.runif import RunIf


# TODO: add pytest.deprecated_call @daniellepintz
@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    ["auto_select_gpus", "gpus", "expected_error"],
    [(True, 0, MisconfigurationException), (True, -1, None), (False, 0, None), (False, -1, None)],
)
def test_trainer_with_gpus_options_combination_at_available_gpus_env(auto_select_gpus, gpus, expected_error):
    if expected_error:
        with pytest.raises(
            expected_error,
            match=re.escape(
                "auto_select_gpus=True, gpus=0 is not a valid configuration."
                " Please select a valid number of GPU resources when using auto_select_gpus."
            ),
        ):
            Trainer(auto_select_gpus=auto_select_gpus, gpus=gpus)
    else:
        Trainer(auto_select_gpus=auto_select_gpus, gpus=gpus)


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    ["nb", "expected_gpu_idxs", "expected_error"],
    [(0, [], MisconfigurationException), (-1, list(range(torch.cuda.device_count())), None), (1, [0], None)],
)
def test_pick_multiple_gpus(nb, expected_gpu_idxs, expected_error):
    if expected_error:
        with pytest.raises(
            expected_error,
            match=re.escape(
                "auto_select_gpus=True, gpus=0 is not a valid configuration."
                " Please select a valid number of GPU resources when using auto_select_gpus."
            ),
        ):
            pick_multiple_gpus(nb)
    else:
        assert expected_gpu_idxs == pick_multiple_gpus(nb)
