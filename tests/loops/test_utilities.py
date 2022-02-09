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
import pytest
import torch

from pi_ml.loops.utilities import _extract_hiddens
from pi_ml.utilities.exceptions import MisconfigurationException


def test_extract_hiddens():
    # tbptt not enabled, no hiddens return
    training_step_output = 1  # anything
    hiddens = _extract_hiddens(training_step_output, 0)
    assert hiddens is None

    # tbptt enabled, hiddens return
    hiddens = torch.tensor(321.12, requires_grad=True)
    training_step_output = {"hiddens": hiddens}
    hiddens = _extract_hiddens(training_step_output, 2)
    assert "hiddens" in training_step_output
    assert not hiddens.requires_grad

    # tbptt not enabled, hiddens return
    with pytest.raises(MisconfigurationException, match='returned "hiddens" .* but `truncated_bptt_steps` is disabled'):
        _extract_hiddens(training_step_output, 0)
    # tbptt enabled, no hiddens return
    with pytest.raises(MisconfigurationException, match="enabled `truncated_bptt_steps` but did not `return"):
        _extract_hiddens(None, 1)
