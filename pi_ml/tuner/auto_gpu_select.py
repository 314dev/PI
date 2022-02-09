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
from typing import List

import torch

from pi_ml.utilities.exceptions import MisconfigurationException


def pick_multiple_gpus(nb: int) -> List[int]:
    """
    Raises:
        MisconfigurationException:
            If ``gpus`` is set to 0, when ``auto_select_gpus=True``.
    """
    if nb == 0:
        raise MisconfigurationException(
            "auto_select_gpus=True, gpus=0 is not a valid configuration."
            " Please select a valid number of GPU resources when using auto_select_gpus."
        )

    nb = torch.cuda.device_count() if nb == -1 else nb

    picked: List[int] = []
    for _ in range(nb):
        picked.append(pick_single_gpu(exclude_gpus=picked))

    return picked


def pick_single_gpu(exclude_gpus: List[int]) -> int:
    """
    Raises:
        RuntimeError:
            If you try to allocate a GPU, when no GPUs are available.
    """
    for i in range(torch.cuda.device_count()):
        if i in exclude_gpus:
            continue
        # Try to allocate on device:
        device = torch.device(f"cuda:{i}")
        try:
            torch.ones(1).to(device)
        except RuntimeError:
            continue
        return i
    raise RuntimeError("No GPUs available.")
