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
from typing import Any, List, MutableSequence, Optional, Tuple, Union

import torch

from pi_ml.plugins.environments import TorchElasticEnvironment
from pi_ml.tuner.auto_gpu_select import pick_multiple_gpus
from pi_ml.utilities import _TPU_AVAILABLE
from pi_ml.utilities.exceptions import MisconfigurationException


def determine_root_gpu_device(gpus: List[int]) -> Optional[int]:
    """
    Args:
        gpus: non-empty list of ints representing which gpus to use

    Returns:
        designated root GPU device id

    Raises:
        TypeError:
            If ``gpus`` is not a list
        AssertionError:
            If GPU list is empty
    """
    if gpus is None:
        return None

    if not isinstance(gpus, list):
        raise TypeError("gpus should be a list")

    assert len(gpus) > 0, "gpus should be a non empty list"

    # set root gpu
    root_gpu = gpus[0]

    return root_gpu


def _parse_devices(
    gpus: Optional[Union[List[int], str, int]],
    auto_select_gpus: bool,
    tpu_cores: Optional[Union[List[int], str, int]],
) -> Tuple[Optional[List[int]], Optional[Union[List[int], int]]]:
    if auto_select_gpus and isinstance(gpus, int):
        gpus = pick_multiple_gpus(gpus)

    # TODO (@seannaren, @kaushikb11): Include IPU parsing logic here
    gpu_ids = parse_gpu_ids(gpus)
    tpu_cores = parse_tpu_cores(tpu_cores)
    return gpu_ids, tpu_cores


def parse_gpu_ids(gpus: Optional[Union[int, str, List[int]]]) -> Optional[List[int]]:
    """
    Parses the GPU ids given in the format as accepted by the
    :class:`~pi_ml.trainer.Trainer`.

    Args:
        gpus: An int -1 or string '-1' indicate that all available GPUs should be used.
            A list of unique ints or a string containing list of comma separated unique integers
            indicates specific GPUs to use.
            An int 0 means that no GPUs should be used.
            Any int N > 0 indicates that GPUs [0..N) should be used.

    Returns:
        a list of gpus to be used or ``None`` if no GPUs were requested

    If no GPUs are available but the value of gpus variable indicates request for GPUs
    then a MisconfigurationException is raised.
    """
    # Check that gpus param is None, Int, String or List
    _check_data_type(gpus)

    # Handle the case when no gpus are requested
    if gpus is None or (isinstance(gpus, int) and gpus == 0) or str(gpus).strip() in ("0", "[]"):
        return None

    # We know user requested GPUs therefore if some of the
    # requested GPUs are not available an exception is thrown.
    gpus = _normalize_parse_gpu_string_input(gpus)
    gpus = _normalize_parse_gpu_input_to_list(gpus)
    if not gpus:
        raise MisconfigurationException("GPUs requested but none are available.")
    if TorchElasticEnvironment.detect() and len(gpus) != 1 and len(_get_all_available_gpus()) == 1:
        # omit sanity check on torchelastic as by default shows one visible GPU per process
        return gpus

    # Check that gpus are unique. Duplicate gpus are not supported by the backend.
    _check_unique(gpus)

    return _sanitize_gpu_ids(gpus)


def parse_tpu_cores(tpu_cores: Optional[Union[int, str, List[int]]]) -> Optional[Union[int, List[int]]]:
    """
    Parses the tpu_cores given in the format as accepted by the
    :class:`~pi_ml.trainer.Trainer`.

    Args:
        tpu_cores: An int 1 or string '1' indicate that 1 core with multi-processing should be used
            An int 8 or string '8' indicate that all 8 cores with multi-processing should be used
            A list of int or a string containing list of comma separated integer
            indicates specific TPU core to use.

    Returns:
        a list of tpu_cores to be used or ``None`` if no TPU cores were requested

    Raises:
        MisconfigurationException:
            If TPU cores aren't 1 or 8 cores, or no TPU devices are found
    """
    _check_data_type(tpu_cores)

    if isinstance(tpu_cores, str):
        tpu_cores = _parse_tpu_cores_str(tpu_cores.strip())

    if not _tpu_cores_valid(tpu_cores):
        raise MisconfigurationException("`tpu_cores` can only be 1, 8 or [<1-8>]")

    if tpu_cores is not None and not _TPU_AVAILABLE:
        raise MisconfigurationException("No TPU devices were found.")

    return tpu_cores


def _normalize_parse_gpu_string_input(s: Union[int, str, List[int]]) -> Union[int, List[int]]:
    if not isinstance(s, str):
        return s
    if s == "-1":
        return -1
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if len(x) > 0]
    return int(s.strip())


def _sanitize_gpu_ids(gpus: List[int]) -> List[int]:
    """Checks that each of the GPUs in the list is actually available. Raises a MisconfigurationException if any of
    the GPUs is not available.

    Args:
        gpus: list of ints corresponding to GPU indices

    Returns:
        unmodified gpus variable

    Raises:
        MisconfigurationException:
            If machine has fewer available GPUs than requested.
    """
    all_available_gpus = _get_all_available_gpus()
    for gpu in gpus:
        if gpu not in all_available_gpus:
            raise MisconfigurationException(
                f"You requested GPUs: {gpus}\n But your machine only has: {all_available_gpus}"
            )
    return gpus


def _normalize_parse_gpu_input_to_list(gpus: Union[int, List[int], Tuple[int, ...]]) -> Optional[List[int]]:
    assert gpus is not None
    if isinstance(gpus, (MutableSequence, tuple)):
        return list(gpus)

    # must be an int
    if not gpus:  # gpus==0
        return None
    if gpus == -1:
        return _get_all_available_gpus()

    return list(range(gpus))


def _get_all_available_gpus() -> List[int]:
    """
    Returns:
         a list of all available gpus
    """
    return list(range(torch.cuda.device_count()))


def _check_unique(device_ids: List[int]) -> None:
    """Checks that the device_ids are unique.

    Args:
        device_ids: list of ints corresponding to gpus indices

    Raises:
        MisconfigurationException:
            If ``device_ids`` of GPUs aren't unique
    """
    if len(device_ids) != len(set(device_ids)):
        raise MisconfigurationException("Device ID's (GPU) must be unique.")


def _check_data_type(device_ids: Any) -> None:
    """Checks that the device_ids argument is one of: None, Int, String or List. Raises a MisconfigurationException
    otherwise.

    Args:
        device_ids: gpus/tpu_cores parameter as passed to the Trainer

    Raises:
        MisconfigurationException:
            If ``device_ids`` of GPU/TPUs aren't ``int``, ``str``, sequence of ``int`` or ``None``
    """
    if device_ids is not None and (
        not isinstance(device_ids, (int, str, MutableSequence, tuple)) or isinstance(device_ids, bool)
    ):
        raise MisconfigurationException("Device ID's (GPU/TPU) must be int, string or sequence of ints or None.")


def _tpu_cores_valid(tpu_cores: Any) -> bool:
    # allow 1 or 8 cores
    if tpu_cores in (1, 8, None):
        return True

    # allow picking 1 of 8 indexes
    if isinstance(tpu_cores, (list, tuple, set)):
        has_1_tpu_idx = len(tpu_cores) == 1
        is_valid_tpu_idx = 1 <= list(tpu_cores)[0] <= 8

        is_valid_tpu_core_choice = has_1_tpu_idx and is_valid_tpu_idx
        return is_valid_tpu_core_choice

    return False


def _parse_tpu_cores_str(tpu_cores: str) -> Union[int, List[int]]:
    if tpu_cores in ("1", "8"):
        return int(tpu_cores)
    return [int(x.strip()) for x in tpu_cores.split(",") if len(x) > 0]
