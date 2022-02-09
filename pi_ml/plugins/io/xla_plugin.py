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
import os
from typing import Any, Dict, Optional

from pi_ml.plugins.io.torch_plugin import TorchCheckpointIO
from pi_ml.utilities import _OMEGACONF_AVAILABLE, _TPU_AVAILABLE
from pi_ml.utilities.apply_func import apply_to_collection
from pi_ml.utilities.cloud_io import get_filesystem
from pi_ml.utilities.types import _PATH

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm

if _OMEGACONF_AVAILABLE:
    from omegaconf import DictConfig, ListConfig, OmegaConf


class XLACheckpointIO(TorchCheckpointIO):
    """CheckpointIO that utilizes :func:`xm.save` to save checkpoints for TPU training strategies."""

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: Optional parameters when saving the model/training states.
        """
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        # Todo: TypeError: 'mappingproxy' object does not support item assignment
        # Ref: https://github.com/pytorch/xla/issues/2773
        if _OMEGACONF_AVAILABLE:
            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)
        xm.save({k: v for k, v in checkpoint.items() if k != "callbacks"}, path)
