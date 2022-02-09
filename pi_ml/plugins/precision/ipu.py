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
from typing import Any, Callable, Union

from torch.nn import Module
from torch.optim import LBFGS, Optimizer

import pi_ml as pl
from pi_ml.plugins.precision.precision_plugin import PrecisionPlugin
from pi_ml.utilities import GradClipAlgorithmType
from pi_ml.utilities.exceptions import MisconfigurationException
from pi_ml.utilities.model_helpers import is_overridden
from pi_ml.utilities.warnings import WarningCache

warning_cache = WarningCache()


class IPUPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for IPU integration."""

    def __init__(self, precision: int) -> None:
        super().__init__()
        self.precision = precision

    def backward(self, model: "pl.LightningModule", *args: Any, **kwargs: Any) -> None:
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since IPUs handle"
                " the backward logic internally."
            )

    def optimizer_step(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        """IPUs handle the optimizer step internally."""
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"IPUs and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if isinstance(model, pl.LightningModule) and model.automatic_optimization and skipped_backward:
            # we lack coverage here and IPUs are (currently) limited - something to explore if there's demand
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not implemented for IPUs."
                " Please, open an issue in `https://github.com/PyTorchLightning/pytorch-lightning/issues`"
                " requesting this feature."
            )

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        if clip_val <= 0:
            return
        raise MisconfigurationException("IPUs currently do not support clipping gradients.")
