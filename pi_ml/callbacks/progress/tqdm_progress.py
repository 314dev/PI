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
import importlib
import math
import os
import sys
from typing import Any, Dict, Optional, Union

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

import pi_ml as pl
from pi_ml.callbacks.progress.base import ProgressBarBase
from pi_ml.utilities.rank_zero import rank_zero_debug

_PAD_SIZE = 5


class Tqdm(_tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from
        flickering."""
        # this just to make the make docs happy, otherwise it pulls docs which has some issues...
        super().__init__(*args, **kwargs)

    @staticmethod
    def format_num(n: Union[int, float, str]) -> str:
        """Add additional padding to the formatted numbers."""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
            assert isinstance(n, str)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n


class TQDMProgressBar(ProgressBarBase):
    r"""
    This is the default progress bar used by Lightning. It prints to ``stdout`` using the
    :mod:`tqdm` package and shows up to four different bars:

        - **sanity check progress:** the progress during the sanity check run
        - **main progress:** shows training + validation progress combined. It also accounts for
          multiple validation runs during training when
          :paramref:`~pi_ml.trainer.trainer.Trainer.val_check_interval` is used.
        - **validation progress:** only visible during validation;
          shows total progress over all validation datasets.
        - **test progress:** only active when testing; shows total progress over all test datasets.

    For infinite datasets, the progress bar never ends.

    If you want to customize the default ``tqdm`` progress bars used by Lightning, you can override
    specific methods of the callback class and pass your custom implementation to the
    :class:`~pi_ml.trainer.trainer.Trainer`.

    Example:

        >>> class LitProgressBar(TQDMProgressBar):
        ...     def init_validation_tqdm(self):
        ...         bar = super().init_validation_tqdm()
        ...         bar.set_description('running validation ...')
        ...         return bar
        ...
        >>> bar = LitProgressBar()
        >>> from pi_ml import Trainer
        >>> trainer = Trainer(callbacks=[bar])

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display. By default, the :class:`~pi_ml.trainer.trainer.Trainer`
            uses this implementation of the progress bar and sets the refresh rate to the value provided to the
            :paramref:`~pi_ml.trainer.trainer.Trainer.progress_bar_refresh_rate` argument in the
            :class:`~pi_ml.trainer.trainer.Trainer`.
        process_position: Set this to a value greater than ``0`` to offset the progress bars by this many lines.
            This is useful when you have progress bars defined elsewhere and want to show all of them
            together. This corresponds to
            :paramref:`~pi_ml.trainer.trainer.Trainer.process_position` in the
            :class:`~pi_ml.trainer.trainer.Trainer`.
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__()
        self._refresh_rate = self._resolve_refresh_rate(refresh_rate)
        self._process_position = process_position
        self._enabled = True
        self._main_progress_bar: Optional[_tqdm] = None
        self._val_progress_bar: Optional[_tqdm] = None
        self._test_progress_bar: Optional[_tqdm] = None
        self._predict_progress_bar: Optional[_tqdm] = None

    def __getstate__(self) -> Dict:
        # can't pickle the tqdm objects
        return {k: v if not isinstance(v, _tqdm) else None for k, v in vars(self).items()}

    @property
    def main_progress_bar(self) -> _tqdm:
        if self._main_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._main_progress_bar` reference has not been set yet.")
        return self._main_progress_bar

    @main_progress_bar.setter
    def main_progress_bar(self, bar: _tqdm) -> None:
        self._main_progress_bar = bar

    @property
    def val_progress_bar(self) -> _tqdm:
        if self._val_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet.")
        return self._val_progress_bar

    @val_progress_bar.setter
    def val_progress_bar(self, bar: _tqdm) -> None:
        self._val_progress_bar = bar

    @property
    def test_progress_bar(self) -> _tqdm:
        if self._test_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._test_progress_bar` reference has not been set yet.")
        return self._test_progress_bar

    @test_progress_bar.setter
    def test_progress_bar(self, bar: _tqdm) -> None:
        self._test_progress_bar = bar

    @property
    def predict_progress_bar(self) -> _tqdm:
        if self._predict_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._predict_progress_bar` reference has not been set yet.")
        return self._predict_progress_bar

    @predict_progress_bar.setter
    def predict_progress_bar(self, bar: _tqdm) -> None:
        self._predict_progress_bar = bar

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    @property
    def _val_processed(self) -> int:
        if self.trainer.state.fn == "fit":
            # use total in case validation runs more than once per training epoch
            return self.trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.total.processed
        return self.trainer.validate_loop.epoch_loop.batch_progress.current.processed

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = Tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = Tqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def on_sanity_check_start(self, *_: Any) -> None:
        self.val_progress_bar = self.init_sanity_tqdm()
        self.main_progress_bar = Tqdm(disable=True)  # dummy progress bar

    def on_sanity_check_end(self, *_: Any) -> None:
        self.main_progress_bar.close()
        self.val_progress_bar.close()

    def on_train_start(self, *_: Any) -> None:
        self.main_progress_bar = self.init_train_tqdm()

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf") and total_val_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        self.main_progress_bar.total = convert_inf(total_batches)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        if self._should_update(self.train_batch_idx):
            _update_n(self.main_progress_bar, self.train_batch_idx + self._val_processed)
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        _update_n(self.main_progress_bar, self.train_batch_idx + self._val_processed)
        if not self.main_progress_bar.disable:
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, *_: Any) -> None:
        self.main_progress_bar.close()

    def on_validation_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if trainer.sanity_checking:
            self.val_progress_bar.total = sum(trainer.num_sanity_val_batches)
        else:
            self.val_progress_bar = self.init_validation_tqdm()
            self.val_progress_bar.total = convert_inf(self.total_val_batches)

    def on_validation_batch_end(self, trainer: "pl.Trainer", *_: Any) -> None:
        if self._should_update(self.val_batch_idx):
            _update_n(self.val_progress_bar, self.val_batch_idx)
            if trainer.state.fn == "fit":
                _update_n(self.main_progress_bar, self.train_batch_idx + self._val_processed)

    def on_validation_epoch_end(self, *_: Any) -> None:
        _update_n(self.val_progress_bar, self._val_processed)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._main_progress_bar is not None and trainer.state.fn == "fit":
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        self.val_progress_bar.close()

    def on_test_start(self, *_: Any) -> None:
        self.test_progress_bar = self.init_test_tqdm()
        self.test_progress_bar.total = convert_inf(self.total_test_batches)

    def on_test_batch_end(self, *_: Any) -> None:
        if self._should_update(self.test_batch_idx):
            _update_n(self.test_progress_bar, self.test_batch_idx)

    def on_test_epoch_end(self, *_: Any) -> None:
        _update_n(self.test_progress_bar, self.test_batch_idx)

    def on_test_end(self, *_: Any) -> None:
        self.test_progress_bar.close()

    def on_predict_epoch_start(self, *_: Any) -> None:
        self.predict_progress_bar = self.init_predict_tqdm()
        self.predict_progress_bar.total = convert_inf(self.total_predict_batches)

    def on_predict_batch_end(self, *_: Any) -> None:
        if self._should_update(self.predict_batch_idx):
            _update_n(self.predict_progress_bar, self.predict_batch_idx)

    def on_predict_end(self, *_: Any) -> None:
        self.predict_progress_bar.close()

    def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
        active_progress_bar = None

        if self._main_progress_bar is not None and not self.main_progress_bar.disable:
            active_progress_bar = self.main_progress_bar
        elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, **kwargs)

    def _should_update(self, idx: int) -> bool:
        return self.refresh_rate > 0 and idx % self.refresh_rate == 0

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if os.getenv("COLAB_GPU") and refresh_rate == 1:
            # smaller refresh rate on colab causes crashes, choose a higher value
            rank_zero_debug("Using a higher refresh rate on Colab. Setting it to `20`")
            refresh_rate = 20
        return refresh_rate


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.
    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x


def _update_n(bar: _tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()
