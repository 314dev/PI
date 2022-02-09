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
import shutil
from collections import OrderedDict
from typing import Any, IO, Iterable, List, Optional, Sequence, Union

import torch
from deprecate.utils import void
from torch.utils.data.dataloader import DataLoader

from pi_ml.loops.dataloader import DataLoaderLoop
from pi_ml.loops.epoch import EvaluationEpochLoop
from pi_ml.trainer.connectors.logger_connector.result import _OUT_DICT, _ResultCollection
from pi_ml.trainer.states import TrainerFn
from pi_ml.utilities.apply_func import apply_to_collection
from pi_ml.utilities.imports import _RICH_AVAILABLE
from pi_ml.utilities.types import EPOCH_OUTPUT

if _RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Column, Table


class EvaluationLoop(DataLoaderLoop):
    """Loops over all dataloaders for evaluation."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.epoch_loop = EvaluationEpochLoop()
        self.verbose = verbose

        self._results = _ResultCollection(training=False)
        self._outputs: List[EPOCH_OUTPUT] = []
        self._logged_outputs: List[_OUT_DICT] = []
        self._max_batches: List[int] = []
        self._has_run: bool = False

    @property
    def num_dataloaders(self) -> int:
        """Returns the total number of dataloaders."""
        # case where user does:
        # return dl1, dl2
        dataloaders = self.dataloaders
        if dataloaders is None:
            return 0
        length = len(dataloaders)
        if length > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length

    @property
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns the validation or test dataloaders."""
        dataloaders = self.trainer.test_dataloaders if self.trainer.testing else self.trainer.val_dataloaders
        if dataloaders is None:
            raise RuntimeError("Dataloaders should be available.")
        return dataloaders

    def connect(self, epoch_loop: EvaluationEpochLoop) -> None:  # type: ignore[override]
        """Connect the evaluation epoch loop with this loop."""
        self.epoch_loop = epoch_loop

    @property
    def done(self) -> bool:
        """Returns whether all dataloaders are processed or evaluation should be skipped altogether."""
        return super().done or self.skip

    @property
    def skip(self) -> bool:
        """Returns whether the evaluation should be skipped."""
        max_batches = self._get_max_batches()
        return sum(max_batches) == 0

    def reset(self) -> None:
        """Resets the internal state of the loop."""
        self._max_batches = self._get_max_batches()
        # bookkeeping
        self._outputs = []
        self._logged_outputs = []

        if isinstance(self._max_batches, int):
            self._max_batches = [self._max_batches] * len(self.dataloaders)

        super().reset()
        # when restarting, if we are running `validate` or `test` twice, since there's no concept of `max_epochs` we
        # need to reset the current state when the loop has finished running
        if self.done and self.trainer.state.fn != TrainerFn.FITTING:
            self.dataloader_progress.reset_on_run()

    def on_skip(self) -> List:
        return []

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs the ``_on_evaluation_model_eval``, ``_on_evaluation_start`` and ``_on_evaluation_epoch_start``
        hooks."""
        void(*args, **kwargs)

        # hook
        self._on_evaluation_model_eval()
        self.trainer.lightning_module.zero_grad()
        self._on_evaluation_start()
        self._on_evaluation_epoch_start()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs evaluation on one single dataloader."""
        void(*args, **kwargs)

        dataloader_idx = self.current_dataloader_idx
        dataloader = self.trainer.strategy.process_dataloader(self.current_dataloader)
        self.data_fetcher = dataloader = self.trainer._data_connector.get_profiled_dataloader(
            dataloader, dataloader_idx=dataloader_idx
        )
        dl_max_batches = self._max_batches[dataloader_idx]

        kwargs = OrderedDict()
        if self.num_dataloaders > 1:
            kwargs["dataloader_idx"] = dataloader_idx
        dl_outputs = self.epoch_loop.run(dataloader, dl_max_batches, kwargs)

        # store batch level output per dataloader
        self._outputs.append(dl_outputs)

        if not self.trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True

    def on_advance_end(self) -> None:
        self.trainer.logger_connector.epoch_end_reached()

        self._logged_outputs.append(self.trainer.logger_connector.update_eval_epoch_metrics())

        super().on_advance_end()

    def on_run_end(self) -> List[_OUT_DICT]:
        """Runs the ``_on_evaluation_epoch_end`` hook."""
        # if `done` returned True before any iterations were done, this won't have been called in `on_advance_end`
        self.trainer.logger_connector.epoch_end_reached()

        # hook
        self._evaluation_epoch_end(self._outputs)
        self._outputs = []  # free memory

        # hook
        self._on_evaluation_epoch_end()

        logged_outputs, self._logged_outputs = self._logged_outputs, []  # free memory
        # include any logged outputs on epoch_end
        epoch_end_logged_outputs = self.trainer.logger_connector.update_eval_epoch_metrics()
        for dl_outputs in logged_outputs:
            dl_outputs.update(epoch_end_logged_outputs)

        # log metrics
        self.trainer.logger_connector.log_eval_end_metrics()

        # hook
        self._on_evaluation_end()

        # enable train mode again
        self._on_evaluation_model_train()

        if self.verbose and self.trainer.is_global_zero:
            assert self.trainer.state.stage is not None
            self._print_results(logged_outputs, self.trainer.state.stage)

        return logged_outputs

    def teardown(self) -> None:
        self._results.cpu()
        self.epoch_loop.teardown()

    def _get_max_batches(self) -> List[int]:
        """Returns the max number of batches for each dataloader."""
        if self.trainer.testing:
            max_batches = self.trainer.num_test_batches
        else:
            if self.trainer.sanity_checking:
                max_batches = self.trainer.num_sanity_val_batches
            else:
                max_batches = self.trainer.num_val_batches
        return max_batches

    def _reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders if necessary."""
        if self.trainer.testing:
            self.trainer.reset_test_dataloader()
        elif self.trainer.val_dataloaders is None or self.trainer._data_connector._should_reload_val_dl:
            self.trainer.reset_val_dataloader()

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks."""
        assert self._results is not None
        self._results.to(device=self.trainer.lightning_module.device)

        if self.trainer.testing:
            self.trainer._call_callback_hooks("on_test_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_test_start", *args, **kwargs)
            self.trainer._call_strategy_hook("on_test_start", *args, **kwargs)
        else:
            self.trainer._call_callback_hooks("on_validation_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_validation_start", *args, **kwargs)
            self.trainer._call_strategy_hook("on_validation_start", *args, **kwargs)

    def _on_evaluation_model_eval(self) -> None:
        """Sets model to eval mode."""
        if self.trainer.testing:
            self.trainer._call_lightning_module_hook("on_test_model_eval")
        else:
            self.trainer._call_lightning_module_hook("on_validation_model_eval")

    def _on_evaluation_model_train(self) -> None:
        """Sets model to train mode."""
        if self.trainer.testing:
            self.trainer._call_lightning_module_hook("on_test_model_train")
        else:
            self.trainer._call_lightning_module_hook("on_validation_model_train")

    def _on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_end`` hook."""
        if self.trainer.testing:
            self.trainer._call_callback_hooks("on_test_end", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_test_end", *args, **kwargs)
            self.trainer._call_strategy_hook("on_test_end", *args, **kwargs)
        else:
            self.trainer._call_callback_hooks("on_validation_end", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_validation_end", *args, **kwargs)
            self.trainer._call_strategy_hook("on_validation_end", *args, **kwargs)

        # reset the logger connector state
        self.trainer.logger_connector.reset_results()

    def _on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_epoch_start`` and ``on_{validation/test}_epoch_start`` hooks."""
        self.trainer.logger_connector.on_epoch_start()
        self.trainer._call_callback_hooks("on_epoch_start", *args, **kwargs)
        self.trainer._call_lightning_module_hook("on_epoch_start", *args, **kwargs)

        if self.trainer.testing:
            self.trainer._call_callback_hooks("on_test_epoch_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_test_epoch_start", *args, **kwargs)
        else:
            self.trainer._call_callback_hooks("on_validation_epoch_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_validation_epoch_start", *args, **kwargs)

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]) -> None:
        """Runs ``{validation/test}_epoch_end``"""
        self.trainer.logger_connector._evaluation_epoch_end()

        # with a single dataloader don't pass a 2D list
        output_or_outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )

        # call the model epoch end
        if self.trainer.testing:
            self.trainer._call_lightning_module_hook("test_epoch_end", output_or_outputs)
        else:
            self.trainer._call_lightning_module_hook("validation_epoch_end", output_or_outputs)

    def _on_evaluation_epoch_end(self) -> None:
        """Runs ``on_{validation/test}_epoch_end`` hook."""
        hook_name = "on_test_epoch_end" if self.trainer.testing else "on_validation_epoch_end"
        self.trainer._call_callback_hooks(hook_name)
        self.trainer._call_lightning_module_hook(hook_name)

        self.trainer._call_callback_hooks("on_epoch_end")
        self.trainer._call_lightning_module_hook("on_epoch_end")
        self.trainer.logger_connector.on_epoch_end()

    @staticmethod
    def _get_keys(data: dict) -> Iterable[str]:
        if any(isinstance(v, dict) for v in data.values()):
            for v in data.values():
                yield from apply_to_collection(v, dict, dict.keys)
        else:
            yield from data.keys()

    @staticmethod
    def _find_value(data: dict, target: str) -> Iterable[Any]:
        for k, v in data.items():
            if k == target:
                yield v
            elif isinstance(v, dict):
                yield from EvaluationLoop._find_value(v, target)

    @staticmethod
    def _print_results(results: List[_OUT_DICT], stage: str, file: Optional[IO[str]] = None) -> None:
        # remove the dl idx suffix
        results = [{k.split("/dataloader_idx_")[0]: v for k, v in result.items()} for result in results]
        metrics = sorted({k for keys in apply_to_collection(results, dict, EvaluationLoop._get_keys) for k in keys})
        headers = [f"DataLoader {i}" for i in range(len(results))]

        # fallback is useful for testing of printed output
        term_size = shutil.get_terminal_size(fallback=(120, 30)).columns or 120
        max_length = int(min(max(len(max(metrics + headers, key=len)), 25), term_size / 2))

        rows: List[List[Any]] = [[] for _ in metrics]

        for result in results:
            for metric, row in zip(metrics, rows):
                v = list(EvaluationLoop._find_value(result, metric))
                if v:
                    val = v[0]
                    if isinstance(val, torch.Tensor):
                        val = val.item() if val.numel() == 1 else val.tolist()
                    row.append(f"{val}")
                else:
                    row.append(" ")

        # keep one column with max length for metrics
        num_cols = int((term_size - max_length) / max_length)

        for i in range(0, len(headers), num_cols):
            table_headers = headers[i : (i + num_cols)]
            table_rows = [row[i : (i + num_cols)] for row in rows]

            table_headers.insert(0, f"{stage} Metric".capitalize())

            if _RICH_AVAILABLE:
                console = Console(file=file)

                columns = [Column(h, justify="center", style="magenta", width=max_length) for h in table_headers]
                columns[0].style = "cyan"

                table = Table(*columns)
                for metric, row in zip(metrics, table_rows):
                    row.insert(0, metric)
                    table.add_row(*row)
                console.print(table)
            else:
                row_format = f"{{:^{max_length}}}" * len(table_headers)
                half_term_size = int(term_size / 2)

                bar = "─" * term_size
                lines = [bar, row_format.format(*table_headers).rstrip(), bar]
                for metric, row in zip(metrics, table_rows):
                    # deal with column overflow
                    if len(metric) > half_term_size:
                        while len(metric) > half_term_size:
                            row_metric = metric[:half_term_size]
                            metric = metric[half_term_size:]
                            lines.append(row_format.format(row_metric, *row).rstrip())
                        lines.append(row_format.format(metric, " ").rstrip())
                    else:
                        lines.append(row_format.format(metric, *row).rstrip())
                lines.append(bar)
                print(os.linesep.join(lines), file=file)
