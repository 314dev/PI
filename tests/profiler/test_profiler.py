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
import logging
import os
import platform
import time
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pi_ml import Callback, Trainer
from pi_ml.callbacks import EarlyStopping, StochasticWeightAveraging
from pi_ml.loggers.base import LoggerCollection
from pi_ml.loggers.tensorboard import TensorBoardLogger
from pi_ml.profiler import AdvancedProfiler, PassThroughProfiler, PyTorchProfiler, SimpleProfiler
from pi_ml.profiler.pytorch import RegisterRecordFunction, warning_cache
from pi_ml.utilities.exceptions import MisconfigurationException
from pi_ml.utilities.imports import _KINETO_AVAILABLE
from tests.helpers import BoringModel, ManualOptimBoringModel
from tests.helpers.runif import RunIf

PROFILER_OVERHEAD_MAX_TOLERANCE = 0.0005


def _get_python_cprofile_total_duration(profile):
    return sum(x.inlinetime for x in profile.getstats())


def _sleep_generator(durations):
    """the profile_iterable method needs an iterable in which we can ensure that we're properly timing how long it
    takes to call __next__"""
    for duration in durations:
        time.sleep(duration)
        yield duration


@pytest.fixture
def simple_profiler():
    return SimpleProfiler()


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(["action", "expected"], [("a", [3, 1]), ("b", [2]), ("c", [1])])
def test_simple_profiler_durations(simple_profiler, action: str, expected: list):
    """Ensure the reported durations are reasonably accurate."""

    for duration in expected:
        with simple_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    np.testing.assert_allclose(simple_profiler.recorded_durations[action], expected, rtol=0.2)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(["action", "expected"], [("a", [3, 1]), ("b", [2]), ("c", [1])])
def test_simple_profiler_iterable_durations(simple_profiler, action: str, expected: list):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in simple_profiler.profile_iterable(iterable, action):
        pass

    # we exclude the last item in the recorded durations since that's when StopIteration is raised
    np.testing.assert_allclose(simple_profiler.recorded_durations[action][:-1], expected, rtol=0.2)


def test_simple_profiler_overhead(simple_profiler, n_iter=5):
    """Ensure that the profiler doesn't introduce too much overhead during training."""
    for _ in range(n_iter):
        with simple_profiler.profile("no-op"):
            pass

    durations = np.array(simple_profiler.recorded_durations["no-op"])
    assert all(durations < PROFILER_OVERHEAD_MAX_TOLERANCE)


def test_simple_profiler_value_errors(simple_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        simple_profiler.stop(action)

    simple_profiler.start(action)

    with pytest.raises(ValueError):
        simple_profiler.start(action)

    simple_profiler.stop(action)


def test_simple_profiler_deepcopy(tmpdir):
    simple_profiler = SimpleProfiler(dirpath=tmpdir, filename="test")
    simple_profiler.describe()
    assert deepcopy(simple_profiler)


def test_simple_profiler_dirpath(tmpdir):
    """Ensure the profiler dirpath defaults to `trainer.log_dir` when not present."""
    profiler = SimpleProfiler(filename="profiler")
    assert profiler.dirpath is None

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, profiler=profiler)
    trainer.fit(model)

    expected = tmpdir / "lightning_logs" / "version_0"
    assert trainer.log_dir == expected
    assert profiler.dirpath == trainer.log_dir
    assert expected.join("fit-profiler.txt").exists()


def test_simple_profiler_with_nonexisting_log_dir(tmpdir):
    """Ensure the profiler dirpath defaults to `trainer.log_dir`and creates it when not present."""
    nonexisting_tmpdir = tmpdir / "nonexisting"

    profiler = SimpleProfiler(filename="profiler")
    assert profiler.dirpath is None

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=nonexisting_tmpdir, max_epochs=1, limit_train_batches=1, limit_val_batches=1, profiler=profiler
    )
    trainer.fit(model)

    expected = nonexisting_tmpdir / "lightning_logs" / "version_0"
    assert expected.exists()
    assert trainer.log_dir == expected
    assert profiler.dirpath == trainer.log_dir
    assert expected.join("fit-profiler.txt").exists()


def test_simple_profiler_with_nonexisting_dirpath(tmpdir):
    """Ensure the profiler creates non-existing dirpath."""
    nonexisting_tmpdir = tmpdir / "nonexisting"

    profiler = SimpleProfiler(dirpath=nonexisting_tmpdir, filename="profiler")

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=1, limit_val_batches=1, profiler=profiler
    )
    trainer.fit(model)

    assert nonexisting_tmpdir.exists()
    assert nonexisting_tmpdir.join("fit-profiler.txt").exists()


@RunIf(skip_windows=True, skip_49370=True)
def test_simple_profiler_distributed_files(tmpdir):
    """Ensure the proper files are saved in distributed."""
    profiler = SimpleProfiler(dirpath=tmpdir, filename="profiler")
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=2,
        strategy="ddp_spawn",
        accelerator="cpu",
        devices=2,
        profiler=profiler,
        logger=False,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)

    actual = set(os.listdir(profiler.dirpath))
    expected = {f"{stage}-profiler-{rank}.txt" for stage in ("fit", "validate", "test") for rank in (0, 1)}
    assert actual == expected

    for f in profiler.dirpath.listdir():
        assert f.read_text("utf-8")


def test_simple_profiler_logs(tmpdir, caplog, simple_profiler):
    """Ensure that the number of printed logs is correct."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2, profiler=simple_profiler, logger=False)
    with caplog.at_level(logging.INFO, logger="pi_ml.profiler"):
        trainer.fit(model)
        trainer.test(model)

    assert caplog.text.count("Profiler Report") == 2


@pytest.mark.parametrize("extended", [True, False])
@patch("time.monotonic", return_value=70)
def test_simple_profiler_summary(tmpdir, extended):
    """Test the summary of `SimpleProfiler`."""
    profiler = SimpleProfiler(extended=extended)
    profiler.start_time = 63.0
    hooks = [
        "on_train_start",
        "on_train_end",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_before_batch_transfer",
        "on_fit_start",
    ]
    sometime = 0.773434
    sep = os.linesep
    max_action_len = len("on_before_batch_transfer")

    for i, hook in enumerate(hooks):
        with profiler.profile(hook):
            pass

        profiler.recorded_durations[hook] = [sometime + i]

    if extended:
        header_string = (
            f"{sep}|  {'Action':<{max_action_len}s}\t|  {'Mean duration (s)':<15}\t|  {'Num calls':<15}\t|"
            f"  {'Total time (s)':<15}\t|  {'Percentage %':<15}\t|"
        )
        output_string_len = len(header_string.expandtabs())
        sep_lines = f"{sep}{'-'* output_string_len}"
        expected_text = (
            f"Profiler Report{sep}"
            f"{sep_lines}"
            f"{sep}|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |"  # noqa: E501
            f"{sep_lines}"
            f"{sep}|  Total                        |  -                    |  6                    |  7.0                  |  100 %                |"  # noqa: E501
            f"{sep_lines}"
            f"{sep}|  on_fit_start                 |  5.7734               |  1                    |  5.7734               |  82.478               |"  # noqa: E501
            f"{sep}|  on_before_batch_transfer     |  4.7734               |  1                    |  4.7734               |  68.192               |"  # noqa: E501
            f"{sep}|  on_train_epoch_end           |  3.7734               |  1                    |  3.7734               |  53.906               |"  # noqa: E501
            f"{sep}|  on_train_epoch_start         |  2.7734               |  1                    |  2.7734               |  39.62                |"  # noqa: E501
            f"{sep}|  on_train_end                 |  1.7734               |  1                    |  1.7734               |  25.335               |"  # noqa: E501
            f"{sep}|  on_train_start               |  0.77343              |  1                    |  0.77343              |  11.049               |"  # noqa: E501
            f"{sep_lines}{sep}"
        )
    else:
        header_string = (
            f"{sep}|  {'Action':<{max_action_len}s}\t|  {'Mean duration (s)':<15}\t|  {'Total time (s)':<15}\t|"
        )
        output_string_len = len(header_string.expandtabs())
        sep_lines = f"{sep}{'-'* output_string_len}"
        expected_text = (
            f"Profiler Report{sep}"
            f"{sep_lines}"
            f"{sep}|  Action                       |  Mean duration (s)    |  Total time (s)       |"
            f"{sep_lines}"
            f"{sep}|  on_fit_start                 |  5.7734               |  5.7734               |"
            f"{sep}|  on_before_batch_transfer     |  4.7734               |  4.7734               |"
            f"{sep}|  on_train_epoch_end           |  3.7734               |  3.7734               |"
            f"{sep}|  on_train_epoch_start         |  2.7734               |  2.7734               |"
            f"{sep}|  on_train_end                 |  1.7734               |  1.7734               |"
            f"{sep}|  on_train_start               |  0.77343              |  0.77343              |"
            f"{sep_lines}{sep}"
        )

    summary = profiler.summary().expandtabs()
    assert expected_text == summary


@pytest.fixture
def advanced_profiler(tmpdir):
    return AdvancedProfiler(dirpath=tmpdir, filename="profiler")


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(["action", "expected"], [("a", [3, 1]), ("b", [2]), ("c", [1])])
def test_advanced_profiler_durations(advanced_profiler, action: str, expected: list):

    for duration in expected:
        with advanced_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    recored_total_duration = _get_python_cprofile_total_duration(advanced_profiler.profiled_actions[action])
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(recored_total_duration, expected_total_duration, rtol=0.2)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(["action", "expected"], [("a", [3, 1]), ("b", [2]), ("c", [1])])
def test_advanced_profiler_iterable_durations(advanced_profiler, action: str, expected: list):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in advanced_profiler.profile_iterable(iterable, action):
        pass

    recored_total_duration = _get_python_cprofile_total_duration(advanced_profiler.profiled_actions[action])
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(recored_total_duration, expected_total_duration, rtol=0.2)


@pytest.mark.flaky(reruns=3)
def test_advanced_profiler_overhead(advanced_profiler, n_iter=5):
    """ensure that the profiler doesn't introduce too much overhead during training."""
    for _ in range(n_iter):
        with advanced_profiler.profile("no-op"):
            pass

    action_profile = advanced_profiler.profiled_actions["no-op"]
    total_duration = _get_python_cprofile_total_duration(action_profile)
    average_duration = total_duration / n_iter
    assert average_duration < PROFILER_OVERHEAD_MAX_TOLERANCE


def test_advanced_profiler_describe(tmpdir, advanced_profiler):
    """ensure the profiler won't fail when reporting the summary."""
    # record at least one event
    with advanced_profiler.profile("test"):
        pass
    # log to stdout and print to file
    advanced_profiler.describe()
    path = advanced_profiler.dirpath / f"{advanced_profiler.filename}.txt"
    data = path.read_text("utf-8")
    assert len(data) > 0


def test_advanced_profiler_value_errors(advanced_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        advanced_profiler.stop(action)

    advanced_profiler.start(action)
    advanced_profiler.stop(action)


def test_advanced_profiler_deepcopy(advanced_profiler):
    advanced_profiler.describe()
    assert deepcopy(advanced_profiler)


@pytest.fixture
def pytorch_profiler(tmpdir):
    return PyTorchProfiler(dirpath=tmpdir, filename="profiler")


@RunIf(max_torch="1.8.1")
def test_pytorch_profiler_describe(pytorch_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    with pytorch_profiler.profile("on_test_start"):
        torch.tensor(0)

    # log to stdout and print to file
    pytorch_profiler.describe()
    path = pytorch_profiler.dirpath / f"{pytorch_profiler.filename}.txt"
    data = path.read_text("utf-8")
    assert len(data) > 0


def test_advanced_profiler_cprofile_deepcopy(tmpdir):
    """Checks for pickle issue reported in #6522."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, profiler="advanced", callbacks=StochasticWeightAveraging()
    )
    trainer.fit(model)


@RunIf(min_gpus=2, standalone=True)
def test_pytorch_profiler_trainer_ddp(tmpdir, pytorch_profiler):
    """Ensure that the profiler can be given to the training and default step are properly recorded."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=5,
        profiler=pytorch_profiler,
        strategy="ddp",
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)
    expected = {"[Strategy]DDPStrategy.validation_step"}
    if not _KINETO_AVAILABLE:
        expected |= {
            "[Strategy]DDPStrategy.training_step",
            "[Strategy]DDPStrategy.backward",
        }
    for name in expected:
        assert sum(e.name == name for e in pytorch_profiler.function_events), name

    files = set(os.listdir(pytorch_profiler.dirpath))
    expected = f"fit-profiler-{trainer.local_rank}.txt"
    assert expected in files

    path = pytorch_profiler.dirpath / expected
    assert path.read_text("utf-8")

    if _KINETO_AVAILABLE:
        files = os.listdir(pytorch_profiler.dirpath)
        files = [file for file in files if file.endswith(".json")]
        assert len(files) == 2, files
        local_rank = trainer.local_rank
        assert any(f"{local_rank}-optimizer_step_with_closure_" in f for f in files)
        assert any(f"{local_rank}-[Strategy]DDPStrategy.validation_step" in f for f in files)


@pytest.mark.parametrize("fast_dev_run", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("boring_model_cls", [ManualOptimBoringModel, BoringModel])
def test_pytorch_profiler_trainer_fit(fast_dev_run, boring_model_cls, tmpdir):
    """Ensure that the profiler can be given to the trainer and test step are properly recorded."""
    pytorch_profiler = PyTorchProfiler(dirpath=tmpdir, filename="profile")
    model = boring_model_cls()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, fast_dev_run=fast_dev_run, profiler=pytorch_profiler)
    trainer.fit(model)

    assert sum(e.name == "[Strategy]SingleDeviceStrategy.validation_step" for e in pytorch_profiler.function_events)

    path = pytorch_profiler.dirpath / f"fit-{pytorch_profiler.filename}.txt"
    assert path.read_text("utf-8")

    if _KINETO_AVAILABLE:
        files = sorted(file for file in os.listdir(tmpdir) if file.endswith(".json"))
        assert any(f"fit-{pytorch_profiler.filename}" in f for f in files)


@pytest.mark.parametrize("fn, step_name", [("test", "test"), ("validate", "validation"), ("predict", "predict")])
@pytest.mark.parametrize("boring_model_cls", [BoringModel, ManualOptimBoringModel])
def test_pytorch_profiler_trainer(fn, step_name, boring_model_cls, tmpdir):
    """Ensure that the profiler can be given to the trainer and test step are properly recorded."""
    pytorch_profiler = PyTorchProfiler(dirpath=tmpdir, filename="profile", schedule=None)
    model = boring_model_cls()
    model.predict_dataloader = model.train_dataloader
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_test_batches=2, profiler=pytorch_profiler)
    getattr(trainer, fn)(model)

    assert sum(e.name.endswith(f"{step_name}_step") for e in pytorch_profiler.function_events)

    path = pytorch_profiler.dirpath / f"{fn}-{pytorch_profiler.filename}.txt"
    assert path.read_text("utf-8")

    if _KINETO_AVAILABLE:
        files = sorted(file for file in os.listdir(tmpdir) if file.endswith(".json"))
        assert any(f"{fn}-{pytorch_profiler.filename}" in f for f in files)


def test_pytorch_profiler_nested(tmpdir):
    """Ensure that the profiler handles nested context."""

    pytorch_profiler = PyTorchProfiler(
        record_functions={"a", "b", "c"}, use_cuda=False, dirpath=tmpdir, filename="profiler", schedule=None
    )

    with pytorch_profiler.profile("a"):
        a = torch.ones(42)
        with pytorch_profiler.profile("b"):
            b = torch.zeros(42)
        with pytorch_profiler.profile("c"):
            _ = a + b

    pytorch_profiler.describe()

    events_name = {e.name for e in pytorch_profiler.function_events}

    names = {"a", "b", "c"}
    ops = {"add", "empty", "fill_", "ones", "zero_", "zeros"}
    ops = {"aten::" + op for op in ops}

    expected = names.union(ops)
    assert events_name == expected, (events_name, torch.__version__, platform.system())


def test_pytorch_profiler_logger_collection(tmpdir):
    """Tests whether the PyTorch profiler is able to write its trace locally when the Trainer's logger is an
    instance of LoggerCollection.

    See issue #8157.
    """

    def look_for_trace(trace_dir):
        """Determines if a directory contains a PyTorch trace."""
        return any("trace.json" in filename for filename in os.listdir(trace_dir))

    # Sanity check
    assert not look_for_trace(tmpdir)

    model = BoringModel()
    # Wrap the logger in a list so it becomes a LoggerCollection
    logger = [TensorBoardLogger(save_dir=tmpdir)]
    trainer = Trainer(default_root_dir=tmpdir, profiler="pytorch", logger=logger, limit_train_batches=5, max_epochs=1)
    assert isinstance(trainer.logger, LoggerCollection)
    trainer.fit(model)
    assert look_for_trace(tmpdir)


@RunIf(min_gpus=1, standalone=True)
def test_pytorch_profiler_nested_emit_nvtx(tmpdir):
    """This test check emit_nvtx is correctly supported."""
    profiler = PyTorchProfiler(use_cuda=True, emit_nvtx=True)

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, profiler=profiler, accelerator="gpu", devices=1)
    trainer.fit(model)


def test_register_record_function(tmpdir):

    use_cuda = torch.cuda.is_available()
    pytorch_profiler = PyTorchProfiler(
        export_to_chrome=False,
        record_functions={"a"},
        use_cuda=use_cuda,
        dirpath=tmpdir,
        filename="profiler",
        schedule=None,
        on_trace_ready=None,
    )

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1))

    model = TestModel()
    input = torch.rand((1, 1))

    if use_cuda:
        model = model.cuda()
        input = input.cuda()

    with pytorch_profiler.profile("a"):
        with RegisterRecordFunction(model):
            model(input)

    pytorch_profiler.describe()
    event_names = [e.name for e in pytorch_profiler.function_events]
    assert "torch.nn.modules.container.Sequential: layer" in event_names
    assert "torch.nn.modules.linear.Linear: layer.0" in event_names
    assert "torch.nn.modules.activation.ReLU: layer.1" in event_names
    assert "torch.nn.modules.linear.Linear: layer.2" in event_names


@pytest.mark.parametrize("cls", (SimpleProfiler, AdvancedProfiler, PyTorchProfiler))
def test_profiler_teardown(tmpdir, cls):
    """This test checks if profiler teardown method is called when trainer is exiting."""

    class TestCallback(Callback):
        def on_fit_end(self, trainer, *args, **kwargs) -> None:
            # describe sets it to None
            assert trainer.profiler._output_file is None

    profiler = cls(dirpath=tmpdir, filename="profiler")
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1, profiler=profiler, callbacks=[TestCallback()])
    trainer.fit(model)

    assert profiler._output_file is None


def test_pytorch_profiler_deepcopy(tmpdir):
    pytorch_profiler = PyTorchProfiler(dirpath=tmpdir, filename="profiler", schedule=None)
    pytorch_profiler.start("on_train_start")
    torch.tensor(1)
    pytorch_profiler.describe()
    assert deepcopy(pytorch_profiler)


@pytest.mark.parametrize(
    ["profiler", "expected"],
    [
        (None, PassThroughProfiler),
        (SimpleProfiler(), SimpleProfiler),
        (AdvancedProfiler(), AdvancedProfiler),
        ("simple", SimpleProfiler),
        ("Simple", SimpleProfiler),
        ("advanced", AdvancedProfiler),
        ("pytorch", PyTorchProfiler),
    ],
)
def test_trainer_profiler_correct_args(profiler, expected):
    kwargs = {"profiler": profiler} if profiler is not None else {}
    trainer = Trainer(**kwargs)
    assert isinstance(trainer.profiler, expected)


def test_trainer_profiler_incorrect_str_arg():
    with pytest.raises(
        MisconfigurationException,
        match=r"When passing string value for the `profiler` parameter of `Trainer`, it can only be one of.*",
    ):
        Trainer(profiler="unknown_profiler")


@pytest.mark.skipif(not _KINETO_AVAILABLE, reason="Requires PyTorch Profiler Kineto")
@pytest.mark.parametrize(
    ["trainer_config", "trainer_fn"],
    [
        ({"limit_train_batches": 4, "limit_val_batches": 7}, "fit"),
        ({"limit_train_batches": 7, "limit_val_batches": 4, "num_sanity_val_steps": 0}, "fit"),
        (
            {
                "limit_train_batches": 7,
                "limit_val_batches": 2,
            },
            "fit",
        ),
        ({"limit_val_batches": 4}, "validate"),
        ({"limit_test_batches": 4}, "test"),
        ({"limit_predict_batches": 4}, "predict"),
    ],
)
def test_pytorch_profiler_raises_warning_for_limited_steps(tmpdir, trainer_config, trainer_fn):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, profiler="pytorch", max_epochs=1, **trainer_config)
    warning_cache.clear()
    with pytest.warns(UserWarning, match="not enough steps to properly record traces"):
        getattr(trainer, trainer_fn)(model)
        assert trainer.profiler._schedule is None
        warning_cache.clear()


def test_profile_callbacks(tmpdir):
    """Checks if profiling callbacks works correctly, specifically when there are two of the same callback type."""

    pytorch_profiler = PyTorchProfiler(dirpath=tmpdir, filename="profiler", record_functions=set("on_train_end"))
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        profiler=pytorch_profiler,
        callbacks=[EarlyStopping("val_loss"), EarlyStopping("train_loss")],
    )
    trainer.fit(model)
    assert sum(
        e.name == "[Callback]EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}.on_validation_start"
        for e in pytorch_profiler.function_events
    )
    assert sum(
        e.name == "[Callback]EarlyStopping{'monitor': 'train_loss', 'mode': 'min'}.on_validation_start"
        for e in pytorch_profiler.function_events
    )
