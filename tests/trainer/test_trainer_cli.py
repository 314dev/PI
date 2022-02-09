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
import inspect
import pickle
from argparse import ArgumentParser, Namespace
from unittest import mock

import pytest

import tests.helpers.utils as tutils
from pi_ml import Trainer
from pi_ml.utilities import argparse
from tests.helpers.runif import RunIf


@mock.patch("argparse.ArgumentParser.parse_args")
def test_default_args(mock_argparse, tmpdir):
    """Tests default argument parser for Trainer."""
    mock_argparse.return_value = Namespace(**Trainer.default_attributes())

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    parser = ArgumentParser(add_help=False)
    args = parser.parse_args()
    args.logger = logger

    args.max_epochs = 5
    trainer = Trainer.from_argparse_args(args)

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 5


@pytest.mark.parametrize("cli_args", [["--accumulate_grad_batches=22"], ["--weights_save_path=./"], []])
def test_add_argparse_args_redefined(cli_args: list):
    """Redefines some default Trainer arguments via the cli and tests the Trainer initialization correctness."""
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parent_parser=parser)

    args = parser.parse_args(cli_args)

    # make sure we can pickle args
    pickle.dumps(args)

    # Check few deprecated args are not in namespace:
    for depr_name in ("gradient_clip", "nb_gpu_nodes", "max_nb_epochs"):
        assert depr_name not in args

    trainer = Trainer.from_argparse_args(args=args)
    pickle.dumps(trainer)

    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize("cli_args", [["--accumulate_grad_batches=22"], ["--weights_save_path=./"], []])
def test_add_argparse_args(cli_args: list):
    """Simple test ensuring Trainer.add_argparse_args works."""
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(cli_args)
    assert Trainer.from_argparse_args(args)

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser, use_argument_group=False)
    args = parser.parse_args(cli_args)
    assert Trainer.from_argparse_args(args)


def test_get_init_arguments_and_types():
    """Asserts a correctness of the `get_init_arguments_and_types` Trainer classmethod."""
    args = argparse.get_init_arguments_and_types(Trainer)
    parameters = inspect.signature(Trainer).parameters
    assert len(parameters) == len(args)
    for arg in args:
        assert parameters[arg[0]].default == arg[2]

    kwargs = {arg[0]: arg[2] for arg in args}
    trainer = Trainer(**kwargs)
    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize("cli_args", [["--callbacks=1", "--logger"], ["--foo", "--bar=1"]])
def test_add_argparse_args_redefined_error(cli_args: list, monkeypatch):
    """Asserts thar an error raised in case of passing not default cli arguments."""

    class _UnkArgError(Exception):
        pass

    def _raise():
        raise _UnkArgError

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parent_parser=parser)

    monkeypatch.setattr(parser, "exit", lambda *args: _raise(), raising=True)

    with pytest.raises(_UnkArgError):
        parser.parse_args(cli_args)


@pytest.mark.parametrize(
    ["cli_args", "expected"],
    [
        ("--auto_lr_find --auto_scale_batch_size power", {"auto_lr_find": True, "auto_scale_batch_size": "power"}),
        (
            "--auto_lr_find any_string --auto_scale_batch_size",
            {"auto_lr_find": "any_string", "auto_scale_batch_size": True},
        ),
        ("--auto_lr_find TRUE --auto_scale_batch_size FALSE", {"auto_lr_find": True, "auto_scale_batch_size": False}),
        ("--auto_lr_find t --auto_scale_batch_size ON", {"auto_lr_find": True, "auto_scale_batch_size": True}),
        ("--auto_lr_find 0 --auto_scale_batch_size n", {"auto_lr_find": False, "auto_scale_batch_size": False}),
        (
            "",
            {
                # These parameters are marked as Optional[...] in Trainer.__init__, with None as default.
                # They should not be changed by the argparse interface.
                "min_steps": None,
                "accelerator": None,
                "weights_save_path": None,
                "profiler": None,
            },
        ),
    ],
)
def test_argparse_args_parsing(cli_args, expected):
    """Test multi type argument with bool."""
    cli_args = cli_args.split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = Trainer.add_argparse_args(parent_parser=parser)
        args = Trainer.parse_argparser(parser)

    for k, v in expected.items():
        assert getattr(args, k) == v
    assert Trainer.from_argparse_args(args)


@pytest.mark.parametrize(
    "cli_args,expected",
    [("", False), ("--fast_dev_run=0", False), ("--fast_dev_run=True", True), ("--fast_dev_run 2", 2)],
)
def test_argparse_args_parsing_fast_dev_run(cli_args, expected):
    """Test multi type argument with bool."""
    cli_args = cli_args.split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = Trainer.add_argparse_args(parent_parser=parser)
        args = Trainer.parse_argparser(parser)
    assert args.fast_dev_run is expected


@pytest.mark.parametrize(
    ["cli_args", "expected_parsed", "expected_device_ids"],
    [("", None, None), ("--gpus 1", 1, [0]), ("--gpus 0,", "0,", [0])],
)
@RunIf(min_gpus=1)
def test_argparse_args_parsing_gpus(cli_args, expected_parsed, expected_device_ids):
    """Test multi type argument with bool."""
    cli_args = cli_args.split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = Trainer.add_argparse_args(parent_parser=parser)
        args = Trainer.parse_argparser(parser)

    assert args.gpus == expected_parsed
    trainer = Trainer.from_argparse_args(args)
    assert trainer.data_parallel_device_ids == expected_device_ids


@pytest.mark.parametrize(
    ["cli_args", "extra_args"],
    [
        ({}, {}),
        ({"logger": False}, {}),
        ({"logger": False}, {"logger": True}),
        ({"logger": False}, {"enable_checkpointing": True}),
    ],
)
def test_init_from_argparse_args(cli_args, extra_args):
    unknown_args = dict(unknown_arg=0)

    # unkown args in the argparser/namespace should be ignored
    with mock.patch("pi_ml.Trainer.__init__", autospec=True, return_value=None) as init:
        trainer = Trainer.from_argparse_args(Namespace(**cli_args, **unknown_args), **extra_args)
        expected = dict(cli_args)
        expected.update(extra_args)  # extra args should override any cli arg
        init.assert_called_with(trainer, **expected)

    # passing in unknown manual args should throw an error
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'unknown_arg'"):
        Trainer.from_argparse_args(Namespace(**cli_args), **extra_args, **unknown_args)
