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
from typing import Sequence
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

from pi_ml import Trainer
from pi_ml.trainer.supporters import (
    _nested_calc_num_data,
    CombinedDataset,
    CombinedLoader,
    CombinedLoaderIterator,
    CycleIterator,
    TensorRunningAccum,
)
from pi_ml.utilities.apply_func import apply_to_collection
from pi_ml.utilities.auto_restart import CaptureMapDataset, FastForwardSampler
from pi_ml.utilities.data import get_len
from pi_ml.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset


def test_tensor_running_accum_reset():
    """Test that reset would set all attributes to the initialization state."""

    window_length = 10

    accum = TensorRunningAccum(window_length=window_length)
    assert accum.last() is None
    assert accum.mean() is None

    accum.append(torch.tensor(1.5))
    assert accum.last() == torch.tensor(1.5)
    assert accum.mean() == torch.tensor(1.5)

    accum.reset()
    assert accum.window_length == window_length
    assert accum.memory is None
    assert accum.current_idx == 0
    assert accum.last_idx is None
    assert not accum.rotated


def test_cycle_iterator():
    """Test the cycling function of `CycleIterator`"""

    iterator = CycleIterator(range(100), 1000)
    assert len(iterator) == 1000
    for idx, item in enumerate(iterator):
        assert item < 100

    assert idx == len(iterator) - 1


def test_none_length_cycle_iterator():
    """Test the infinite cycling function of `CycleIterator`"""
    iterator = CycleIterator(range(100))
    assert iterator.__len__() == float("inf")

    # test infinite loop
    for idx, item in enumerate(iterator):
        if idx == 1000:
            break
    assert item == 0


@pytest.mark.parametrize(
    ["dataset_1", "dataset_2"],
    [
        ([list(range(10)), list(range(20))]),
        ([range(10), range(20)]),
        ([torch.randn(10, 3, 2), torch.randn(20, 5, 6)]),
        ([TensorDataset(torch.randn(10, 3, 2)), TensorDataset(torch.randn(20, 5, 6))]),
    ],
)
def test_combined_dataset(dataset_1, dataset_2):
    """Verify the length of the CombinedDataset."""
    datasets = [dataset_1, dataset_2]
    combined_dataset = CombinedDataset(datasets)

    assert combined_dataset.max_len == 20
    assert combined_dataset.min_len == len(combined_dataset) == 10


def test_combined_dataset_length_mode_error():
    dset = CombinedDataset([range(10)])
    with pytest.raises(MisconfigurationException, match="Invalid Mode"):
        dset._calc_num_data([range(10)], "test")


def test_combined_loader_iterator_dict_min_size():
    """Test `CombinedLoaderIterator` given mapping loaders."""
    loaders = {
        "a": torch.utils.data.DataLoader(range(10), batch_size=4),
        "b": torch.utils.data.DataLoader(range(20), batch_size=5),
    }

    combined_iter = CombinedLoaderIterator(loaders)

    for idx, item in enumerate(combined_iter):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert "a" in item and "b" in item

    assert idx == min(len(loaders["a"]), len(loaders["b"])) - 1


def test_combined_loader_init_mode_error():
    """Test the ValueError when constructing `CombinedLoader`"""
    with pytest.raises(MisconfigurationException, match="Invalid Mode"):
        CombinedLoader([range(10)], "testtt")


def test_combined_loader_loader_type_error():
    """Test the ValueError when wrapping the loaders."""
    with pytest.raises(TypeError, match="Expected data to be int, Sequence or Mapping, but got NoneType"):
        CombinedLoader(None, "max_size_cycle")


def test_combined_loader_calc_length_mode_error():
    """Test the ValueError when calculating the number of batches."""
    with pytest.raises(TypeError, match="Expected data to be int, Sequence or Mapping, but got NoneType"):
        CombinedLoader._calc_num_batches(None)


def test_combined_loader_dict_min_size():
    """Test `CombinedLoader` of mode 'min_size' given mapping loaders."""
    loaders = {
        "a": torch.utils.data.DataLoader(range(10), batch_size=4),
        "b": torch.utils.data.DataLoader(range(20), batch_size=5),
    }

    combined_loader = CombinedLoader(loaders, "min_size")

    assert len(combined_loader) == min(len(v) for v in loaders.values())

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert "a" in item and "b" in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_dict_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given mapping loaders."""
    loaders = {
        "a": torch.utils.data.DataLoader(range(10), batch_size=4),
        "b": torch.utils.data.DataLoader(range(20), batch_size=5),
    }

    combined_loader = CombinedLoader(loaders, "max_size_cycle")

    assert len(combined_loader) == max(len(v) for v in loaders.values())

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert "a" in item and "b" in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_min_size():
    """Test `CombinedLoader` of mode 'min_size' given sequence loaders."""
    loaders = [
        torch.utils.data.DataLoader(range(10), batch_size=4),
        torch.utils.data.DataLoader(range(20), batch_size=5),
    ]

    combined_loader = CombinedLoader(loaders, "min_size")

    assert len(combined_loader) == min(len(v) for v in loaders)

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


class TestIterableDataset(IterableDataset):
    def __init__(self, size: int = 10):
        self.size = size

    def __iter__(self):
        self.sampler = SequentialSampler(range(self.size))
        self.sampler_iter = iter(self.sampler)
        return self

    def __next__(self):
        return next(self.sampler_iter)


@pytest.mark.parametrize("mode", ["min_size", "max_size_cycle"])
@pytest.mark.parametrize("use_multiple_dataloaders", [False, True])
def test_combined_loader_sequence_iterable_dataset(mode, use_multiple_dataloaders):
    """Test `CombinedLoader` of mode 'min_size' given sequence loaders."""
    if use_multiple_dataloaders:
        loaders = [
            torch.utils.data.DataLoader(TestIterableDataset(10), batch_size=2),
            torch.utils.data.DataLoader(TestIterableDataset(20), batch_size=2),
        ]
    else:
        loaders = [
            torch.utils.data.DataLoader(TestIterableDataset(10), batch_size=2),
        ]

    combined_loader = CombinedLoader(loaders, mode)

    has_break = False

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2 if use_multiple_dataloaders else 1
        if not use_multiple_dataloaders and idx == 4:
            has_break = True
            break

    if mode == "max_size_cycle":
        assert combined_loader.loaders[0].state.done == (not has_break)
    expected = (10 if mode == "max_size_cycle" else 5) if use_multiple_dataloaders else 5
    assert (expected - 1) == idx, (mode, use_multiple_dataloaders)


@pytest.mark.parametrize("lengths", [[4, 6], [5, 5], [6, 4]])
def test_combined_loader_sequence_with_map_and_iterable(lengths):
    class MyIterableDataset(IterableDataset):
        def __init__(self, size: int = 10):
            self.size = size

        def __iter__(self):
            self.sampler = SequentialSampler(range(self.size))
            self.iter_sampler = iter(self.sampler)
            return self

        def __next__(self):
            return next(self.iter_sampler)

    class MyMapDataset(Dataset):
        def __init__(self, size: int = 10):
            self.size = size

        def __getitem__(self, index):
            return index

        def __len__(self):
            return self.size

    x, y = lengths
    loaders = [DataLoader(MyIterableDataset(x)), DataLoader(MyMapDataset(y))]
    dataloader = CombinedLoader(loaders, mode="max_size_cycle")
    counter = 0
    for _ in dataloader:
        counter += 1
    assert counter == max(x, y)


def test_combined_loader_sequence_max_size_cycle():
    """Test `CombinedLoader` of mode 'max_size_cycle' given sequence loaders."""
    loaders = [
        torch.utils.data.DataLoader(range(10), batch_size=4),
        torch.utils.data.DataLoader(range(20), batch_size=5),
    ]

    combined_loader = CombinedLoader(loaders, "max_size_cycle")

    assert len(combined_loader) == max(len(v) for v in loaders)

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


@pytest.mark.parametrize(
    ["input_data", "compute_func", "expected_length"],
    [
        ([*range(10), list(range(1, 20))], min, 0),
        ([*range(10), list(range(1, 20))], max, 19),
        ([*range(10), {str(i): i for i in range(1, 20)}], min, 0),
        ([*range(10), {str(i): i for i in range(1, 20)}], max, 19),
        ({**{str(i): i for i in range(10)}, "nested": {str(i): i for i in range(1, 20)}}, min, 0),
        ({**{str(i): i for i in range(10)}, "nested": {str(i): i for i in range(1, 20)}}, max, 19),
        ({**{str(i): i for i in range(10)}, "nested": list(range(20))}, min, 0),
        ({**{str(i): i for i in range(10)}, "nested": list(range(20))}, max, 19),
    ],
)
def test_nested_calc_num_data(input_data, compute_func, expected_length):
    calculated_length = _nested_calc_num_data(input_data, compute_func)

    assert calculated_length == expected_length


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "PL_TRAINER_GPUS": "2"})
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.is_available", return_value=True)
@pytest.mark.parametrize("use_fault_tolerant", [False, True])
@pytest.mark.parametrize("replace_sampler_ddp", [False, True])
def test_combined_data_loader_validation_test(
    cuda_available_mock, device_count_mock, use_fault_tolerant, replace_sampler_ddp, tmpdir
):
    """This test makes sure distributed sampler has been properly injected in dataloaders when using
    CombinedLoader."""

    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    class CustomSampler(RandomSampler):
        def __init__(self, data_source, name) -> None:
            super().__init__(data_source)
            self.name = name

    dataset = CustomDataset(range(10))
    dataloader = CombinedLoader(
        {
            "a": DataLoader(CustomDataset(range(10))),
            "b": DataLoader(dataset, sampler=CustomSampler(dataset, "custom_sampler")),
            "c": {"c": DataLoader(CustomDataset(range(10))), "d": DataLoader(CustomDataset(range(10)))},
            "d": [DataLoader(CustomDataset(range(10))), DataLoader(CustomDataset(range(10)))],
        }
    )

    with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": str(int(use_fault_tolerant))}):

        trainer = Trainer(replace_sampler_ddp=replace_sampler_ddp, strategy="ddp", gpus=2)
        dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=True)
        _count = 0
        _has_fastforward_sampler = False

    def _assert_distributed_sampler(v):
        nonlocal _count
        nonlocal _has_fastforward_sampler
        _count += 1
        if use_fault_tolerant:
            _has_fastforward_sampler = True
            assert isinstance(v, FastForwardSampler)
            v = v._sampler
        if replace_sampler_ddp:
            assert isinstance(v, DistributedSampler)
        else:
            assert isinstance(v, (SequentialSampler, CustomSampler))

    apply_to_collection(dataloader.sampler, Sampler, _assert_distributed_sampler)
    assert _count == 6
    assert _has_fastforward_sampler == use_fault_tolerant

    def _assert_dataset(loader):
        d = loader.dataset
        if use_fault_tolerant:
            assert isinstance(d, CaptureMapDataset)
        else:
            assert isinstance(d, CustomDataset)

    apply_to_collection(dataloader.loaders, DataLoader, _assert_dataset)


@pytest.mark.parametrize("replace_sampler_ddp", [False, True])
def test_combined_data_loader_with_max_size_cycle_and_ddp(replace_sampler_ddp):
    """This test makes sure distributed sampler has been properly injected in dataloaders when using CombinedLoader
    with ddp and `max_size_cycle` mode."""
    trainer = Trainer(strategy="ddp", accelerator="auto", devices=2, replace_sampler_ddp=replace_sampler_ddp)

    dataloader = CombinedLoader(
        {"a": DataLoader(RandomDataset(32, 8), batch_size=1), "b": DataLoader(RandomDataset(32, 8), batch_size=1)},
    )
    dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=False)
    assert len(dataloader) == 4 if replace_sampler_ddp else 8

    for a_length in [6, 8, 10]:
        dataloader = CombinedLoader(
            {
                "a": DataLoader(range(a_length), batch_size=1),
                "b": DataLoader(range(8), batch_size=1),
            },
            mode="max_size_cycle",
        )

        length = max(a_length, 8)
        assert len(dataloader) == length
        dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=False)
        assert len(dataloader) == length // 2 if replace_sampler_ddp else length
        if replace_sampler_ddp:
            last_batch = list(dataloader)[-1]
            if a_length == 6:
                assert last_batch == {"a": torch.tensor([0]), "b": torch.tensor([6])}
            elif a_length == 8:
                assert last_batch == {"a": torch.tensor([6]), "b": torch.tensor([6])}
            elif a_length == 10:
                assert last_batch == {"a": torch.tensor([8]), "b": torch.tensor([0])}

    class InfiniteDataset(IterableDataset):
        def __iter__(self):
            while True:
                yield 1

    dataloader = CombinedLoader(
        {
            "a": DataLoader(InfiniteDataset(), batch_size=1),
            "b": DataLoader(range(8), batch_size=1),
        },
        mode="max_size_cycle",
    )
    assert get_len(dataloader) == float("inf")
    assert len(dataloader.loaders["b"].loader) == 8
    dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=False)
    assert len(dataloader.loaders["b"].loader) == 4 if replace_sampler_ddp else 8
    assert get_len(dataloader) == float("inf")


@pytest.mark.parametrize("replace_sampler_ddp", [False, True])
@pytest.mark.parametrize("is_min_size_mode", [False, True])
@pytest.mark.parametrize("use_combined_loader", [False, True])
def test_combined_dataloader_for_training_with_ddp(
    replace_sampler_ddp: bool, is_min_size_mode: bool, use_combined_loader: bool
):
    """When providing a CombinedLoader as the training data, it should be correctly receive the distributed
    samplers."""
    mode = "min_size" if is_min_size_mode else "max_size_cycle"
    dim = 3
    n1 = 8
    n2 = 6
    dataloader = {
        "a": DataLoader(RandomDataset(dim, n1), batch_size=1),
        "b": DataLoader(RandomDataset(dim, n2), batch_size=1),
    }
    if use_combined_loader:
        dataloader = CombinedLoader(dataloader, mode=mode)
    expected_length_before_ddp = min(n1, n2) if is_min_size_mode else max(n1, n2)
    expected_length_after_ddp = expected_length_before_ddp // 2 if replace_sampler_ddp else expected_length_before_ddp
    model = BoringModel()
    trainer = Trainer(
        strategy="ddp",
        accelerator="auto",
        devices=2,
        replace_sampler_ddp=replace_sampler_ddp,
        multiple_trainloader_mode="max_size_cycle" if use_combined_loader else mode,
    )
    trainer._data_connector.attach_data(
        model=model, train_dataloaders=dataloader, val_dataloaders=None, datamodule=None
    )
    trainer.reset_train_dataloader(model=model)
    assert trainer.train_dataloader is not None
    assert isinstance(trainer.train_dataloader, CombinedLoader)
    assert trainer.train_dataloader.mode == mode
    assert trainer.num_training_batches == expected_length_after_ddp
