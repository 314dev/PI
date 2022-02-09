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
import pickle
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pi_ml import Trainer
from pi_ml.plugins import DoublePrecisionPlugin
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


class RandomFloatIntDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.float_data = torch.randn(length, size)
        self.int_data = torch.randint(10, (length, 1))

    def __getitem__(self, index):
        return self.float_data[index], self.int_data[index]

    def __len__(self):
        return self.len


class DoublePrecisionBoringModel(BoringModel):
    def training_step(self, batch, batch_idx):
        float_data, int_data = batch
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        assert float_data.dtype == torch.float64
        output = self(float_data)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        assert torch.tensor([0.0]).dtype == torch.float32
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        assert torch.tensor([0.0], dtype=torch.float16).dtype == torch.float16
        return self(batch)

    def on_fit_start(self):
        assert self.layer.weight.dtype == torch.float64

    def on_after_backward(self):
        assert self.layer.weight.grad.dtype == torch.float64

    def train_dataloader(self):
        dataset = RandomFloatIntDataset(32, 64)
        assert dataset.float_data.dtype == torch.float32  # Don't start with double data
        return DataLoader(dataset)

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class DoublePrecisionBoringModelNoForward(BoringModel):
    def training_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        loss = self.loss(batch, output)
        return {"y": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch.dtype == torch.float64
        output = self.layer(batch)
        assert output.dtype == torch.float64
        return output

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class DoublePrecisionBoringModelComplexBuffer(BoringModel):
    def __init__(self):
        super().__init__()

        self.register_buffer("complex_buffer", torch.complex(torch.rand(10), torch.rand(10)), False)

    def on_fit_start(self):
        assert self.layer.weight.dtype == torch.float64
        assert self.complex_buffer.dtype == torch.complex64


@pytest.mark.parametrize(
    "boring_model",
    [
        DoublePrecisionBoringModel,
        DoublePrecisionBoringModelNoForward,
        DoublePrecisionBoringModelComplexBuffer,
    ],
)
def test_double_precision(tmpdir, boring_model):
    model = boring_model()

    trainer = Trainer(max_epochs=2, default_root_dir=tmpdir, fast_dev_run=2, precision=64, log_every_n_steps=1)
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)


@RunIf(min_gpus=2)
def test_double_precision_ddp(tmpdir):
    model = DoublePrecisionBoringModel()

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        strategy="ddp_spawn",
        gpus=2,
        fast_dev_run=2,
        precision=64,
        log_every_n_steps=1,
    )
    trainer.fit(model)


def test_double_precision_pickle(tmpdir):
    model = BoringModel()
    plugin = DoublePrecisionPlugin()
    model, _, __ = plugin.connect(model, MagicMock(), MagicMock())
    pickle.dumps(model)
