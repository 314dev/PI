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
from unittest import mock

import pytest
import torch

import tests.helpers.utils as tutils
from pi_ml import Trainer
from pi_ml.accelerators import CPUAccelerator, GPUAccelerator, IPUAccelerator, TPUAccelerator
from pi_ml.utilities.seed import seed_everything
from tests.accelerators.test_dp import CustomClassificationModelDP
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf


@pytest.mark.parametrize(
    "trainer_kwargs",
    (
        pytest.param(dict(gpus=1), marks=RunIf(min_gpus=1)),
        pytest.param(dict(strategy="dp", gpus=2), marks=RunIf(min_gpus=2)),
        pytest.param(dict(strategy="ddp_spawn", gpus=2), marks=RunIf(min_gpus=2)),
    ),
)
def test_evaluate(tmpdir, trainer_kwargs):
    tutils.set_random_main_port()
    seed_everything(1)
    dm = ClassifDataModule()
    model = CustomClassificationModelDP()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=2, limit_train_batches=10, limit_val_batches=10, **trainer_kwargs
    )

    trainer.fit(model, datamodule=dm)
    assert "ckpt" in trainer.checkpoint_callback.best_model_path

    old_weights = model.layer_0.weight.clone().detach().cpu()

    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)

    # make sure weights didn't change
    new_weights = model.layer_0.weight.clone().detach().cpu()
    torch.testing.assert_allclose(old_weights, new_weights)


def test_model_parallel_setup_called(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.configure_sharded_model_called = False
            self.layer = None

        def configure_sharded_model(self):
            self.configure_sharded_model_called = True
            self.layer = torch.nn.Linear(32, 2)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=1)
    trainer.fit(model)

    assert model.configure_sharded_model_called


@mock.patch("torch.cuda.device_count", return_value=2)
def test_auto_device_count(device_count_mock):
    assert CPUAccelerator.auto_device_count() == 1
    assert GPUAccelerator.auto_device_count() == 2
    assert TPUAccelerator.auto_device_count() == 8
    assert IPUAccelerator.auto_device_count() == 4
