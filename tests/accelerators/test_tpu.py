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
# limitations under the License
import collections
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pi_ml import Trainer
from pi_ml.accelerators.cpu import CPUAccelerator
from pi_ml.accelerators.tpu import TPUAccelerator
from pi_ml.plugins import TPUPrecisionPlugin, XLACheckpointIO
from pi_ml.strategies import DDPStrategy, TPUSpawnStrategy
from pi_ml.utilities import find_shared_parameters
from pi_ml.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf
from tests.helpers.utils import pl_multi_process_test


class WeightSharingModule(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(32, 10, bias=False)
        self.layer_2 = nn.Linear(10, 32, bias=False)
        self.layer_3 = nn.Linear(32, 10, bias=False)
        self.layer_3.weight = self.layer_1.weight

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


@RunIf(tpu=True)
@pl_multi_process_test
def test_resume_training_on_cpu(tmpdir):
    """Checks if training can be resumed from a saved checkpoint on CPU."""
    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(max_epochs=1, tpu_cores=8)
    trainer.fit(model)

    model_path = trainer.checkpoint_callback.best_model_path

    # Verify saved Tensors are on CPU
    ckpt = torch.load(model_path)
    weight_tensor = list(ckpt["state_dict"].values())[0]
    assert weight_tensor.device == torch.device("cpu")

    # Verify that training is resumed on CPU
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model, ckpt_path=model_path)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(tpu=True)
@pl_multi_process_test
def test_if_test_works_after_train(tmpdir):
    """Ensure that .test() works after .fit()"""

    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(max_epochs=1, tpu_cores=8, default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    assert len(trainer.test(model)) == 1


@RunIf(tpu=True)
def test_accelerator_tpu():

    trainer = Trainer(accelerator="tpu", tpu_cores=8)

    assert trainer._device_type == "tpu"
    assert isinstance(trainer.accelerator, TPUAccelerator)

    with pytest.raises(
        MisconfigurationException, match="You passed `accelerator='tpu'`, but you didn't pass `tpu_cores` to `Trainer`"
    ):
        trainer = Trainer(accelerator="tpu")


@RunIf(tpu=True)
def test_accelerator_cpu_with_tpu_cores_flag():

    trainer = Trainer(accelerator="cpu", tpu_cores=8)

    assert trainer._device_type == "cpu"
    assert isinstance(trainer.accelerator, CPUAccelerator)


@RunIf(tpu=True)
def test_accelerator_tpu_with_auto():

    trainer = Trainer(accelerator="auto", tpu_cores=8)

    assert trainer._device_type == "tpu"
    assert isinstance(trainer.accelerator, TPUAccelerator)


@RunIf(tpu=True)
def test_accelerator_tpu_with_devices():

    trainer = Trainer(accelerator="tpu", devices=8)

    assert trainer.tpu_cores == 8
    assert isinstance(trainer.strategy, TPUSpawnStrategy)
    assert isinstance(trainer.accelerator, TPUAccelerator)


@RunIf(tpu=True)
def test_accelerator_auto_with_devices_tpu():

    trainer = Trainer(accelerator="auto", devices=8)

    assert trainer._device_type == "tpu"
    assert trainer.tpu_cores == 8


@RunIf(tpu=True)
def test_accelerator_tpu_with_tpu_cores_priority():
    """Test for checking `tpu_cores` flag takes priority over `devices`."""

    tpu_cores = 8
    with pytest.warns(UserWarning, match="The flag `devices=1` will be ignored,"):
        trainer = Trainer(accelerator="tpu", devices=1, tpu_cores=tpu_cores)

    assert trainer.tpu_cores == tpu_cores


@RunIf(tpu=True)
def test_set_devices_if_none_tpu():

    trainer = Trainer(accelerator="tpu", tpu_cores=8)
    assert trainer.devices == 8


@RunIf(tpu=True)
def test_manual_optimization_tpus(tmpdir):
    class ManualOptimizationModel(BoringModel):

        count = 0
        called = collections.defaultdict(int)

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        @property
        def should_update(self):
            return self.count % 2 == 0

        def on_train_batch_start(self, batch, batch_idx):
            self.called["on_train_batch_start"] += 1
            self.weight_before = self.layer.weight.clone()

        def training_step(self, batch, batch_idx):
            self.called["training_step"] += 1
            opt = self.optimizers()
            output = self.layer(batch)
            loss = self.loss(batch, output)

            if self.should_update:
                self.manual_backward(loss)
                opt.step()
                opt.zero_grad()
            return loss

        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.called["on_train_batch_end"] += 1
            after_before = self.layer.weight.clone()
            if self.should_update:
                assert not torch.equal(self.weight_before, after_before), self.count
            else:
                assert torch.equal(self.weight_before, after_before)
            assert torch.all(self.layer.weight.grad == 0)
            self.count += 1

        def on_train_start(self):
            opt = self.optimizers()
            self.opt_step_patch = patch.object(opt, "step", wraps=opt.step)
            self.opt_step_mock = self.opt_step_patch.start()

        def on_train_end(self):
            assert self.called["training_step"] == 5
            assert self.called["on_train_batch_start"] == 5
            assert self.called["on_train_batch_end"] == 5

            self.opt_step_patch.stop()
            assert self.opt_step_mock.call_count == 3

    model = ManualOptimizationModel()
    model_copy = deepcopy(model)
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=5,
        limit_test_batches=0,
        limit_val_batches=0,
        tpu_cores=8,
    )
    trainer.fit(model)

    for param, param_copy in zip(model.parameters(), model_copy.parameters()):
        assert not torch.equal(param.cpu().data, param_copy.data)


@RunIf(tpu=True)
def test_ddp_cpu_not_supported_on_tpus():
    with pytest.raises(MisconfigurationException, match="`accelerator='ddp_cpu'` is not supported on TPU machines"):
        Trainer(accelerator="ddp_cpu")


@RunIf(tpu=True)
@pytest.mark.parametrize("strategy", ["ddp_spawn", "tpu_spawn_debug"])
def test_strategy_choice_tpu_str(tmpdir, strategy):
    trainer = Trainer(strategy=strategy, accelerator="tpu", devices=8)
    assert isinstance(trainer.strategy, TPUSpawnStrategy)


@RunIf(tpu=True)
def test_strategy_choice_tpu_strategy(tmpdir):
    trainer = Trainer(strategy=TPUSpawnStrategy(), accelerator="tpu", devices=8)
    assert isinstance(trainer.strategy, TPUSpawnStrategy)


@RunIf(tpu=True)
def test_auto_parameters_tying_tpus(tmpdir):

    model = WeightSharingModule()
    shared_params = find_shared_parameters(model)

    assert shared_params[0] == ["layer_1.weight", "layer_3.weight"]

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=5, tpu_cores=8, max_epochs=1)
    trainer.fit(model)

    assert torch.all(torch.eq(model.layer_1.weight, model.layer_3.weight))


@RunIf(tpu=True)
def test_auto_parameters_tying_tpus_nested_module(tmpdir):
    class SubModule(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return self.layer(x)

    class NestedModule(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 10, bias=False)
            self.net_a = SubModule(self.layer)
            self.layer_2 = nn.Linear(10, 32, bias=False)
            self.net_b = SubModule(self.layer)

        def forward(self, x):
            x = self.net_a(x)
            x = self.layer_2(x)
            x = self.net_b(x)
            return x

    model = NestedModule()

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=5, tpu_cores=8, max_epochs=1)
    trainer.fit(model)

    assert torch.all(torch.eq(model.net_a.layer.weight, model.net_b.layer.weight))


def test_tpu_invalid_raises():
    training_type_plugin = TPUSpawnStrategy(accelerator=TPUAccelerator(), precision_plugin=Mock())
    with pytest.raises(ValueError, match="TPUAccelerator` can only be used with a `TPUPrecisionPlugin"):
        Trainer(strategy=training_type_plugin)

    training_type_plugin = DDPStrategy(accelerator=TPUAccelerator(), precision_plugin=TPUPrecisionPlugin())
    with pytest.raises(ValueError, match="TPUAccelerator` can only be used with a `SingleTPUStrategy`"):
        Trainer(strategy=training_type_plugin)


def test_tpu_invalid_raises_set_precision_with_strategy():
    accelerator = TPUAccelerator()
    training_type_plugin = TPUSpawnStrategy(accelerator=accelerator, precision_plugin=object())
    with pytest.raises(ValueError, match="`TPUAccelerator` can only be used with a `TPUPrecisionPlugin`"):
        Trainer(strategy=training_type_plugin)

    accelerator = TPUAccelerator()
    training_type_plugin = DDPStrategy(accelerator=accelerator, precision_plugin=TPUPrecisionPlugin())
    with pytest.raises(
        ValueError, match="The `TPUAccelerator` can only be used with a `SingleTPUStrategy` or `TPUSpawnStrategy"
    ):
        Trainer(strategy=training_type_plugin)


@RunIf(tpu=True)
def test_xla_checkpoint_plugin_being_default():
    trainer = Trainer(tpu_cores=8)
    assert isinstance(trainer.strategy.checkpoint_io, XLACheckpointIO)


@RunIf(tpu=True)
@patch("pi_ml.strategies.tpu_spawn.xm")
def test_mp_device_dataloader_attribute(_):
    dataset = RandomDataset(32, 64)
    dataloader = TPUSpawnStrategy().process_dataloader(DataLoader(dataset))
    assert dataloader.dataset == dataset


@RunIf(tpu=True)
def test_devices_auto_choice_tpu():
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.devices == 8
    assert trainer.tpu_cores == 8
