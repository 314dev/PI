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
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader

import tests.helpers.utils as tutils
from pi_ml import Trainer
from pi_ml.tuner.tuning import Tuner
from pi_ml.utilities import AMPType
from pi_ml.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringDataModule, BoringModel, RandomDataset
from tests.helpers.runif import RunIf


class BatchSizeDataModule(BoringDataModule):
    def __init__(self, batch_size):
        super().__init__()
        if batch_size is not None:
            self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.random_train, batch_size=getattr(self, "batch_size", 1))


class BatchSizeModel(BoringModel):
    def __init__(self, batch_size):
        super().__init__()
        if batch_size is not None:
            self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))


@pytest.mark.parametrize(["model_bs", "dm_bs"], [(2, -1), (2, 2), (2, None), (None, 2), (16, 16)])
def test_scale_batch_size_method_with_model_or_datamodule(tmpdir, model_bs, dm_bs):
    """Test the tuner method `Tuner.scale_batch_size` with a datamodule."""
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=0, max_epochs=1)
    tuner = Tuner(trainer)

    model = BatchSizeModel(model_bs)
    datamodule = BatchSizeDataModule(dm_bs) if dm_bs != -1 else None

    new_batch_size = tuner.scale_batch_size(model, mode="binsearch", init_val=4, max_trials=2, datamodule=datamodule)
    assert new_batch_size == 16

    if model_bs is not None:
        assert model.batch_size == new_batch_size
        if dm_bs == -1:
            # datamodule batch size takes precedence
            assert trainer.train_dataloader.loaders.batch_size == new_batch_size
    if dm_bs not in (-1, None):
        assert datamodule.batch_size == new_batch_size
        assert trainer.train_dataloader.loaders.batch_size == new_batch_size


def test_model_reset_correctly(tmpdir):
    """Check that model weights are correctly reset after scaling batch size."""
    tutils.reset_seed()

    model = BatchSizeModel(batch_size=2)

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    before_state_dict = deepcopy(model.state_dict())

    trainer.tuner.scale_batch_size(model, max_trials=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(
            torch.eq(before_state_dict[key], after_state_dict[key])
        ), "Model was not reset correctly after scaling batch size"

    assert not any(f for f in os.listdir(tmpdir) if f.startswith(".scale_batch_size"))


def test_trainer_reset_correctly(tmpdir):
    """Check that all trainer parameters are reset correctly after scaling batch size."""
    tutils.reset_seed()

    model = BatchSizeModel(batch_size=2)

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    changed_attributes = [
        "callbacks",
        "checkpoint_callback",
        "current_epoch",
        "limit_train_batches",
        "logger",
        "max_steps",
        "global_step",
    ]
    expected = {ca: getattr(trainer, ca) for ca in changed_attributes}
    trainer.tuner.scale_batch_size(model, max_trials=5)
    actual = {ca: getattr(trainer, ca) for ca in changed_attributes}

    assert actual == expected


@RunIf(min_gpus=1)
@pytest.mark.parametrize("scale_arg", ["power", "binsearch", True])
def test_auto_scale_batch_size_trainer_arg(tmpdir, scale_arg):
    """Test possible values for 'batch size auto scaling' Trainer argument."""
    tutils.reset_seed()
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, auto_scale_batch_size=scale_arg, accelerator="gpu", devices=1
    )
    trainer.tune(model)
    after_batch_size = model.batch_size
    assert before_batch_size != after_batch_size, "Batch size was not altered after running auto scaling of batch size"

    assert not os.path.exists(tmpdir / "scale_batch_size_temp_model.ckpt")


@RunIf(min_gpus=1)
@pytest.mark.parametrize("use_hparams", [True, False])
def test_auto_scale_batch_size_set_model_attribute(tmpdir, use_hparams):
    """Test that new batch size gets written to the correct hyperparameter attribute."""
    tutils.reset_seed()

    hparams = {"batch_size": 2}
    before_batch_size = hparams.get("batch_size")

    class HparamsBatchSizeModel(BatchSizeModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_hyperparameters()

        def dataloader(self, *args, **kwargs):
            # artificially set batch_size so we can get a dataloader
            # remove it immediately after, because we want only self.hparams.batch_size
            setattr(self, "batch_size", before_batch_size)
            dataloader = super().dataloader(*args, **kwargs)
            del self.batch_size
            return dataloader

    class HparamsBatchSizeDataModule(BoringDataModule):
        def __init__(self, data_dir, batch_size):
            super().__init__(data_dir)
            self.batch_size = batch_size

        def train_dataloader(self):
            return DataLoader(self.random_train, batch_size=self.batch_size)

    datamodule_fit = HparamsBatchSizeDataModule(data_dir=tmpdir, batch_size=before_batch_size)
    model_class = HparamsBatchSizeModel if use_hparams else BatchSizeModel
    model = model_class(**hparams)

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, auto_scale_batch_size=True, accelerator="gpu", devices=1)
    trainer.tune(model, datamodule_fit)
    after_batch_size = model.hparams.batch_size if use_hparams else model.batch_size
    assert trainer.datamodule == datamodule_fit
    assert before_batch_size != after_batch_size
    assert after_batch_size <= len(trainer.train_dataloader.dataset)
    assert datamodule_fit.batch_size == after_batch_size


def test_auto_scale_batch_size_duplicate_attribute_warning(tmpdir):
    """Test for a warning when model.batch_size and model.hparams.batch_size both present."""

    class TestModel(BoringModel):
        def __init__(self, batch_size=1):
            super().__init__()
            # now we have model.batch_size and model.hparams.batch_size
            self.batch_size = 1
            self.save_hyperparameters()

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, max_epochs=1000, auto_scale_batch_size=True)
    expected_message = "Field `model.batch_size` and `model.hparams.batch_size` are mutually exclusive!"
    with pytest.warns(UserWarning, match=expected_message):
        trainer.tune(model)


@pytest.mark.parametrize("scale_method", ["power", "binsearch"])
def test_call_to_trainer_method(tmpdir, scale_method):
    """Test that calling the trainer method itself works."""
    tutils.reset_seed()

    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    after_batch_size = trainer.tuner.scale_batch_size(model, mode=scale_method, max_trials=5)
    model.batch_size = after_batch_size
    trainer.fit(model)

    assert before_batch_size != after_batch_size, "Batch size was not altered after running auto scaling of batch size"


def test_error_on_dataloader_passed_to_fit(tmpdir):
    """Verify that when the auto scale batch size feature raises an error if a train dataloader is passed to
    fit."""

    # only train passed to fit
    model = BatchSizeModel(batch_size=2)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        auto_scale_batch_size="power",
    )
    fit_options = dict(train_dataloaders=model.train_dataloader())

    with pytest.raises(
        MisconfigurationException,
        match="The batch scaling feature cannot be used with dataloaders passed directly",
    ):
        trainer.tune(model, **fit_options)


@RunIf(min_gpus=1)
def test_auto_scale_batch_size_with_amp(tmpdir):
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(
        default_root_dir=tmpdir, max_steps=1, auto_scale_batch_size=True, accelerator="gpu", devices=1, precision=16
    )
    trainer.tune(model)
    after_batch_size = model.batch_size
    assert trainer.amp_backend == AMPType.NATIVE
    assert trainer.scaler is not None
    assert after_batch_size != before_batch_size


def test_scale_batch_size_no_trials(tmpdir):
    """Check the result is correct even when no trials are run."""
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_val_batches=1, limit_train_batches=1, auto_scale_batch_size="power"
    )
    model = BatchSizeModel(batch_size=2)
    result = trainer.tuner.scale_batch_size(model, max_trials=0)
    assert result == 2


def test_scale_batch_size_fails_with_unavailable_mode(tmpdir):
    """Check the tuning raises error when called with mode that does not exist."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.batch_size = 2

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=1,
        auto_scale_batch_size="ThisModeDoesNotExist",
    )

    with pytest.raises(ValueError, match="could either be `power` or `binsearch`"):
        trainer.tune(model)
    with pytest.raises(ValueError, match="could either be `power` or `binsearch`"):
        trainer.tuner.scale_batch_size(model, mode="ThisModeDoesNotExist")


@pytest.mark.parametrize("scale_method", ["power", "binsearch"])
def test_dataloader_reset_with_scale_batch_size(tmpdir, scale_method):
    """Test that train and val dataloaders are reset at every update in scale batch size."""
    model = BatchSizeModel(batch_size=16)
    scale_batch_size_kwargs = {"max_trials": 5, "init_val": 4, "mode": scale_method}

    trainer = Trainer(max_epochs=2, auto_scale_batch_size=True)
    new_batch_size = trainer.tune(model, scale_batch_size_kwargs=scale_batch_size_kwargs)["scale_batch_size"]

    assert trainer.train_dataloader.loaders.batch_size == new_batch_size
    assert trainer.val_dataloaders[0].batch_size == new_batch_size
