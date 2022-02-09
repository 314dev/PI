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

import pytest

from pi_ml import Trainer
from pi_ml.trainer.states import RunningStage
from tests.helpers.boring_model import BoringModel


def test_num_dataloader_batches(tmpdir):
    """Tests that the correct number of batches are allocated."""
    # when we have fewer batches in the dataloader we should use those instead of the limit
    model = BoringModel()
    trainer = Trainer(limit_val_batches=100, limit_train_batches=100, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert isinstance(trainer.num_val_batches, list)
    assert trainer.num_val_batches[0] == 64
    assert trainer.num_training_batches == 64

    # when we have more batches in the dataloader we should limit them
    model = BoringModel()
    trainer = Trainer(limit_val_batches=7, limit_train_batches=7, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert isinstance(trainer.num_val_batches, list)
    assert trainer.num_val_batches[0] == 7
    assert trainer.num_training_batches == 7


@pytest.mark.parametrize(
    ["stage", "mode"],
    [
        (RunningStage.VALIDATING, "val"),
        (RunningStage.TESTING, "test"),
        (RunningStage.PREDICTING, "predict"),
    ],
)
@pytest.mark.parametrize("limit_batches", [0.1, 10])
def test_eval_limit_batches(stage, mode, limit_batches):
    limit_eval_batches = f"limit_{mode}_batches"
    dl_hook = f"{mode}_dataloader"
    model = BoringModel()
    eval_loader = getattr(model, dl_hook)()

    trainer = Trainer(**{limit_eval_batches: limit_batches})
    model.trainer = trainer
    trainer._data_connector.attach_dataloaders(model)
    loader_num_batches, dataloaders = trainer._data_connector._reset_eval_dataloader(stage, model=model)
    expected_batches = int(limit_batches * len(eval_loader)) if isinstance(limit_batches, float) else limit_batches
    assert loader_num_batches[0] == expected_batches
    assert len(dataloaders[0]) == len(eval_loader)
