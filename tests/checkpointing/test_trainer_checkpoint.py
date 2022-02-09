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

import torch

import pi_ml as pl
from pi_ml import Trainer
from pi_ml.callbacks import ModelCheckpoint
from tests.helpers import BoringModel


def test_finetuning_with_ckpt_path(tmpdir):
    """This test validates that generated ModelCheckpoint is pointing to the right best_model_path during test."""

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=tmpdir, filename="{epoch:02d}", save_top_k=-1)

    class ExtendedBoringModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    model = ExtendedBoringModel()
    model.validation_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=12,
        limit_val_batches=6,
        limit_test_batches=12,
        callbacks=[checkpoint_callback],
        logger=False,
    )
    trainer.fit(model)
    assert os.listdir(tmpdir) == ["epoch=00.ckpt"]

    best_model_paths = [checkpoint_callback.best_model_path]
    for idx in range(3, 6):
        # load from checkpoint
        trainer = pl.Trainer(
            default_root_dir=tmpdir,
            max_epochs=idx,
            limit_train_batches=12,
            limit_val_batches=12,
            limit_test_batches=12,
            enable_progress_bar=False,
        )
        trainer.fit(model, ckpt_path=best_model_paths[-1])
        trainer.test()
        best_model_paths.append(trainer.checkpoint_callback.best_model_path)

    for idx, best_model_path in enumerate(best_model_paths):
        if idx == 0:
            assert best_model_path.endswith(f"epoch=0{idx}.ckpt")
        else:
            assert f"epoch={idx + 1}" in best_model_path


def test_accumulated_gradient_batches_with_ckpt_path(tmpdir):
    """This test validates that accumulated gradient is properly recomputed and reset on the trainer."""

    ckpt = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    model = BoringModel()
    trainer_kwargs = dict(
        max_epochs=1, accumulate_grad_batches={0: 2}, callbacks=ckpt, limit_train_batches=1, limit_val_batches=0
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    trainer_kwargs["max_epochs"] = 2
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, ckpt_path=ckpt.last_model_path)
