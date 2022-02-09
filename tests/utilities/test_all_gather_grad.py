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
import sys

import numpy as np
import torch

from pi_ml import seed_everything, Trainer
from pi_ml.utilities import AllGatherGrad
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


def setup_ddp(rank, world_size):
    """Setup ddp enviroment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8088"

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _test_all_gather_ddp(rank, world_size):
    setup_ddp(rank, world_size)

    tensor1 = torch.ones(8, requires_grad=True)
    tensor2 = torch.ones((8, 16, 32), requires_grad=True)

    tensor1_gathered = AllGatherGrad.apply(tensor1)
    tensor2_gathered = AllGatherGrad.apply(tensor2)

    tensor1_gathered = tensor1_gathered * rank
    tensor2_gathered = tensor2_gathered * rank

    tensor1_gathered.sum().backward()
    tensor2_gathered.sum().backward()

    grad1 = torch.zeros_like(tensor1.grad).fill_(torch.arange(world_size).sum().float())
    grad2 = torch.zeros_like(tensor2.grad).fill_(torch.arange(world_size).sum().float())

    assert torch.allclose(grad1, tensor1.grad)
    assert torch.allclose(grad2, tensor2.grad)


@RunIf(skip_windows=True, skip_49370=True, skip_hanging_spawn=True)
def test_all_gather_ddp_spawn():
    world_size = 3
    torch.multiprocessing.spawn(_test_all_gather_ddp, args=(world_size,), nprocs=world_size)


@RunIf(min_gpus=2, skip_windows=True, standalone=True)
def test_all_gather_collection(tmpdir):
    class TestModel(BoringModel):

        training_epoch_end_called = False

        def training_epoch_end(self, outputs) -> None:
            losses = torch.stack([x["loss"] for x in outputs])
            gathered_loss = self.all_gather(
                {
                    "losses_tensor_int": torch.rand(2, 2).int().t(),
                    "losses_tensor_float": torch.rand(2, 2).t(),
                    "losses_np_ndarray": np.array([1, 2, 3]),
                    "losses_bool": [True, False],
                    "losses_float": [0.0, 1.0, 2.0],
                    "losses_int": [0, 1, 2],
                    "losses": losses,
                    "losses_list": [losses, losses],
                }
            )
            assert gathered_loss["losses_tensor_int"][0].dtype == torch.int32
            assert gathered_loss["losses_tensor_float"][0].dtype == torch.float
            assert gathered_loss["losses_np_ndarray"][0].dtype == torch.int64
            # torch.bool can't be all_gathered
            assert gathered_loss["losses_bool"][0].dtype == torch.uint8
            assert gathered_loss["losses_float"][0].dtype == torch.float
            assert gathered_loss["losses_int"][0].dtype == torch.int
            assert gathered_loss["losses_list"][0].numel() == 2 * len(losses)
            assert gathered_loss["losses"].numel() == 2 * len(losses)
            self.training_epoch_end_called = True

    seed_everything(42)

    model = TestModel()

    limit_train_batches = 8
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
    )

    trainer.fit(model)
    assert model.training_epoch_end_called


@RunIf(min_gpus=2, skip_windows=True, standalone=True)
def test_all_gather_sync_grads(tmpdir):
    class TestModel(BoringModel):

        training_step_called = False

        def training_step(self, batch, batch_idx):
            self.training_step_called = True
            tensor = torch.rand(2, 2, requires_grad=True, device=self.device)
            gathered_tensor = self.all_gather(tensor, sync_grads=True)
            assert gathered_tensor.shape == torch.Size([2, 2, 2])

            loss = gathered_tensor.sum()

            return loss

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, accelerator="gpu", devices=2, strategy="ddp")
    trainer.fit(model)
    assert model.training_step_called
