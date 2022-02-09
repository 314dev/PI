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
import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pi_ml.callbacks import EarlyStopping
from pi_ml.trainer import Trainer
from pi_ml.utilities import memory
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


@RunIf(min_gpus=2)
def test_multi_gpu_early_stop_ddp_spawn(tmpdir):
    tutils.set_random_main_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping(monitor="train_acc")],
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        strategy="ddp_spawn",
    )

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_gpus=2)
def test_multi_gpu_model_ddp_spawn(tmpdir):
    tutils.set_random_main_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        strategy="ddp_spawn",
        enable_progress_bar=False,
    )

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model)

    # test memory helper functions
    memory.get_memory_profile("min_max")


@RunIf(min_gpus=2)
def test_ddp_all_dataloaders_passed_to_fit(tmpdir):
    """Make sure DDP works with dataloaders passed to fit()"""
    tutils.set_random_main_port()

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        gpus=[0, 1],
        strategy="ddp_spawn",
    )
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
    assert trainer.state.finished, "DDP doesn't work with dataloaders passed to fit()."
