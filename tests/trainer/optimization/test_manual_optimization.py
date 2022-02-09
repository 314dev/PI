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
import collections
from copy import deepcopy
from unittest import mock
from unittest.mock import ANY, call, patch

import pytest
import torch
import torch.distributed as torch_distrib
import torch.nn.functional as F

from pi_ml import seed_everything, Trainer
from pi_ml.strategies import Strategy
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class ManualOptModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt_a, opt_b = self.optimizers()

        # make sure there are no grads
        if batch_idx > 0:
            assert torch.all(self.layer.weight.grad == 0)

        loss_1 = self.step(batch[0])
        self.manual_backward(loss_1)
        opt_a.step()
        opt_a.zero_grad()
        assert torch.all(self.layer.weight.grad == 0)

        loss_2 = self.step(batch[0])
        # ensure we forward the correct params to the optimizer
        # without retain_graph we can't do multiple backward passes
        self.manual_backward(loss_2, retain_graph=True)
        self.manual_backward(loss_2)
        assert self.layer.weight.grad is not None
        opt_b.step()
        opt_b.zero_grad()
        assert torch.all(self.layer.weight.grad == 0)

        return loss_2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return optimizer, optimizer_2


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        pytest.param({"gpus": 1, "precision": 16, "amp_backend": "native"}, marks=RunIf(min_gpus=1)),
        pytest.param(
            {"gpus": 1, "precision": 16, "amp_backend": "apex", "amp_level": "O2"},
            marks=RunIf(amp_apex=True, min_gpus=1),
        ),
    ],
)
def test_multiple_optimizers_manual_no_return(tmpdir, kwargs):
    apex_optimizer_patches = []
    apex_optimizer_steps = []

    class TestModel(ManualOptModel):
        def training_step(self, batch, batch_idx):
            # avoid returning a value
            super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs):
            # outputs is empty as training_step does not return
            # and it is not automatic optimization
            assert not outputs

        def on_train_start(self):
            if kwargs.get("amp_backend") != "apex":
                return
            # extremely ugly. APEX patches all the native torch optimizers on `_initialize` which we call on
            # `ApexMixedPrecisionPlugin.dispatch`. Additionally, their replacement `new_step` functions are locally
            # defined so can't even patch those, thus we need to create the mock after APEX has been initialized
            nonlocal apex_optimizer_patches, apex_optimizer_steps
            for opt in self.trainer.optimizers:
                # `amp.scale_loss` will also patch the step to avoid it when gradient overflow happens. avoid it
                opt._amp_stash.already_patched = True
                patch = mock.patch.object(opt, "step")
                apex_optimizer_patches.append(patch)
                apex_optimizer_steps.append(patch.start())

        def on_train_end(self):
            if kwargs.get("amp_backend") == "apex":
                for p in apex_optimizer_patches:
                    p.stop()

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        **kwargs,
    )

    if kwargs.get("amp_backend") == "native":
        # mock the scaler instead of the optimizer step because it can be skipped with NaNs
        scaler_step_patch = mock.patch.object(
            trainer.precision_plugin.scaler, "step", wraps=trainer.precision_plugin.scaler.step
        )
        scaler_step = scaler_step_patch.start()

    with mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward) as bwd_mock:
        trainer.fit(model)
    assert bwd_mock.call_count == limit_train_batches * 3

    if kwargs.get("amp_backend") == "native":
        scaler_step_patch.stop()
        assert scaler_step.call_count == len(model.optimizers()) * limit_train_batches
    if kwargs.get("amp_backend") == "apex":
        assert [s.call_count for s in apex_optimizer_steps] == [len(model.optimizers())] * limit_train_batches


def test_multiple_optimizers_manual_return(tmpdir):
    class TestModel(ManualOptModel):
        def training_step(self, batch, batch_idx):
            super().training_step(batch, batch_idx)
            return {"something": "else"}

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert outputs == [{"something": "else"}, {"something": "else"}]

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    with mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward) as bwd_mock:
        trainer.fit(model)
    assert bwd_mock.call_count == limit_train_batches * 3


def test_multiple_optimizers_manual_log(tmpdir):
    class TestModel(ManualOptModel):
        def training_step(self, batch, batch_idx):
            loss_2 = super().training_step(batch, batch_idx)
            self.log("a", loss_2, on_epoch=True)

        def training_epoch_end(self, outputs) -> None:
            assert not outputs

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    with mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward) as bwd_mock:
        trainer.fit(model)
    assert bwd_mock.call_count == limit_train_batches * 3
    assert set(trainer.logged_metrics) == {"a_step", "a_epoch"}


@RunIf(min_gpus=1)
def test_multiple_optimizers_manual_native_amp(tmpdir):
    model = ManualOptModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        precision=16,
        gpus=1,
    )

    with mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward) as bwd_mock:
        trainer.fit(model)
    assert bwd_mock.call_count == limit_train_batches * 3


class ManualOptimizationExtendedModel(BoringModel):

    count = 0
    called = collections.defaultdict(int)
    detach = False

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
        loss /= loss.clone().detach()
        loss *= 0.1

        if self.should_update:

            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()

        return loss.detach() if self.detach else loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.called["on_train_batch_end"] += 1
        after_before = self.layer.weight.clone()
        if self.should_update:
            try:
                assert not torch.equal(self.weight_before, after_before), self.count
            # todo: specify the possible exception
            except Exception:
                # TODO: Figure out why 1 every 3 runs, weights don't get updated on count = 4"
                pass
        else:
            try:
                assert torch.equal(self.weight_before, after_before)
            # todo: specify the possible exception
            except Exception:
                # almost no diff between before and after
                assert torch.abs(torch.sum(self.weight_before) - torch.sum(after_before)).item() < 10e-6
        assert torch.all(self.layer.weight.grad == 0)
        self.count += 1

    def on_train_end(self):
        assert self.called["training_step"] == 10
        assert self.called["on_train_batch_start"] == 10
        assert self.called["on_train_batch_end"] == 10


@RunIf(min_gpus=2)
def test_manual_optimization_and_return_tensor(tmpdir):
    """This test verify that in `manual_optimization` we don't add gradient when the user return loss in
    `training_step`"""

    model = ManualOptimizationExtendedModel()
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_test_batches=0,
        limit_val_batches=0,
        precision=16,
        amp_backend="native",
        strategy="ddp_spawn",
        gpus=2,
    )
    trainer.fit(model)


@RunIf(min_gpus=1)
def test_manual_optimization_and_accumulated_gradient(tmpdir):
    """This test verify that in `automatic_optimization=False`, step is being called only when we shouldn't
    accumulate."""
    seed_everything(234)

    class ExtendedModel(BoringModel):

        count = 1
        called = collections.defaultdict(int)
        detach = False

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        @property
        def should_update(self):
            return self.count % 2 == 0

        @property
        def should_have_updated(self):
            return self.count % 4 == 0

        @property
        def has_gradient(self):
            return self.layer.weight.grad is not None

        def on_train_batch_start(self, batch, batch_idx):
            self.called["on_train_batch_start"] += 1
            self.weight_before = self.layer.weight.clone()

        def training_step(self, batch, batch_idx):
            self.called["training_step"] += 1
            opt = self.optimizers()
            output = self.layer(batch)

            loss = self.loss(batch, output)
            loss /= loss.clone().detach()
            loss *= 0.1

            if self.should_update:

                self.manual_backward(loss)
                if self.should_have_updated:
                    opt.step()
                    opt.zero_grad()

            return loss.detach() if self.detach else loss

        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.called["on_train_batch_end"] += 1
            after_before = self.layer.weight.clone()
            if self.should_update and self.should_have_updated:
                assert not torch.equal(self.weight_before, after_before), self.count
                assert torch.all(self.layer.weight.grad == 0)
            else:
                assert torch.equal(self.weight_before, after_before)
                if self.count > 1:
                    if self.count % 4 == 1:
                        assert torch.all(self.layer.weight.grad == 0)
                    else:
                        assert torch.sum(self.layer.weight.grad) != 0
            self.count += 1

        def on_train_epoch_end(self, *_, **__):
            assert self.called["training_step"] == 20
            assert self.called["on_train_batch_start"] == 20
            assert self.called["on_train_batch_end"] == 20

    model = ExtendedModel()
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=20,
        limit_test_batches=0,
        limit_val_batches=0,
        precision=16,
        amp_backend="native",
        gpus=1,
    )
    trainer.fit(model)


@RunIf(min_gpus=1)
def test_multiple_optimizers_step(tmpdir):
    """Tests that `step` works with several optimizers."""

    class TestModel(ManualOptModel):
        def training_step(self, batch, batch_idx):
            opt_a, opt_b = self.optimizers()
            x = batch[0]

            loss_1 = self(x)
            loss_1 = self.loss(loss_1, loss_1)

            # make sure there are no grads
            if self.layer.weight.grad is not None:
                assert torch.all(self.layer.weight.grad == 0)

            self.manual_backward(loss_1)
            opt_a.step()

            # fake discriminator
            loss_2 = self(x)
            loss_2 = self.loss(loss_2, loss_2)

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.manual_backward(loss_2, retain_graph=True)
            self.manual_backward(loss_2, retain_graph=True)

            assert self.layer.weight.grad is not None
            opt_b.step()
            opt_b.zero_grad()

            return {"loss1": loss_1.detach(), "loss2": loss_2.detach()}

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        # sister test: tests/plugins/test_amp_plugins.py::test_amp_gradient_unscale
        def on_after_backward(self) -> None:
            # check grads are scaled
            scale = self.trainer.precision_plugin.scaler.get_scale()
            assert scale != 1.0  # the return value if not enabled
            grads = [p.grad for p in self.parameters()]
            inv_scale = 1 / scale
            self.original_grads = [p * inv_scale for p in grads]

        def check_grads_unscaled(self, optimizer=None):
            if optimizer is not None:
                scaler = self.trainer.precision_plugin.scaler
                state = scaler._per_optimizer_states[id(optimizer)]
                assert state["stage"].name == "UNSCALED"

            grads = [p.grad for p in self.parameters()]
            assert len(grads) == len(self.original_grads)
            for actual, expected in zip(grads, self.original_grads):
                torch.testing.assert_allclose(actual, expected)

        def on_before_optimizer_step(self, optimizer, *_):
            self.check_grads_unscaled(optimizer)

        def log_grad_norm(self, grad_norm_dict):
            self.check_grads_unscaled()
            assert len(grad_norm_dict)

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        precision=16,
        amp_backend="native",
        gpus=1,
        track_grad_norm=2,
    )

    with mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward) as bwd_mock:
        trainer.fit(model)
    assert bwd_mock.call_count == limit_train_batches * 3


def test_step_with_optimizer_closure(tmpdir):
    """Tests that `step` works with optimizer_closure."""

    class TestModel(BoringModel):

        _losses = []

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            # make sure there are no grads
            if self.layer.weight.grad is not None:
                assert torch.all(self.layer.weight.grad == 0)

            opt = self.optimizers()

            def compute_loss():
                x = batch[0]
                x = F.dropout(x, 0.1)
                predictions = self(x)
                predictions = F.dropout(predictions, 0.1)
                loss = self.loss(None, predictions)
                return loss

            def optimizer_closure():
                # emulate bayesian optimization.
                num_backward = 2
                losses = []
                for backward_idx in range(num_backward):
                    loss = compute_loss()
                    losses.append(loss)
                    retain_graph = (num_backward - 1) != backward_idx
                    self.manual_backward(loss, retain_graph=retain_graph)
                # emulate MC dropout training
                loss = torch.stack(losses).mean()
                self._losses.append(loss)
                self.log("train_loss", loss, on_step=True, prog_bar=True, on_epoch=True)
                assert losses[0] != losses[1]

            weight_before = self.layer.weight.clone()

            opt.step(closure=optimizer_closure)
            opt.zero_grad()

            weight_after = self.layer.weight.clone()
            assert not torch.equal(weight_before, weight_after)

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
    )

    with mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward) as bwd_mock:
        trainer.fit(model)
    assert bwd_mock.call_count == limit_train_batches * 2
    assert trainer.progress_bar_metrics["train_loss_step"] == model._losses[-1]
    assert trainer.progress_bar_metrics["train_loss_epoch"] == torch.stack(model._losses).mean()


def test_step_with_optimizer_closure_and_accumulated_grad(tmpdir):
    """Tests that `step` works with optimizer_closure and accumulated_grad."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            # manual
            opt = self.optimizers()
            x = batch[0]

            loss_1 = self(x)
            loss_1 = self.loss(loss_1, loss_1)

            def optimizer_closure():
                # emulate bayesian optimization.
                num_backward = 1
                for backward_idx in range(num_backward + 1):
                    retain_graph = num_backward != backward_idx
                    self.manual_backward(loss_1, retain_graph=retain_graph)

            weight_before = self.layer.weight.clone()

            opt.step(closure=optimizer_closure)

            weight_after = self.layer.weight.clone()
            if not self.trainer.fit_loop._should_accumulate():
                assert not torch.equal(weight_before, weight_after)
            else:
                assert self.layer.weight.grad is not None

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 4
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
    )

    with mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward) as bwd_mock:
        trainer.fit(model)
    assert bwd_mock.call_count == limit_train_batches * 2


@patch("torch.optim.SGD.step")
def test_step_with_optimizer_closure_and_extra_arguments(step_mock, tmpdir):
    """Tests that `step` works with optimizer_closure and extra arguments."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def on_train_start(self) -> None:
            step_mock.reset_mock()

        def training_step(self, batch, batch_idx):
            # manual
            opt = self.optimizers()
            x = batch[0]

            loss_1 = self(x)
            loss_1 = self.loss(loss_1, loss_1)

            def optimizer_closure():
                # emulate bayesian optimization.
                num_backward = 1
                for backward_idx in range(num_backward + 1):
                    retain_graph = num_backward != backward_idx
                    self.manual_backward(loss_1, retain_graph=retain_graph)

            opt.step(closure=optimizer_closure)
            opt.zero_grad()

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 4
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
    )

    trainer.fit(model)
    assert step_mock.mock_calls == [call(closure=ANY) for _ in range(limit_train_batches)]


@patch("torch.optim.Adam.step")
@patch("torch.optim.SGD.step")
def test_step_with_optimizer_closure_with_different_frequencies(mock_sgd_step, mock_adam_step, tmpdir):
    """Tests that `step` works with optimizer_closure and different accumulated_gradient frequency."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def on_train_start(self) -> None:
            mock_sgd_step.reset_mock()
            mock_adam_step.reset_mock()

        def training_step(self, batch, batch_idx):

            # emulate gans training
            opt_gen, opt_dis = self.optimizers()

            # Note: Be careful, don't log on the same key in self.log in both closure
            # as they will be aggregated together on epoch_end

            def compute_loss():
                x = batch[0]
                x = F.dropout(x, 0.1)
                predictions = self(x)
                predictions = F.dropout(predictions, 0.1)
                loss = self.loss(None, predictions)
                return loss

            def gen_closure():
                loss_gen = compute_loss()
                self.log("loss_gen", loss_gen, on_step=True, on_epoch=True)
                self.manual_backward(loss_gen)

            def dis_closure():
                loss_dis = compute_loss()
                self.log("loss_dis", loss_dis, on_step=True, on_epoch=True)
                self.manual_backward(loss_dis)

            # this will accumulate gradients for 2 batches and then call opt_gen.step()
            gen_closure()
            if batch_idx % 2 == 0:
                opt_gen.step(closure=gen_closure, optim="sgd")
                opt_gen.zero_grad()

            # update discriminator every 4 baches
            # therefore, no gradient accumulation for discriminator
            if batch_idx % 4 == 0:
                opt_dis.step(closure=dis_closure)
                opt_dis.zero_grad()

        def configure_optimizers(self):
            optimizer_gen = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_dis = torch.optim.Adam(self.layer.parameters(), lr=0.001)
            return [optimizer_gen, optimizer_dis]

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 8
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
    )

    trainer.fit(model)
    assert mock_sgd_step.mock_calls == [call(closure=ANY, optim="sgd") for _ in range(4)]
    assert mock_adam_step.mock_calls == [call(closure=ANY) for _ in range(2)]


class TesManualOptimizationDDPModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def loss_ones(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def loss_zeros(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.zeros_like(prediction))

    def manual_sync_grad(self) -> bool:
        torch_distrib.all_reduce(self.layer.weight.grad.data, async_op=False)
        return True

    def training_step(self, batch, batch_idx):

        # emulate gans training
        opt_gen, opt_dis = self.optimizers()

        # Note: Be careful, don't log on the same key in self.log in both closure
        # as they will be aggregated together on epoch_end

        world_size = torch_distrib.get_world_size(torch_distrib.group.WORLD)
        assert world_size == 2

        make_gen_optimizer_step = batch_idx % 2 == 1
        make_dis_optimizer_step = batch_idx % 4 == 0

        def compute_loss():
            x = batch[0]
            x = F.dropout(x, 0.1)
            predictions = self(x)
            predictions = F.dropout(predictions, 0.1)
            loss_ones = self.loss_ones(None, predictions)
            loss_zeros = self.loss_zeros(None, predictions)
            return loss_ones, loss_zeros

        def make_manual_backward(loss, retain_graph=False, make_optimizer_step=True):
            self.manual_backward(loss, retain_graph=retain_graph)
            if make_optimizer_step:
                grad_clone = self.layer.weight.grad.clone()
                assert self.manual_sync_grad()
                self.layer.weight.grad /= world_size
                assert torch.equal(self.layer.weight.grad, grad_clone)

        def gen_closure():
            loss_ones_gen, loss_zeros = compute_loss()
            make_manual_backward(loss_ones_gen, retain_graph=True, make_optimizer_step=make_gen_optimizer_step)
            make_manual_backward(loss_ones_gen, make_optimizer_step=make_gen_optimizer_step)

        def dis_closure():
            loss_ones_gen, loss_zeros = compute_loss()
            make_manual_backward(loss_ones_gen, retain_graph=True, make_optimizer_step=make_dis_optimizer_step)
            make_manual_backward(loss_ones_gen, make_optimizer_step=make_dis_optimizer_step)

        # this will accumulate gradients for 2 batches and then call opt_gen.step()
        if make_gen_optimizer_step:
            opt_gen.step(closure=gen_closure)
            opt_gen.zero_grad()

        # update discriminator every 4 baches
        # therefore, no gradient accumulation for discriminator
        if make_dis_optimizer_step:
            opt_dis.step(closure=dis_closure)

    def configure_optimizers(self):
        optimizer_gen = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        optimizer_dis = torch.optim.Adam(self.layer.parameters(), lr=0.001)
        return [optimizer_gen, optimizer_dis]

    def on_train_start(self):
        # this is done here instead of in the calling function due to `spawn`
        sgd, adam = self.optimizers()
        self.sgd_step_patch = patch.object(sgd, "step", wraps=sgd.step)
        self.sgd_step_mock = self.sgd_step_patch.start()
        self.adam_step_patch = patch.object(adam, "step", wraps=adam.step)
        self.adam_step_mock = self.adam_step_patch.start()

    def on_train_end(self):
        self.sgd_step_patch.stop()
        assert self.sgd_step_mock.call_count == 4
        self.adam_step_patch.stop()
        assert self.adam_step_mock.call_count == 2


def train_manual_optimization(tmpdir, strategy, model_cls=TesManualOptimizationDDPModel):

    seed_everything(42)

    model = model_cls()
    model_copy = deepcopy(model)
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 8
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        gpus=2,
        strategy=strategy,
    )

    trainer.fit(model)

    for param, param_copy in zip(model.parameters(), model_copy.parameters()):
        assert not torch.equal(param.cpu().data, param_copy.data)


@RunIf(min_gpus=2, standalone=True)
def test_step_with_optimizer_closure_with_different_frequencies_ddp(tmpdir):
    """Tests that `step` works with optimizer_closure and different accumulated_gradient frequency."""

    train_manual_optimization(tmpdir, "ddp")


@RunIf(min_gpus=2)
def test_step_with_optimizer_closure_with_different_frequencies_ddp_spawn(tmpdir):
    """Tests that `step` works with optimizer_closure and different accumulated_gradient frequency."""

    train_manual_optimization(tmpdir, "ddp_spawn")


class TestManualOptimizationDDPModelToggleModel(TesManualOptimizationDDPModel):
    def training_step(self, batch, batch_idx):

        # emulate gans training
        opt_gen, opt_dis = self.optimizers()

        # Note: Be careful, don't log on the same key in self.log in both closure
        # as they will be aggregated together on epoch_end

        world_size = torch_distrib.get_world_size(torch_distrib.group.WORLD)
        assert world_size == 2

        make_gen_optimizer_step = batch_idx % 2 == 1
        make_dis_optimizer_step = batch_idx % 4 == 0

        def compute_loss():
            x = batch[0]
            x = F.dropout(x, 0.1)
            predictions = self(x)
            predictions = F.dropout(predictions, 0.1)
            loss_ones = self.loss_ones(None, predictions)
            loss_zeros = self.loss_zeros(None, predictions)
            return loss_ones, loss_zeros

        def make_manual_backward(loss, retain_graph=False, make_optimizer_step=True):
            self.manual_backward(loss, retain_graph=retain_graph)
            if make_optimizer_step:
                grad_clone = self.layer.weight.grad.clone()
                assert self.manual_sync_grad()
                self.layer.weight.grad /= world_size
                assert torch.equal(self.layer.weight.grad, grad_clone)

        def gen_closure():
            loss_ones_gen, loss_zeros = compute_loss()
            make_manual_backward(loss_ones_gen, retain_graph=True, make_optimizer_step=make_gen_optimizer_step)
            make_manual_backward(loss_ones_gen, make_optimizer_step=make_gen_optimizer_step)

        def dis_closure():
            loss_ones_gen, loss_zeros = compute_loss()
            make_manual_backward(loss_ones_gen, retain_graph=True, make_optimizer_step=make_dis_optimizer_step)
            make_manual_backward(loss_ones_gen, make_optimizer_step=make_dis_optimizer_step)

        # this will accumulate gradients for 2 batches and then call opt_gen.step()
        with opt_gen.toggle_model(sync_grad=make_gen_optimizer_step):
            gen_closure()
            if make_gen_optimizer_step:
                opt_gen.step()
                opt_gen.zero_grad()

        with opt_dis.toggle_model(sync_grad=make_dis_optimizer_step):
            dis_closure()
            if make_dis_optimizer_step:
                opt_dis.step()
                opt_dis.zero_grad()


@RunIf(min_gpus=2, standalone=True)
def test_step_with_optimizer_closure_with_different_frequencies_ddp_with_toggle_model(tmpdir):
    train_manual_optimization(tmpdir, "ddp", model_cls=TestManualOptimizationDDPModelToggleModel)


def test_lr_schedulers(tmpdir):
    """Test `lr_schedulers()` returns the same objects in the same order as `configure_optimizers()` returns."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            scheduler_1, scheduler_2 = self.lr_schedulers()
            assert scheduler_1 is self.scheduler_1
            assert scheduler_2 is self.scheduler_2

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.parameters(), lr=0.1)
            self.scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            self.scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=1)
            return [optimizer_1, optimizer_2], [self.scheduler_1, self.scheduler_2]

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=1, limit_val_batches=1, limit_test_batches=1
    )

    trainer.fit(model)


@pytest.mark.parametrize("scheduler_as_dict", [True, False])
def test_lr_schedulers_reduce_lr_on_plateau(tmpdir, scheduler_as_dict):
    class TestModel(BoringModel):
        def __init__(self, scheduler_as_dict):
            super().__init__()
            self.scheduler_as_dict = scheduler_as_dict
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            return {"train_loss": torch.tensor([0.0])}

        def training_epoch_end(self, outputs):
            scheduler = self.lr_schedulers()

            loss = torch.stack([x["train_loss"] for x in outputs]).mean()
            scheduler.step(loss)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

            if self.scheduler_as_dict:
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                    "monitor": "train_loss",
                }
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

            return [optimizer], [scheduler]

    model = TestModel(scheduler_as_dict=scheduler_as_dict)

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=1, limit_val_batches=1, limit_test_batches=1
    )

    if scheduler_as_dict:
        with pytest.warns(RuntimeWarning, match="but the keys will be ignored"):
            trainer.fit(model)
    else:
        trainer.fit(model)


def test_lr_scheduler_step_not_called(tmpdir):
    """Test `lr_scheduler.step()` is not called in manual optimization."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            opt = self.optimizers()

            output = self(batch)
            loss = self.loss(batch, output)

            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

    model = TestModel()
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir, fast_dev_run=2)

    with patch("torch.optim.lr_scheduler.StepLR.step") as lr_step:
        trainer.fit(model)

    # If a lr scheduler inherits `torch.optim.lr_scheduler._LRScheduler`,
    # `.step()` is called once during its instantiation.
    # Thus, the call count should be 1, not 0.
    assert lr_step.call_count == 1


@RunIf(min_gpus=1)
@pytest.mark.parametrize("precision", [16, 32])
def test_multiple_optimizers_logging(precision, tmpdir):
    """Tests that metrics are properly being logged."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            # Discriminator.
            optimizer_idx = 0
            optimizer = self.optimizers()[optimizer_idx]
            self.toggle_optimizer(optimizer, optimizer_idx)

            loss_d = self.loss(batch, self.layer(batch))
            self.log("loss_d", loss_d, prog_bar=True)

            optimizer.zero_grad()
            self.manual_backward(loss_d)
            optimizer.step()
            self.untoggle_optimizer(optimizer_idx)

            # Generator.
            optimizer_idx = 1
            optimizer = self.optimizers()[optimizer_idx]
            self.toggle_optimizer(optimizer, optimizer_idx)

            loss_g = self.loss(batch, self.layer(batch))
            self.log("loss_g", loss_g, prog_bar=True)

            optimizer.zero_grad()
            self.manual_backward(loss_g)
            optimizer.step()
            self.untoggle_optimizer(optimizer_idx)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.training_epoch_end = None
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        gpus=1,
        precision=precision,
    )

    trainer.fit(model)

    assert set(trainer.logged_metrics) == {"loss_d", "loss_g"}
    assert set(trainer.progress_bar_metrics) == {"loss_d", "loss_g"}


def test_manual_optimization_training_step_signature(tmpdir):
    """Test that Lightning raises an exception if the training_step signature has an optimier_idx by mistake."""

    class ConfusedAutomaticManualModel(ManualOptModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

    model = ConfusedAutomaticManualModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2)

    with pytest.raises(ValueError, match="Your `LightningModule.training_step` signature contains an `optimizer_idx`"):
        trainer.fit(model)
