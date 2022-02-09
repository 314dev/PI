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
from typing import Dict, Optional

import pytest

from pi_ml import Trainer
from pi_ml.callbacks import DeviceStatsMonitor
from pi_ml.callbacks.device_stats_monitor import _prefix_metric_keys
from pi_ml.loggers import CSVLogger
from pi_ml.utilities.exceptions import MisconfigurationException
from pi_ml.utilities.rank_zero import rank_zero_only
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@RunIf(min_torch="1.8")
@RunIf(min_gpus=1)
def test_device_stats_gpu_from_torch(tmpdir):
    """Test GPU stats are logged using a logger with Pytorch >= 1.8.0."""
    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            fields = ["allocated_bytes.all.freed", "inactive_split.all.peak", "reserved_bytes.large_pool.peak"]
            for f in fields:
                assert any(f in h for h in metrics.keys())

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=7,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


@RunIf(max_torch="1.7")
@RunIf(min_gpus=1)
def test_device_stats_gpu_from_nvidia(tmpdir):
    """Test GPU stats are logged using a logger with Pytorch < 1.8.0."""
    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            fields = ["utilization.gpu", "memory.used", "memory.free", "utilization.memory"]
            for f in fields:
                assert any(f in h for h in metrics.keys())

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=7,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


@RunIf(tpu=True)
def test_device_stats_monitor_tpu(tmpdir):
    """Test TPU stats are logged using a logger."""

    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            fields = ["avg. free memory (MB)", "avg. peak memory (MB)"]
            for f in fields:
                assert any(f in h for h in metrics.keys())

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        accelerator="tpu",
        devices=8,
        log_every_n_steps=1,
        callbacks=[device_stats],
        logger=DebugLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)


def test_device_stats_monitor_no_logger(tmpdir):
    """Test DeviceStatsMonitor with no logger in Trainer."""

    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[device_stats],
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    with pytest.raises(MisconfigurationException, match="Trainer that has no logger."):
        trainer.fit(model)


def test_prefix_metric_keys(tmpdir):
    """Test that metric key names are converted correctly."""
    metrics = {"1": 1.0, "2": 2.0, "3": 3.0}
    prefix = "foo"
    separator = "."
    converted_metrics = _prefix_metric_keys(metrics, prefix, separator)
    assert converted_metrics == {"foo.1": 1.0, "foo.2": 2.0, "foo.3": 3.0}
