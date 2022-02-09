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

import numpy as np
import pytest

from pi_ml import Trainer
from pi_ml.callbacks import XLAStatsMonitor
from pi_ml.loggers import CSVLogger
from pi_ml.loggers.csv_logs import ExperimentWriter
from pi_ml.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@RunIf(tpu=True)
def test_xla_stats_monitor(tmpdir):
    """Test XLA stats are logged using a logger."""

    model = BoringModel()
    xla_stats = XLAStatsMonitor()
    logger = CSVLogger(tmpdir)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=5,
        accelerator="tpu",
        devices=8,
        callbacks=[xla_stats],
        logger=logger,
    )

    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    path_csv = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    met_data = np.genfromtxt(path_csv, delimiter=",", names=True, deletechars="", replace_space=" ")

    fields = ["avg. free memory (MB)", "avg. peak memory (MB)"]

    for f in fields:
        assert any(f in h for h in met_data.dtype.names)


@RunIf(tpu=True)
def test_xla_stats_monitor_no_logger(tmpdir):
    """Test XLAStatsMonitor with no logger in Trainer."""

    model = BoringModel()
    xla_stats = XLAStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir, callbacks=[xla_stats], max_epochs=1, accelerator="tpu", devices=[1], logger=False
    )

    with pytest.raises(MisconfigurationException, match="Trainer that has no logger."):
        trainer.fit(model)
