API References
==============

.. include:: links.rst

Accelerator API
---------------

.. currentmodule:: pi_ml.accelerators

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    GPUAccelerator
    IPUAccelerator
    TPUAccelerator

Core API
--------

.. currentmodule:: pi_ml.core

.. autosummary::
    :toctree: api
    :nosignatures:

    datamodule
    decorators
    hooks
    lightning

Strategy API
------------

.. currentmodule:: pi_ml.strategies

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    BaguaStrategy
    DDP2Strategy
    DDPFullyShardedStrategy
    DDPShardedStrategy
    DDPSpawnShardedStrategy
    DDPSpawnStrategy
    DDPStrategy
    DataParallelStrategy
    DeepSpeedStrategy
    HorovodStrategy
    IPUStrategy
    ParallelStrategy
    SingleDeviceStrategy
    SingleTPUStrategy
    Strategy
    TPUSpawnStrategy

Callbacks API
-------------

.. currentmodule:: pi_ml.callbacks

.. autosummary::
    :toctree: api
    :nosignatures:

    base
    early_stopping
    gpu_stats_monitor
    gradient_accumulation_scheduler
    lr_monitor
    model_checkpoint
    progress

Loggers API
-----------

.. currentmodule:: pi_ml.loggers

.. autosummary::
    :toctree: api
    :nosignatures:

    base
    comet
    csv_logs
    mlflow
    neptune
    tensorboard
    test_tube
    wandb

Loop API
--------

Base Classes
^^^^^^^^^^^^

.. currentmodule:: pi_ml.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~dataloader.dataloader_loop.DataLoaderLoop
    ~base.Loop


Default Loop Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training
""""""""

.. currentmodule:: pi_ml.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~batch.TrainingBatchLoop
    ~epoch.TrainingEpochLoop
    FitLoop
    ~optimization.ManualOptimization
    ~optimization.OptimizerLoop


Validation and Testing
""""""""""""""""""""""

.. currentmodule:: pi_ml.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~epoch.EvaluationEpochLoop
    ~dataloader.EvaluationLoop


Prediction
""""""""""

.. currentmodule:: pi_ml.loops

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ~epoch.PredictionEpochLoop
    ~dataloader.PredictionLoop


Plugins API
-----------

Precision Plugins
^^^^^^^^^^^^^^^^^

.. currentmodule:: pi_ml.plugins.precision

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ApexMixedPrecisionPlugin
    DeepSpeedPrecisionPlugin
    DoublePrecisionPlugin
    FullyShardedNativeMixedPrecisionPlugin
    IPUPrecisionPlugin
    MixedPrecisionPlugin
    NativeMixedPrecisionPlugin
    PrecisionPlugin
    ShardedNativeMixedPrecisionPlugin
    TPUBf16PrecisionPlugin
    TPUPrecisionPlugin

Cluster Environments
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pi_ml.plugins.environments

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    ClusterEnvironment
    KubeflowEnvironment
    LightningEnvironment
    LSFEnvironment
    SLURMEnvironment
    TorchElasticEnvironment

Checkpoint IO Plugins
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pi_ml.plugins.io

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    CheckpointIO
    TorchCheckpointIO
    XLACheckpointIO

Profiler API
------------

.. currentmodule:: pi_ml.profiler

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    AbstractProfiler
    AdvancedProfiler
    BaseProfiler
    PassThroughProfiler
    PyTorchProfiler
    SimpleProfiler
    XLAProfiler


Trainer API
-----------

.. currentmodule:: pi_ml.trainer.trainer

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Trainer

LightningLite API
-----------------

.. currentmodule:: pi_ml.lite

.. autosummary::
    :toctree: api
    :nosignatures:

    LightningLite

Tuner API
---------

.. currentmodule:: pi_ml.tuner.tuning

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: classtemplate.rst

    Tuner

Utilities API
-------------

.. currentmodule:: pi_ml.utilities

.. autosummary::
    :toctree: api
    :nosignatures:

    apply_func
    argparse
    cli
    cloud_io
    deepspeed
    distributed
    finite_checks
    memory
    model_summary
    parsing
    rank_zero
    seed
    warnings
