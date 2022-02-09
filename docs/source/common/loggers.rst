.. testsetup:: *

    from pi_ml.trainer.trainer import Trainer
    from pi_ml.core.lightning import LightningModule

.. _loggers:

*******
Loggers
*******

Lightning supports the most popular logging frameworks (TensorBoard, Comet, Neptune, etc...). TensorBoard is used by default,
but you can pass to the :class:`~pi_ml.trainer.trainer.Trainer` any combination of the following loggers.

.. note::

    All loggers log by default to `os.getcwd()`. To change the path without creating a logger set
    `Trainer(default_root_dir='/your/path/to/save/checkpoints')`

Read more about :doc:`logging <../extensions/logging>` options.

To log arbitrary artifacts like images or audio samples use the `trainer.log_dir` property to resolve
the path.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        img = ...
        log_image(img, self.trainer.log_dir)

Comet.ml
========

`Comet.ml <https://www.comet.ml/site/>`_ is a third-party logger.
To use :class:`~pi_ml.loggers.CometLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install comet-ml

Then configure the logger and pass it to the :class:`~pi_ml.trainer.trainer.Trainer`:

.. testcode::

    import os
    from pi_ml.loggers import CometLogger

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
        save_dir=".",  # Optional
        project_name="default_project",  # Optional
        rest_api_key=os.environ.get("COMET_REST_API_KEY"),  # Optional
        experiment_name="default",  # Optional
    )
    trainer = Trainer(logger=comet_logger)

The :class:`~pi_ml.loggers.CometLogger` is available anywhere except ``__init__`` in your
:class:`~pi_ml.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            self.logger.experiment.add_image("generated_images", some_img, 0)

.. seealso::
    :class:`~pi_ml.loggers.CometLogger` docs.

----------------

MLflow
======

`MLflow <https://mlflow.org/>`_ is a third-party logger.
To use :class:`~pi_ml.loggers.MLFlowLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install mlflow

Then configure the logger and pass it to the :class:`~pi_ml.trainer.trainer.Trainer`:

.. code-block:: python

    from pi_ml.loggers import MLFlowLogger

    mlf_logger = MLFlowLogger(experiment_name="default", tracking_uri="file:./ml-runs")
    trainer = Trainer(logger=mlf_logger)

.. seealso::
    :class:`~pi_ml.loggers.MLFlowLogger` docs.

----------------

Neptune.ai
==========

`Neptune.ai <https://neptune.ai/>`_ is a third-party logger.
To use :class:`~pi_ml.loggers.NeptuneLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install neptune-client

or with conda:

.. code-block:: bash

    conda install -c conda-forge neptune-client

Then configure the logger and pass it to the :class:`~pi_ml.trainer.trainer.Trainer`:

.. code-block:: python

    from pi_ml.loggers import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",  # replace with your own
        project="common/pytorch-lightning-integration",  # format "<WORKSPACE/PROJECT>"
        tags=["training", "resnet"],  # optional
    )
    trainer = Trainer(logger=neptune_logger)

The :class:`~pi_ml.loggers.NeptuneLogger` is available anywhere except ``__init__`` in your
:class:`~pi_ml.core.lightning.LightningModule`.

.. code-block:: python

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            # generic recipe for logging custom metadata (neptune specific)
            metadata = ...
            self.logger.experiment["your/metadata/structure"].log(metadata)

Note that syntax: ``self.logger.experiment["your/metadata/structure"].log(metadata)``
is specific to Neptune and it extends logger capabilities.
Specifically, it allows you to log various types of metadata like scores, files,
images, interactive visuals, CSVs, etc. Refer to the
`Neptune docs <https://docs.neptune.ai/you-should-know/logging-metadata#essential-logging-methods>`_
for more detailed explanations.

You can always use regular logger methods: ``log_metrics()`` and ``log_hyperparams()`` as these are also supported.

.. seealso::
    :class:`~pi_ml.loggers.NeptuneLogger` docs.

    Logger `user guide <https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning>`_.

----------------

Tensorboard
===========

To use `TensorBoard <https://pytorch.org/docs/stable/tensorboard.html>`_ as your logger do the following.

.. testcode::

    from pi_ml.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(logger=logger)

The :class:`~pi_ml.loggers.TensorBoardLogger` is available anywhere except ``__init__`` in your
:class:`~pi_ml.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            self.logger.experiment.add_image("generated_images", some_img, 0)

.. seealso::
    :class:`~pi_ml.loggers.TensorBoardLogger` docs.

----------------

Weights and Biases
==================

`Weights and Biases <https://docs.wandb.ai/integrations/lightning/>`_ is a third-party logger.
To use :class:`~pi_ml.loggers.WandbLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install wandb

Then configure the logger and pass it to the :class:`~pi_ml.trainer.trainer.Trainer`:

.. code-block:: python

    from pi_ml.loggers import WandbLogger

    # instrument experiment with W&B
    wandb_logger = WandbLogger(project="MNIST", log_model="all")
    trainer = Trainer(logger=wandb_logger)

    # log gradients and model topology
    wandb_logger.watch(model)

The :class:`~pi_ml.loggers.WandbLogger` is available anywhere except ``__init__`` in your
:class:`~pi_ml.core.lightning.LightningModule`.

.. code-block:: python

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            # Option 1
            self.logger.experiment.log({"generated_images": [wandb.Image(some_img, caption="...")]})
            # Option 2 for specifically logging images
            self.logger.log_image(key="generated_images", images=[some_img])

.. seealso::
    - :class:`~pi_ml.loggers.WandbLogger` docs.
    - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__
    - `Demo in Google Colab <http://wandb.me/lightning>`__ with hyperparameter search and model logging

----------------

Multiple Loggers
================

Lightning supports the use of multiple loggers, just pass a list to the
:class:`~pi_ml.trainer.trainer.Trainer`.

.. code-block:: python

    from pi_ml.loggers import TensorBoardLogger, WandbLogger

    logger1 = TensorBoardLogger(save_dir="tb_logs", name="my_model")
    logger2 = WandbLogger(save_dir="tb_logs", name="my_model")
    trainer = Trainer(logger=[logger1, logger2])

The loggers are available as a list anywhere except ``__init__`` in your
:class:`~pi_ml.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            # Option 1
            self.logger.experiment[0].add_image("generated_images", some_img, 0)
            # Option 2
            self.logger[0].experiment.add_image("generated_images", some_img, 0)
