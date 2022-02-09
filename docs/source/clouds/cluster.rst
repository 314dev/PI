.. testsetup:: *

    from pi_ml.trainer.trainer import Trainer

*****************
Computing cluster
*****************

With Lightning it is easy to run your training script on a computing cluster without almost any modifications to the script.
In this guide, we cover

1.  General purpose cluster (not managed)

2.  Using `Torch Distributed Run <https://pytorch.org/docs/stable/elastic/run.html>`__

3.  SLURM cluster

4.  Custom cluster environment

5.  General tips for multi-node training

--------

.. _non-slurm:

1. General purpose cluster
==========================

This guide shows how to run a training job on a general purpose cluster. We recommend beginners to try this method
first because it requires the least amount of configuration and changes to the code.
To setup a multi-node computing cluster you need:

1) Multiple computers with PyTorch Lightning installed
2) A network connectivity between them with firewall rules that allow traffic flow on a specified *MASTER_PORT*.
3) Defined environment variables on each node required for the PyTorch Lightning multi-node distributed training

PyTorch Lightning follows the design of `PyTorch distributed communication package <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`_. and requires the following environment variables to be defined on each node:

- *MASTER_PORT* - required; has to be a free port on machine with NODE_RANK 0
- *MASTER_ADDR* - required (except for NODE_RANK 0); address of NODE_RANK 0 node
- *WORLD_SIZE* - required; how many nodes are in the cluster
- *NODE_RANK* - required; id of the node in the cluster

.. _training_script_setup:

Training script setup
---------------------

To train a model using multiple nodes, do the following:

1.  Design your :ref:`lightning_module` (no need to add anything specific here).

2.  Enable DDP in the trainer

    .. code-block:: python

       # train on 32 GPUs across 4 nodes
       trainer = Trainer(gpus=8, num_nodes=4, strategy="ddp")


Submit a job to the cluster
---------------------------

To submit a training job to the cluster you need to run the same training script on each node of the cluster.
This means that you need to:

1. Copy all third-party libraries to each node (usually means - distribute requirements.txt file and install it).
2. Copy all your import dependencies and the script itself to each node.
3. Run the script on each node.


----------

.. _torch_distributed_run:

2. Torch Distributed Run
========================

`Torch Distributed Run <https://pytorch.org/docs/stable/elastic/run.html>`__ provides helper functions to setup distributed environment variables from the `PyTorch distributed communication package <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`__ that need to be defined on each node.

Once the script is setup like described in :ref:`training_script_setup`, you can run the below command across your nodes to start multi-node training.

Like a custom cluster, you have to ensure that there is network connectivity between the nodes with firewall rules that allow traffic flow on a specified *MASTER_PORT*.

Finally, you'll need to decide which node you'd like to be the main node (*MASTER_ADDR*), and the ranks of each node (*NODE_RANK*).

For example:

* *MASTER_ADDR* 10.10.10.16
* *MASTER_PORT* 29500
* *NODE_RANK* 0 for the first node, 1 for the second node

Run the below command with the appropriate variables set on each node.

.. code-block:: bash

    python -m torch.distributed.run
        --nnodes=2 # number of nodes you'd like to run with
        --master_addr <MASTER_ADDR>
        --master_port <MASTER_PORT>
        --node_rank <NODE_RANK>
        train.py (--arg1 ... train script args...)

.. note::

    ``torch.distributed.run`` assumes that you'd like to spawn a process per GPU if GPU devices are found on the node. This can be adjusted with ``-nproc_per_node``.

.. _slurm:

3. SLURM managed cluster
========================

Lightning automates the details behind training on a SLURM-powered cluster. In contrast to the general purpose
cluster above, the user does not start the jobs manually on each node and instead submits it to SLURM which
schedules the resources and time for which the job is allowed to run.


Training script design
----------------------

To train a model using multiple nodes, do the following:

1.  Design your :ref:`lightning_module` (no need to add anything specific here).

2.  Enable DDP in the trainer

    .. code-block:: python

       # train on 32 GPUs across 4 nodes
       trainer = Trainer(gpus=8, num_nodes=4, strategy="ddp")

3.  It's a good idea to structure your training script like this:

    .. testcode::

        # train.py
        def main(hparams):
            model = LightningTemplateModel(hparams)

            trainer = Trainer(gpus=8, num_nodes=4, strategy="ddp")

            trainer.fit(model)


        if __name__ == "__main__":
            root_dir = os.path.dirname(os.path.realpath(__file__))
            parent_parser = ArgumentParser(add_help=False)
            hyperparams = parser.parse_args()

            # TRAIN
            main(hyperparams)

4.  Create the appropriate SLURM job:

    .. code-block:: bash

        # (submit.sh)
        #!/bin/bash -l

        # SLURM SUBMIT SCRIPT
        #SBATCH --nodes=4
        #SBATCH --gres=gpu:8
        #SBATCH --ntasks-per-node=8
        #SBATCH --mem=0
        #SBATCH --time=0-02:00:00

        # activate conda env
        source activate $1

        # debugging flags (optional)
        export NCCL_DEBUG=INFO
        export PYTHONFAULTHANDLER=1

        # on your cluster you might need these:
        # set the network interface
        # export NCCL_SOCKET_IFNAME=^docker0,lo

        # might need the latest CUDA
        # module load NCCL/2.4.7-1-cuda.10.0

        # run script from above
        srun python3 train.py

5.  If you want auto-resubmit (read below), add this line to the submit.sh script

    .. code-block:: bash

        #SBATCH --signal=SIGUSR1@90

6.  Submit the SLURM job

    .. code-block:: bash

        sbatch submit.sh


Wall time auto-resubmit
-----------------------
When you use Lightning in a SLURM cluster, it automatically detects when it is about
to run into the wall time and does the following:

1.  Saves a temporary checkpoint.
2.  Requeues the job.
3.  When the job starts, it loads the temporary checkpoint.

To get this behavior make sure to add the correct signal to your SLURM script

.. code-block:: bash

    # 90 seconds before training ends
    SBATCH --signal=SIGUSR1@90

If auto-resubmit is not desired, it can be turned off in the :class:`~pi_ml.plugins.environments.slurm_environment.SLURMEnvironment` plugin:

.. code-block:: python

    from pi_ml.plugins.environments import SLURMEnvironment

    trainer = Trainer(plugins=[SLURMEnvironment(auto_requeue=False)])


Building SLURM scripts
----------------------

Instead of manually building SLURM scripts, you can use the
`SlurmCluster object <https://williamfalcon.github.io/test-tube/hpc/SlurmCluster>`_
to do this for you. The SlurmCluster can also run a grid search if you pass
in a `HyperOptArgumentParser
<https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser>`_.

Here is an example where you run a grid search of 9 combinations of hyperparameters.
See also the multi-node examples
`here <https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pl_examples/basic_examples>`__.

.. code-block:: python

    # grid search 3 values of learning rate and 3 values of number of layers for your net
    # this generates 9 experiments (lr=1e-3, layers=16), (lr=1e-3, layers=32),
    # (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)
    parser.opt_list("--learning_rate", default=0.001, type=float, options=[1e-3, 1e-2, 1e-1], tunable=True)
    parser.opt_list("--layers", default=1, type=float, options=[16, 32, 64], tunable=True)
    hyperparams = parser.parse_args()

    # Slurm cluster submits 9 jobs, each with a set of hyperparams
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path="/some/path/to/save",
    )

    # OPTIONAL FLAGS WHICH MAY BE CLUSTER DEPENDENT
    # which interface your nodes use for communication
    cluster.add_command("export NCCL_SOCKET_IFNAME=^docker0,lo")

    # see the output of the NCCL connection process
    # NCCL is how the nodes talk to each other
    cluster.add_command("export NCCL_DEBUG=INFO")

    # setting a main port here is a good idea.
    cluster.add_command("export MASTER_PORT=%r" % PORT)

    # ************** DON'T FORGET THIS ***************
    # MUST load the latest NCCL version
    cluster.load_modules(["NCCL/2.4.7-1-cuda.10.0"])

    # configure cluster
    cluster.per_experiment_nb_nodes = 12
    cluster.per_experiment_nb_gpus = 8

    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=8, comment="1 task per gpu")

    # submit a script with 9 combinations of hyper params
    # (lr=1e-3, layers=16), (lr=1e-3, layers=32), (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    cluster.optimize_parallel_cluster_gpu(
        main, nb_trials=9, job_name="name_for_squeue"  # how many permutations of the grid search to run
    )


The other option is that you generate scripts on your own via a bash command or use our
:doc:`native solution <../clouds/cloud_training>`.

----------

.. _custom-cluster:

4. Custom cluster
=================

Lightning provides an interface for providing your own definition of a cluster environment. It mainly consists of
parsing the right environment variables to access information such as world size, global and local rank (process id),
and node rank (node id). Here is an example of a custom
:class:`~pi_ml.plugins.environments.cluster_environment.ClusterEnvironment`:

.. code-block:: python

    import os
    from pi_ml.plugins.environments import ClusterEnvironment


    class MyClusterEnvironment(ClusterEnvironment):
        @property
        def creates_processes_externally(self) -> bool:
            """Return True if the cluster is managed (you don't launch processes yourself)"""
            return True

        def world_size(self) -> int:
            return int(os.environ["WORLD_SIZE"])

        def global_rank(self) -> int:
            return int(os.environ["RANK"])

        def local_rank(self) -> int:
            return int(os.environ["LOCAL_RANK"])

        def node_rank(self) -> int:
            return int(os.environ["NODE_RANK"])

        def main_address(self) -> str:
            return os.environ["MASTER_ADDRESS"]

        def main_port(self) -> int:
            return int(os.environ["MASTER_PORT"])


    trainer = Trainer(plugins=[MyClusterEnvironment()])


----------

5. General tips for multi-node training
=======================================

Debugging flags
---------------

When running in DDP mode, some errors in your code can show up as an NCCL issue.
Set the ``NCCL_DEBUG=INFO`` environment variable to see the ACTUAL error.

.. code-block:: bash

    NCCL_DEBUG=INFO python train.py ...
