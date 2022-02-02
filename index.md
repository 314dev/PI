# PI
PI = [Pytorch(Lightning)](https://github.com/PyTorchLightning/pytorch-lightning) + [IREE](https://github.com/google/iree/)(via [torch-mlir](https://github.com/llvm/torch-mlir))

An Enhanced fork of PyTorch-Lightning with a torch-mlir + IREE backend

![pitorch](https://user-images.githubusercontent.com/74956/151889869-32b39bd9-d1eb-4c32-a5e5-33a9891d7112.jpg)

## GOALS

### PI's Enhancements over Pytorch-Lightning

Easy and integrated install of PyTorch, PyTorch-Lightning, Torch-mlir, IREE (CPU/GPU) and CompilerGYM with a single pip command `pip install pi`

Fast moving focus on torch-mlir + IREE integration (eventual goal is to upstream all the work here)

Python based Op-Authoring with MLIR-Linalg DSL exposed into PyTorch/Python

Pytorch Eager mode support with torch-mlir+IREE as the backend

Support for  [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models), [HF.co models](https://huggingface.co/models), [TorchBench](https://github.com/pytorch/benchmark), and potentially [Flashes](https://github.com/PyTorchLightning/lightning-bolts) and [Bolts](https://github.com/PyTorchLightning/lightning-flash). 


### PI's Advantages over unstructured PyTorch (inherited from PyTorch-Lightning)

See more here: [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning#advantages-over-unstructured-pytorch)

Make Models hardware agnostic

Data scientists focus on the Models, while ML systems engineers focus on the model deployment from laptops to datacenters.

Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate

