# PI
PI = [Pytorch(Lightning)](https://github.com/PyTorchLightning/pytorch-lightning) + [IREE](https://github.com/google/iree/)(via [torch-mlir](https://github.com/llvm/torch-mlir))

An Enhanced fork of PyTorch-Lightning with a torch-mlir + IREE backend

![pitorch](https://user-images.githubusercontent.com/74956/151889869-32b39bd9-d1eb-4c32-a5e5-33a9891d7112.jpg)

## GOALS
### PyTorch-Lightning's Advantages over unstructured PyTorch

Quoted from [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning#advantages-over-unstructured-pytorch)

Models become hardware agnostic

Code is clear to read because engineering code is abstracted away

Easier to reproduce

Make fewer mistakes because lightning handles the tricky engineering

Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate

Lightning has dozens of integrations with popular machine learning tools.

Tested rigorously with every new PR. We test every combination of PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.

Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).

### Enhancements to Pytorch-Lightning
Fast moving focus on torch-mlir + IREE integration (which means other Accelerator Backends may break in the process). But eventual goal is to upstream all the work here. 

Integrated install of PyTorch, PyTorch-Lightning, Torch-mlir, IREE (CPU/GPU) with one pip command `pip install pi`

Python based Op-Authoring

Pytorch Eager mode support with torch-mlir+IREE as the backend

Support for Flashes, Bolts, [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and HF.co models 

