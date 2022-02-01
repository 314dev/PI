# PI
PI = [Pytorch(Lightning)](https://github.com/PyTorchLightning/pytorch-lightning) + [IREE](https://github.com/google/iree/)(via [torch-mlir](https://github.com/llvm/torch-mlir))

An Enhanced fork of PyTorch-Lightning with a torch-mlir + IREE backend

![pitorch](https://user-images.githubusercontent.com/74956/151889869-32b39bd9-d1eb-4c32-a5e5-33a9891d7112.jpg)

## GOALS
### PyTorch-Lightning's Advantages over unstructured PyTorch

See more here: [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning#advantages-over-unstructured-pytorch)

Models become hardware agnostic
Data scientists focus on the Models, while ML systems engineers focus on the model deployment from laptops to datacenters.
Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
Minimal running speed overhead

### Enhancements to Pytorch-Lightning
Fast moving focus on torch-mlir + IREE integration (which means other Accelerator Backends may break in the process). But eventual goal is to upstream all the work here. 

Integrated install of PyTorch, PyTorch-Lightning, Torch-mlir, IREE (CPU/GPU) and CompilerGYM with one pip command `pip install pi`

Python based Op-Authoring

Pytorch Eager mode support with torch-mlir+IREE as the backend

Support for Flashes, Bolts, [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and HF.co models 

