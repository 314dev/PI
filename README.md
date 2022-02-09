# [PI](https://3.14.dev/)
PI = [Pytorch(Lightning)](https://github.com/PyTorchLightning/pytorch-lightning) + [IREE](https://github.com/google/iree/)(via [torch-mlir](https://github.com/llvm/torch-mlir))

An Enhanced fork of PyTorch-Lightning with a torch-mlir + IREE backend

![pitorch](https://user-images.githubusercontent.com/74956/151889869-32b39bd9-d1eb-4c32-a5e5-33a9891d7112.jpg)

## QUICKSTART

### Step 1: Add these imports

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pi_ml as pl
import pi-ml as pi
```

### Step 2: Define a LightningModule (nn.Module subclass)

A LightningModule defines a full *system* (ie: a GAN, autoencoder, BERT or a simple Image Classifier).

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

**Note: Training_step defines the training loop. Forward defines how the LightningModule behaves during inference/prediction.**

### Step 3: Train!

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = pl.Trainer(pi=True)
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

## Or on gpu:


```python
# 8 GPUs
# no code changes needed
trainer = Trainer(max_epochs=1, gpus=8, pi=True)

# 256 GPUs, 8 GPUs/Node
trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32, pi=True)
```

## Or on TPU:
```python
# no code changes needed
trainer = Trainer(tpu_cores=8, pi=True)
```

