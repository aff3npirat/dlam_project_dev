import timm
import torch.nn as nn


model = timm.create_model("resnet50", pretrained=True)

for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        print(f"{m} -> {m.weight.shape}")

# TODO train resnet on cifar10
# TODO random selection
# TODO folder structure