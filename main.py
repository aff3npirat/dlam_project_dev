import timm
import torch
import torch.nn as nn


model = timm.create_model("resnet50", pretrained=True)

for i, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        print(f"{i}: {m} -> {m.weight.shape}")

with torch.no_grad():  # otherwise we cant modify weights (we could, but its easier like this)
    layer_10 = list(model.modules())[10]  # Conv layer with 64 filter, each with shape 64x3x3
    print(layer_10.weight.shape)
    print(f"{layer_10.weight[0, 0, :, :]}\n")  # only print single 3x3 conv filter
    layer_10.weight[0, 0, 0, 0] = 1  # change some values
    layer_10.weight[0, 0, 0, 1] = -1
    print(f"{layer_10.weight[0, 0, :, :]}\n")

    layer_10.weight[0, 0, 0, 0:3] = -1  # set three values to single value
    print(f"{layer_10.weight[0, 0, :, :]}\n")
    layer_10.weight[0, 0, 0, 0:3] = torch.tensor([-1., 0., 1.])  # set three values to three different
    print(layer_10.weight[0, 0, :, :])



# TODO train resnet on cifar10
# TODO random selection
# TODO folder structure