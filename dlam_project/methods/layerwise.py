import torch
from dlam_project.helpers import get_ith_trainable



@torch.no_grad()
def binarize_layer(model, i: int):
    idx = get_ith_trainable(model, i)
    layer = list(model.modules())[idx]

    bweight = layer.weight.sign()
    bweight[bweight == 0] = 1.0

    layer.weight[:] = bweight

    if layer.bias is not None:
        bbias = layer.bias.sign()
        bbias[bbias == 0] = 1.0

        layer.bias[:] = bbias

    return model
