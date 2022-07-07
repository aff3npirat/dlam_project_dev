import torch.nn as nn
import math



def count_params(model, layers=(nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
    """
    Counts number of trainable layers and parameters.
    """
    num_params = 0
    num_layers = 0
    for m in model.modules():
        if isinstance(m, layers):
            num_layers += 1
            num_params += math.prod(m.weight.shape)
            if m.bias is not None:
                num_params += math.prod(m.bias.shape)
    return (num_layers, num_params)


def get_ith_trainable(model, i, layers=(nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
    """
    Returns index of ith trainable layer.
    """
    for idx, m in enumerate(model.modules()):
        if isinstance(m, layers):
            i -= 1
        if i == 0:
            return idx