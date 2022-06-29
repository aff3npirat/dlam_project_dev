import torch
import os
from dlam_project.helpers import count_params, get_ith_trainable
from main import eval



if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

num_layers, _ = count_params(torch.load("./dlam_project/saves/base/model.pt"))

with torch.no_grad():
    # binarize single layer
    for i in range(2, num_layers):
        model = torch.load("./dlam_project/saves/base/model.pt")

        idx = get_ith_trainable(model, i)
        layer = list(model.modules())[idx]
        
        bweight = layer.weight.sign()
        bweight[bweight==0] = 1.0

        layer.weight[:] = bweight

        if layer.bias is not None:
            bbias = layer.bias.sign()
            bbias[bbias==0] = 1.0
            
            layer.bias[:] = bbias

        eval(
            model=model,
            name=f"layerwise/single/layer_{i}",
        )


    # binarize all layers forward direction
    model = torch.load("./dlam_project/saves/base/model.pt")
    for i in range(2, num_layers):

        idx = get_ith_trainable(model, i)
        layer = list(model.modules())[idx]
        
        bweight = layer.weight.sign()
        bweight[bweight==0] = 1.0

        layer.weight[:] = bweight

        if layer.bias is not None:
            bbias = layer.bias.sign()
            bbias[bbias==0] = 1.0
            
            layer.bias[:] = bbias

        eval(
            model=model,
            name=f"layerwise/cumul/layer_{i}",
        )


    # binarize all layers backward direction
    model = torch.load("./dlam_project/saves/base/model.pt")
    for i in reversed(range(2, num_layers)):

        idx = get_ith_trainable(model, i)
        layer = list(model.modules())[idx]
        
        bweight = layer.weight.sign()
        bweight[bweight==0] = 1.0

        layer.weight[:] = bweight

        if layer.bias is not None:
            bbias = layer.bias.sign()
            bbias[bbias==0] = 1.0
            
            layer.bias[:] = bbias

        eval(
            model=model,
            name=f"layerwise/cumul_back/layer_{i}",
        )
