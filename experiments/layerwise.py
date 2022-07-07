import torch

from dlam_project.helpers import count_params
from dlam_project.methods.layerwise import binarize_layer
from utils.utils import test



def run():
    num_layers, _ = count_params(torch.load("./saves/base/model.pt"))

    with torch.no_grad():
        # binarize single layer
        for i in range(2, num_layers):
            # model = torch.load("./saves/base/model.pt")

            print(f"Evaluating layer {i}")
            test(
                lambda x: binarize_layer(model=x, i=i),
                save_to=f"./saves/layerwise/single/layer_{i}",
            )


        # binarize all layers forward direction
        model = torch.load("./saves/base/model.pt")
        for i in range(2, num_layers):

            print(f"Evaluating layer 2-{i}")
            test(
                binarize_layer,
                save_to=f"./saves/layerwise/cumul/layer_{i}",
                model=model,
                i=i,
            )

        # binarize all layers backward direction
        model = torch.load("./saves/base/model.pt")
        for i in reversed(range(2, num_layers)):

            print(f"Evaluating layer {i}-{num_layers - 1}")
            test(
                binarize_layer,
                save_to=f"./saves/layerwise/cumul_back/layer_{i}",
                model=model,
                i=i,
            )