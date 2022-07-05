from torch import load, no_grad

from dlam_project.helpers import count_params
from dlam_project.methods.layerwise import binarize_layer
from method_evaluation import eval_method_4_project


def binarize_layer_experiment():
    num_layers, _ = count_params(load("./dlam_project/saves/base/model.pt"))

    with no_grad():
        # binarize single layer
        for i in range(2, num_layers):

            print(f"Evaluating layer {i}")
            eval_method_4_project(change_model_func=lambda x: binarize_layer(model=x, i=i), name_of_run=f"layerwise/single/layer_{i}")


        # binarize all layers forward direction
        model = load("./dlam_project/saves/base/model.pt")
        for i in range(2, num_layers):

            print(f"Evaluating layer 2-{i}")
            model = eval_method_4_project(change_model_func=lambda x: binarize_layer(model=x, i=i), model=model,
                                          name_of_run=f"layerwise/cumul/layer_{i}")

        # binarize all layers backward direction
        model = load("./dlam_project/saves/base/model.pt")
        for i in reversed(range(2, num_layers)):

            print(f"Evaluating layer {i}-{num_layers - 1}")
            model = eval_method_4_project(change_model_func=lambda x: binarize_layer(model=x, i=i), model=model,
                                          name_of_run=f"layerwise/cumul_back/layer_{i}")
