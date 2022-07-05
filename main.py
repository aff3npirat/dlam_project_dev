from os import makedirs, environ
from random import seed

import numpy as np
from torch import manual_seed, use_deterministic_algorithms, cuda
from torch.backends import cudnn

from train import train
from method_evaluation import eval_method_4_project
from experiments import test, binarize_layer_experiment


if cuda.is_available():
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


seed(1337)
manual_seed(1337)
np.random.seed(1337)
use_deterministic_algorithms(mode=True)
cudnn.benchmark = False


makedirs("./dlam_project/saves/base", exist_ok=False)
makedirs("./dlam_project/saves/base/best", exist_ok=False)


# train()

binarize_layer_experiment()

eval_method_4_project(change_model_func=test, name_of_run='test')
