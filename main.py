import random
import torch
import numpy as np
import os

from experiments import layerwise



if __name__ == "__main__":
    random.seed(1337)
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    layerwise.run()