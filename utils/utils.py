import numpy as np
import random
import torch
import os

from dlam_project.evaluation import eval_model



def test(change_model_fc, *args, save_to=None, model=None, eval_fc=eval_model, device=None, **kwargs):
    if save_to is None:
        raise ValueError("Missing required argument 'save_to'")

    os.makedirs(save_to, exist_ok=True)
    
    if model is None:
        model = torch.load("./saves/base/model.pt")
    
    model = change_model_fc(model=model, *args, **kwargs)

    eval_fc(model, save_to, device=device)


def setup():
    random.seed(1337)
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    