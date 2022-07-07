import numpy as np
import random
import torch
import os
from datetime import datetime

from dlam_project.evaluation import eval_model



def test(change_model_fc, save_to, model=None, eval_fc=eval_model, device=None, raise_exc=True):
    try:
        os.makedirs(save_to, exist_ok=False)
    except FileExistsError as exc:
        if raise_exc:
            raise exc
        else:
            save_to = f"{save_to}_{datetime.now().strftime('%m-%d_%H-%M-%S')}"
            test(change_model_fc, save_to, model, eval_fc, device, raise_exc)
    
    if model is None:
        model = torch.load("./saves/base/model.pt")
    
    model = change_model_fc(model)

    eval_fc(model, save_to, device=device)


def setup():
    random.seed(1337)
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    