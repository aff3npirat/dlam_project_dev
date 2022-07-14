import numpy as np
import random
import torch
import os
from datetime import datetime

from dlam_project.evaluation import eval_model



def test(change_model_fc, save_to, model=None, eval_fc=eval_model, device=None, handle_duplicate="skip"):
    """
    Args:
        duplicate: one of {"skip", "raise", "rename"}.
    """
    if model is None:
        model = torch.load("./saves/base/model.pt")

    try:
        os.makedirs(save_to, exist_ok=False)
    except FileExistsError as exc:
        if handle_duplicate == "raise":
            raise exc
        elif handle_duplicate == "skip":
            return change_model_fc(model)
        elif handle_duplicate == "rename":
            save_to = f"{save_to}_{datetime.now().strftime('%m-%d_%H-%M-%S')}"
            test(change_model_fc, save_to, model, eval_fc, device, handle_duplicate)

    model = change_model_fc(model)

    eval_fc(model, save_to, device=device)
    
    return model


def setup():
    random.seed(1337)
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    