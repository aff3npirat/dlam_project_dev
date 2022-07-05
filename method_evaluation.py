from datetime import datetime

from torch import load
from timm import create_model

from dlam_project import eval_model
from dlam_project.evaluator import Evaluator


def _get_model():
    # model = create_model("resnet50", pretrained=False)
    model = load("./dlam_project/saves/base/model.pt")
    return model


def eval_method_4_project(change_model_func, name_of_run: str, model=None, evaluator: Evaluator = None):
    """
    automatically add the current time in the format MM-DD_hh-mm-ss at the end of the name of the run.
    """
    if evaluator is None:
        evaluator = Evaluator()

    if model is None:
        model = change_model_func(_get_model())
    else:
        model = change_model_func(model)

    name_of_run = f'{name_of_run}_{datetime.now().strftime("%m-%d_%H-%M-%S")}'

    return eval_model(model, eval_obj=evaluator)
