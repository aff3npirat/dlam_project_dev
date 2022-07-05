from copy import deepcopy
from os import makedirs
from os.path import join

from torch import save


class Evaluator:
    """
    von dieser KLasse erben für eine custom Evaluation (möglicherweise mit super.__call__(..., ...) in der sub Klasse
    aufrufen).
    """

    res_folder = "./dlam_project/saves"

    def __init__(self, path_2_save: str = ''):
        self.p = path_2_save
        makedirs(self.res_folder, exist_ok=True)

    def __call__(self, losses, correct, model):
        save(deepcopy(model).cpu(), join(self.p, "model.pt"))
        save(losses, join(self.p, "losses.pt"))
        save(correct, join(self.p, "correct.pt"))
