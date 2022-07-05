from torch import device, empty, no_grad, cuda
from torch.utils.data import DataLoader
import torch.nn as nn

from dlam_project.datasets.cifar import cifar10_test
from evaluator import Evaluator


_device = device("cuda" if cuda.is_available() else "cpu")


def eval_model(model, dataset=cifar10_test, batch_size=64, torch_device=_device, eval_obj: Evaluator = None):
    """
    returns the edited model
    """
    model = model.to(torch_device)

    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    losses = empty(len(dataset))
    correct = empty(len(dataset))

    model.eval()
    with no_grad():
        for batch, (X, Y) in enumerate(testloader):
            x = X.to(torch_device)
            y = Y.to(torch_device)

            pred = model(x)

            start = batch * batch_size
            end = (batch + 1) * batch_size

            losses[start:end] = loss_fn(pred, y).cpu()
            correct[start:end] = (pred.argmax(dim=1) == y).cpu()

        acc = correct.mean().item()

    eval_obj(losses, correct, model)

    print(f"Accuracy: {acc:.3f}")

    return model
