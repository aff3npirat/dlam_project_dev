import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from .datasets.cifar import test_set



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    
    copied from https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
    """
    maxk = max(topk)
    # batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].sum(0)
        # res.append(correct_k.mul_(100.0 / batch_size))
        res.append(torch.mean(correct_k.float()))
    return res


def eval_model(model, save_to, dataset=test_set, batch_size=64, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    losses = torch.empty(len(dataset))
    outputs = torch.empty(len(dataset), len(dataset.classes))

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch, (X, Y) in enumerate(testloader):
            x = X.to(device)
            y = Y.to(device)

            pred = model(x)

            start = batch * batch_size
            end = (batch+1) * batch_size

            losses[start:end] = loss_fn(pred, y).cpu()
            outputs[start:end] = pred.cpu()

        acc = accuracy(outputs, torch.tensor(dataset.targets), topk=(1, 5))

    print(f"Top-1-err: {1 - acc[0]:.3f} | Top-5-err: {1- acc[1]:.3f}")

    torch.save(acc, os.path.join(save_to, "accuracy.pt"))
    torch.save(losses, os.path.join(save_to, "losses.pt"))
    torch.save(outputs, os.path.join(save_to, "outputs.pt"))

    return losses, acc
