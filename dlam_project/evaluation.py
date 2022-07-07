import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from .datasets.cifar import test_set



def eval_model(model, save_to, dataset=test_set, batch_size=64, device=None):
    """
    returns the edited model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    losses = torch.empty(len(dataset))
    correct = torch.empty(len(dataset))

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
            correct[start:end] = (pred.argmax(dim=1) == y).cpu()

        acc = correct.mean().item()

    print(f"Accuracy: {acc:.3f}")

    torch.save(losses, os.path.join(save_to, "losses.pt"))
    torch.save(correct, os.path.join(save_to, "correct.pt"))

    return (losses, correct)
