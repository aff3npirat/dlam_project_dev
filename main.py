import torch
import os
import torch.nn as nn
import timm
import numpy as np
import random
from torch.utils.data import DataLoader

from dlam_project.datasets.cifar import train_set as cifar10_train
from dlam_project.datasets.cifar import test_set as cifar10_test



def train(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Trains a resnet50 on cifar10 and stores model in
    ./dlam_project/saves/base/model.pt
    """
    model = timm.create_model("resnet50", pretrained=False, drop_rate=0.5)
    model.reset_classifier(10)
    model.train()

    trainloader = DataLoader(cifar10_train, batch_size=64, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    model = model.to(device)
    for e in range(10):
        print(f"Epoch {e+1}\n")
        for batch, (X, Y) in enumerate(trainloader):
            x = X.to(device)
            y = Y.to(device)
            pred = model(x)

            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (batch+1) % 100 == 0:
                print(f"Loss: {loss.item()} [{batch+1}/{len(trainloader)}]")

    torch.save(model.cpu(), "./dlam_project/saves/base/model.pt")


def eval(
    model,
    name,
    subdir="",
    dataset=cifar10_test,
    batch_size=64,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    path = f"./dlam_project/saves/{name}/{subdir}"

    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(mode=True)

    model = model.to(device)

    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    losses = torch.empty(len(dataset))
    correct = torch.empty(len(dataset))

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

    torch.save(model.cpu(), os.path.join(path, "model.pt"))
    torch.save(losses, os.path.join(path, "losses.pt"))
    torch.save(correct, os.path.join(path, "correct.pt"))



if __name__ == "__main__":
    os.makedirs("./dlam_project/saves/base", exist_ok=False)

    train()
    eval("base")
