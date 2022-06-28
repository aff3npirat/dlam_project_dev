from genericpath import exists
import torch
import torch.nn as nn
import timm
import numpy as np
import random
from torch.utils.data import DataLoader

from dlam_project.datasets.cifar import train_set as cifar10_train
from dlam_project.datasets.cifar import test_set as cifar10_test


def train():
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

    for e in range(10):
        for batch, (x, y) in enumerate(trainloader):
            pred = model(x)

            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (batch+1) % 100 == 0:
                print(f"Loss: {loss.item()} [{batch+1}/{len(trainloader)}]")

    torch.save(model, "./dlam_project/saves/base/model.pt")


def eval(
    name,
    dataset=cifar10_test,
    batch_size=64,
):
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(mode=True)

    model = torch.load(f"./dlam_project/saves/{name}/model.pt")

    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    losses = torch.empty(len(dataset))
    correct = torch.empty(len(dataset))

    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(testloader):
            start = batch * batch_size
            end = (batch+1) * batch_size
            
            pred = model(x)
            losses[start:end] = loss_fn(pred, y)
            correct[start:end] = pred.argmax(dim=1) == y

        acc = correct.mean().item()
    print(f"Accuracy: {acc:.3f}")

    torch.save(model, f"./dlam_project/saves/{name}/model.pt")
    torch.save(losses, f"./dlam_project/saves/{name}/losses.pt")
    torch.save(correct, f"./dlam_project/saves/{name}/correct.pt")



if __name__ == "__main__":
    import os
    os.mkdir("./dlam_project/saves/base")

    train()
    eval("base")
