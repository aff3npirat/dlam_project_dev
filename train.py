from copy import deepcopy

from torch import device, cuda, save
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from dlam_project.datasets.cifar import cifar10_train


def train(torch_device=device("cuda" if cuda.is_available() else "cpu")):
    """
    Trains a resnet50 on cifar10 and stores model in
    ./dlam_project/saves/base/model.pt
    """
    print(f"Current device: {torch_device}")

    model = timm.create_model("resnet50", pretrained=True, drop_rate=0.5)
    model.reset_classifier(10)

    model.train()

    trainloader = DataLoader(cifar10_train, batch_size=64, shuffle=False)
    loss_fn = CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=0.01)

    model = model.to(torch_device)
    max_acc = 0
    for e in range(50):
        print(f"\nEpoch {e + 1}")
        for batch, (X, Y) in enumerate(trainloader):
            x = X.to(torch_device)
            y = Y.to(torch_device)
            pred = model(x)

            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (batch + 1) % 100 == 0:
                print(f"Loss: {loss.item()} [{batch + 1}/{len(trainloader)}]")

        print("\nEvaluating")
        acc = eval(model, "base")
        if acc > max_acc:
            max_acc = acc
            save(deepcopy(model).cpu(), "./dlam_project/saves/base/best/model.pt")
        model.train()

    save(model.cpu(), "./dlam_project/saves/base/model.pt")
