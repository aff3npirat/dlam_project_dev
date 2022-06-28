from torchvision import datasets, transforms



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 32 x 32 x 3 pixels
cifar10 = datasets.CIFAR10
train_set = cifar10("./dlam_project/datasets/data/cifar10", download=True, train=True, transform=transform)
test_set = cifar10("./dlam_project/datasets/data/cifar10", download=True, train=False, transform=transform)