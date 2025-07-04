import numpy as np
from torchvision import datasets, transforms
from problem_instance import ProblemInstance

def load_cifar10_binary(class0=3, class1=5, train_ratio=0.7, seed=123):
    transform = transforms.ToTensor()

    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    X_all = np.concatenate([np.array(train_data.data), np.array(test_data.data)], axis=0)
    y_all = np.concatenate([train_data.targets, test_data.targets], axis=0)

    mask = (y_all == class0) | (y_all == class1)
    X = X_all[mask]
    y = (np.array(y_all)[mask] == class1).astype(int)

    # Flatten and normalize images
    X = X.transpose(0, 3, 1, 2).reshape(len(X), -1).astype(np.float32)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    X = X.T  # Shape (d, n)

    return ProblemInstance(X, y, mode="train_test", train_ratio=train_ratio, seed=seed)
