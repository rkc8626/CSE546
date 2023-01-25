import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        super().__init__()
        raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters = list(network_init(784, 64, 10, device=device))
        data = X
        for i in range(0, len(parameters), 2):
            W = parameters[i]
            b = parameters[i + 1]
            data = data @ W.t() + b
            if i != len(parameters) - 2:
                data = relu(data)
        return data


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        super().__init__()
        raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters = list(network_init(784, 32, 32, 10, device=device))
        data = X
        for i in range(0, len(parameters), 2):
            W = parameters[i]
            b = parameters[i + 1]
            data = data @ W.t() + b
            if i != len(parameters) - 2:
                data = relu(data)
        return data


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:


@problem.tag("hw3-A", start_line=5)
def main():
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        acc = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data[0].to("cuda"), data[1].to("cuda")
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    with torch.no_grad():
    Y_pred = network_forward(X_test, parameters)
    print("Test loss:", cross_entropy(Y_pred, Y_test).item())
    print("Test accuracy", accuracy(Y_pred, Y_test).item())

    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
