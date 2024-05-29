import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic_data(num_samples, num_cells):
    x = np.linspace(0, 1, num_cells)
    t = np.linspace(0, 1, num_samples)
    X, T = np.meshgrid(x, t)
    u = np.sin(np.pi * X) * np.exp(-np.pi * T)

    return X, T, u


X, T, U = generate_synthetic_data(1000, 100)

# print(f"X shape: {X.shape}")
# print(f"T shape: {T.shape}")
# print(f"U shape: {U.shape}")
# print(f"X sample: {X[:5, :5]}")
# print(f"T sample: {T[:5, :5]}")
# print(f"U sample: {U[:5, :5]}")




class MLP(nn.Module): # MLP: Multi Layer Perceptron
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 128) # first layer
        self.fc2 = nn.Linear(128, 128) # second
        self.fc3 = nn.Linear(128, 1) # third

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
print(model)
