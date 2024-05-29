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
# print(model)
# terminal gives 2 input layer, 128 neurons, and 1 feature of output




num_samples = 1000
num_cells = 100
X, T, U = generate_synthetic_data(num_samples, num_cells)
X_train, X_test, T_train, T_test, U_train, U_test = train_test_split(X.flatten(), T.flatten(), U.flatten(), test_size=0.2)

train_inputs = np.vstack([X_train, T_train]).T
test_inputs = np.vstack([X_test, T_test]).T

train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
train_targets = torch.tensor(U_train, dtype=torch.float32).reshape(-1, 1)
test_targets = torch.tensor(U_test, dtype=torch.float32).reshape(-1, 1)

# print(f"train_inputs shape: {train_inputs.shape}")
# print(f"train_targets shape: {train_targets.shape}")
# # 80000 samples. 
# each sample has 2 feature (spatial coordinate X, temporal cordinate T)
# each sample has 1 target value



train_mean = train_inputs.mean(dim=0, keepdim=True)
train_std = train_inputs.std(dim=0, keepdim=True)
train_inputs = (train_inputs - train_mean) / train_std
test_inputs = (test_inputs - train_mean) / train_std

print(f"train_inputs normalized sample: \n{train_inputs[:5]}")

