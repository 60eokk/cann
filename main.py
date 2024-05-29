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