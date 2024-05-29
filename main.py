import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
def generate_synthetic_data(num_samples, num_cells):
    x = np.linspace(0, 1, num_cells)
    t = np.linspace(0, 1, num_samples)
    X, T = np.meshgrid(x, t)
    u = np.sin(np.pi * X) * np.exp(-np.pi * T)
    return X, T, u

X, T, U = generate_synthetic_data(1000, 100)

# Define the neural network model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
print(model)

# Prepare the data
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

print(f"train_inputs shape: {train_inputs.shape}")
print(f"train_targets shape: {train_targets.shape}")

# Normalize the data
train_mean = train_inputs.mean(dim=0, keepdim=True)
train_std = train_inputs.std(dim=0, keepdim=True)
train_inputs = (train_inputs - train_mean) / train_std
test_inputs = (test_inputs - train_mean) / train_std

print(f"train_inputs normalized sample: \n{train_inputs[:5]}")

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Loss function and optimizer defined.")

# Initial outputs check
initial_outputs = model(train_inputs[:5])
print(f"Initial outputs: \n{initial_outputs}")

# Train the model with added print statements
num_epochs = 300
batch_size = 32

for epoch in range(num_epochs):
    permutation = torch.randperm(train_inputs.size()[0])
    epoch_loss = 0
    for i in range(0, train_inputs.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_inputs, batch_targets = train_inputs[indices], train_targets[indices]

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / (train_inputs.size()[0] // batch_size):.4f}')

    # Check if model parameters are being updated
    for name, param in model.named_parameters():
        print(f'{name} data after epoch {epoch + 1}: {param.data}')

print("Training complete.")

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'Gradients for {name}: {param.grad}')
    else:
        print(f'No gradients for {name}')

# Predict and evaluate the model
model.eval()
with torch.no_grad():
    preds = model(test_inputs)
    mse = criterion(preds, test_targets).item()
    print(f'Mean Squared Error: {mse:.4f}')
print("Prediction and evaluation complete.")

# Visualize the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test, T_test, c=test_targets.numpy(), cmap='viridis', label='True')
plt.title('True Solution')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_test, T_test, c=preds.numpy(), cmap='viridis', label='Predicted')
plt.title('Predicted Solution')
plt.colorbar()

plt.show()
print("Visualization complete.")