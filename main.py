import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
# Data is generated from equatino: u=sin(πX)⋅exp(−πT)
def generate_synthetic_data(num_samples, num_cells):
    x = np.linspace(0, 1, num_cells)
    t = np.linspace(0, 1, num_samples)
    X, T = np.meshgrid(x, t)
    u = np.sin(np.pi * X) * np.exp(-np.pi * T)
    return X, T, u

X, T, U = generate_synthetic_data(1000, 100)

# Define the neural network model
class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = MLP()
model.build(input_shape=(None, 2))
model.summary()

# Prepare the data
num_samples = 1000
num_cells = 100
X, T, U = generate_synthetic_data(num_samples, num_cells)
X_train, X_test, T_train, T_test, U_train, U_test = train_test_split(X.flatten(), T.flatten(), U.flatten(), test_size=0.2)

train_inputs = np.vstack([X_train, T_train]).T
test_inputs = np.vstack([X_test, T_test]).T

# Normalize the data
train_mean = np.mean(train_inputs, axis=0)
train_std = np.std(train_inputs, axis=0)
train_inputs = (train_inputs - train_mean) / train_std
test_inputs = (test_inputs - train_mean) / train_std

print(f"train_inputs normalized sample: \n{train_inputs[:5]}")

# Define the loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Train the model with added print statements
num_epochs = 100
batch_size = 32
model.fit(train_inputs, U_train.reshape(-1, 1), batch_size=batch_size, epochs=num_epochs, verbose=1)

# Predict and evaluate the model
preds = model.predict(test_inputs)
mse = tf.keras.losses.MSE(U_test.reshape(-1, 1), preds).numpy().mean()
print(f'Mean Squared Error: {mse:.4f}')

# Visualize the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test, T_test, c=U_test, cmap='viridis', label='True')
plt.title('True Solution')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_test, T_test, c=preds.flatten(), cmap='viridis', label='Predicted')
plt.title('Predicted Solution')
plt.colorbar()

plt.show()
print("Visualization complete.")