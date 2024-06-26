import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
# Data is generated from equation: u=sin(πX)⋅exp(−πT)
def generate_synthetic_data(num_samples, num_cells):
    # x: spatial coordinate, t: time, u: solution value
    # sin(πX): creates sine graph, exp(−πT): creates exponential decay
    x = np.linspace(0, 1, num_cells)
    t = np.linspace(0, 1, num_samples)
    X, T = np.meshgrid(x, t)
    u = np.sin(np.pi * X) * np.exp(-np.pi * T)
    return X, T, u

X, T, U = generate_synthetic_data(1000, 100)

# Define the neural network model
# MLP: Multilayer Perceptron: feedforward artificial neural network (info flows in one direction w/o cycle or loops)
def create_mlp():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

model = create_mlp()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# Prepare the data
num_samples = 1000 # 1000 time samples
num_cells = 100 # # 100 spatial points
# X, T are 2D array of shape(1000,100) and U is the solution value
X, T, U = generate_synthetic_data(num_samples, num_cells)
# .flatten converts 2D arrays to 1D arrays
# 0.2 means 20% is testing, 80% is training
X_train, X_test, T_train, T_test, U_train, U_test = train_test_split(X.flatten(), T.flatten(), U.flatten(), test_size=0.2)


train_inputs = np.vstack([X_train, T_train]).T # vstack (stack them up) making 2D array
test_inputs = np.vstack([X_test, T_test]).T

# Normalize the data
# Normalization helps in faster convergence during training
train_mean = np.mean(train_inputs, axis=0)
train_std = np.std(train_inputs, axis=0)
train_inputs = (train_inputs - train_mean) / train_std
test_inputs = (test_inputs - train_mean) / train_std

print(f"train_inputs normalized sample: \n{train_inputs[:5]}") # print first 5 rows for verification

# Normalization ensures all input features are on a similar scale, which is important for neural network training.
# The same normalization parameters (mean and std) from the training data are applied to the test data to ensure consistency.




# Define the loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Adam optimizer(adaptive learning algorithm) with a learning rate of 0.001
model.compile(optimizer=optimizer, loss='mse') # MSE is a common loss function for regression problems, measuring the average squared difference between predictions and actual values

# Train the model with added print statements
num_epochs = 100
batch_size = 32 # num of samples processed before model is updated
model.fit(train_inputs, U_train.reshape(-1, 1), batch_size=batch_size, epochs=num_epochs, verbose=1)

# Predict and evaluate the model
preds = model.predict(test_inputs)
mse = tf.keras.losses.MSE(U_test.reshape(-1, 1), preds).numpy().mean()
print(f'Mean Squared Error: {mse:.10f}')

# Below is the previous visualization
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(X_test, T_test, c=U_test, cmap='viridis', label='True')
# plt.title('True Solution')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.scatter(X_test, T_test, c=preds.flatten(), cmap='viridis', label='Predicted')
# plt.title('Predicted Solution')
# plt.colorbar()
# plt.show()
# print("Visualization complete.")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(U_test, preds, alpha=0.5)
plt.plot([U_test.min(), U_test.max()], [U_test.min(), U_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.show()