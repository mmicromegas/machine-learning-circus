import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Perceptron training with gradient descent
def train_perceptron(X, Y, activation_function, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    loss_history = []
    for epoch in range(epochs):
        if activation_function == 'sigmoid':
            for i in range(n_samples):
                # Calculate predicted output
                y_predicted = sigmoid(np.dot(X[i], weights) + bias)

                # Calculate delta w (depends on error function a.k.a loss function and activation function)
                delta_w = (Y[i] - y_predicted) * y_predicted * (1 - y_predicted)

                # Update weights and bias using delta rule
                weights += learning_rate * delta_w * X[i]
                bias += learning_rate * delta_w

            # Optionally: calculate total loss for monitoring
            total_loss = np.sum((Y - sigmoid(np.dot(X, weights) + bias)) ** 2) / 2
            loss_history.append(total_loss)
        else:
            for i in range(n_samples):
                # Calculate predicted output
                y_predicted = np.dot(X[i], weights) + bias

                # Calculate delta w (depends on activation function)
                delta_w = Y[i] - y_predicted

                # Update weights and bias using delta rule
                weights += learning_rate * delta_w * X[i]
                bias += learning_rate * delta_w

            # Optionally: calculate total loss for monitoring
            total_loss = np.sum((Y - (np.dot(X, weights) + bias)) ** 2) / 2
            loss_history.append(total_loss)

    return weights, bias, loss_history

# Example usage
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
Y_train = np.array([0, 0, 0, 1])  # Target outputs

#X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
#Y_train = np.array([0, 1, 1, 1])  # Target outputs

learning_rate = 0.8
activation_function = 'sigmoid'

max_epochs_list = range(1000, 100001, 1000)
weights_history = []
bias_history = []
loss_history_max_epochs = []

for max_epochs in max_epochs_list:
    # print max_epochs
    print(f"Training perceptron with {max_epochs} epochs")
    weights, bias, loss_history = train_perceptron(X_train, Y_train, activation_function, learning_rate, max_epochs)
    weights_history.append(weights)
    bias_history.append(bias)
    if max_epochs == max(max_epochs_list):
        loss_history_max_epochs = loss_history

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(9, 6))

# Plot weights in individual subplots
axs[0, 0].plot(max_epochs_list, [w[0] for w in weights_history], label='Weight 1')
axs[0, 0].set_ylabel('Weight')
axs[0, 0].set_title('Weight 1 as Function of Epochs')
axs[0, 0].set_ylim(0, 12)
axs[0, 0].axhline(y=10, color='r', linestyle='--', label='y=10')
axs[0, 0].legend()

axs[0, 1].plot(max_epochs_list, [w[1] for w in weights_history], label='Weight 2')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Weight')
axs[0, 1].set_title('Weight 2 as Function of Epochs')
axs[0, 1].set_ylim(0, 12)
axs[0, 1].axhline(y=10, color='r', linestyle='--', label='y=10')
axs[0, 1].legend()

# Plot bias in the third subplot
axs[1, 0].plot(max_epochs_list, bias_history, label='Bias')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Bias')
axs[1, 0].set_title('Bias as Function of Epochs')
axs[1, 0].set_ylim(-20, 0)
axs[1, 0].axhline(y=-15, color='r', linestyle='--', label='y=-15')
axs[1, 0].legend()

# Plot loss history in the fourth subplot
axs[1, 1].plot(range(1, len(loss_history_max_epochs) + 1), loss_history_max_epochs, label='Loss History')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].set_title('Loss History as Function of Epochs')
axs[1, 1].set_yscale('log')
axs[1, 1].set_ylim(1e-6, 1)
axs[1, 1].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show(block=False)

# save plot to png
plt.savefig("perceptron_gradient_descent_explore_and_gate.png")