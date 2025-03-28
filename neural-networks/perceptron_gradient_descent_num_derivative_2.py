import os

import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Example activation functions
def activation_sigmoid(h):
    return 1 / (1 + np.exp(-h))  # Sigmoid

def activation_tanh(h):
    return np.tanh(h)  # Hyperbolic Tangent

def activation_relu(h):
    return np.maximum(0, h)  # Rectified Linear Unit (ReLU)

def activation_leaky_relu(h, alpha=0.01):
    return np.where(h > 0, h, h * alpha)  # Leaky ReLU

def activation_gelu(h):
    return 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * np.power(h, 3))))  # GELU

# Additional activation functions
def activation_identity(h):
    return h

def activation_binary_step(h):
    return np.where(h > 0, 1, 0)

def activation_soboleva(h):
    return np.tanh(h) - 0.5 * np.power(h, 3)

def activation_softplus(h):
    return np.log(1 + np.exp(h))

def activation_elu(h, alpha=1.0):
    return np.where(h > 0, h, alpha * (np.exp(h) - 1))

def activation_selu(h, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    return scale * np.where(h > 0, h, alpha * (np.exp(h) - 1))

def activation_prelu(h, alpha=0.01):
    return np.where(h > 0, h, alpha * h)

def activation_rpsu(h, alpha=0.01):
    return np.where(h > 0, 1 / (1 + np.exp(-h)), alpha * h)

def activation_silu(h):
    return h * 1 / (1 + np.exp(-h))

def activation_elish(h):
    return h * np.tanh(np.log(1 + np.exp(h)))

def activation_gaussian(h):
    return np.exp(-np.power(h, 2))

def activation_sinusoid(h):
    return np.sin(h)

# Loss functions
def loss_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def loss_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def loss_cross_entropy(y_true, y_pred):
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

def loss_hinge(y_true, y_pred):
    return np.maximum(0, 1 - y_true * y_pred)

def loss_huber(y_true, y_pred, delta=1.0):
    return np.where(np.abs(y_true - y_pred) < delta, 0.5 * np.square(y_true - y_pred), delta * (np.abs(y_true - y_pred) - 0.5 * delta))

def loss_kl_divergence(y_true, y_pred):
    return y_true * np.log(y_true / y_pred)

def loss_msle(y_true, y_pred):
    return np.mean(np.square(np.log(y_true + 1) - np.log(y_pred + 1)))

def loss_poisson(y_true, y_pred):
    return np.mean(y_pred - y_true * np.log(y_pred))

def loss_squared_hinge(y_true, y_pred):
    return np.mean(np.square(np.maximum(1 - y_true * y_pred, 0)))

def loss_vae(y_true, y_pred, mu, log_var):
    return np.mean(np.square(y_true - y_pred) + np.exp(log_var) - 1 - log_var)

def loss_wasserstein(y_true, y_pred):
    return np.mean(y_true * y_pred)

def loss_exponential(y_true, y_pred):
    return np.exp(-y_true * y_pred)

def loss_quantile(y_true, y_pred, q=0.5):
    return np.maximum(q * (y_true - y_pred), (q - 1) * (y_true - y_pred))

def loss_logistic(y_true, y_pred):
    return np.log(1 + np.exp(-y_true * y_pred))

def loss_tangent(y_true, y_pred):
    return np.tan(y_true - y_pred)

def loss_savage(y_true, y_pred):
    return np.maximum(0, 1 - y_true * y_pred)

def loss_square(y_true, y_pred):
    return np.square(y_true - y_pred)

# Numerical derivatives
def g_prime(h_val):
    return numerical_derivative(activation, h_val)

def dL_dy(y_true, y_pred):
    return numerical_derivative(lambda y: loss(y_true, y), y_pred)

# Perceptron training with gradient descent
def train_perceptron(X, Y, activation_func, loss_func, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    loss_history = []
    for epoch in range(epochs):
        for i in range(n_samples):
            # Calculate predicted output
            o_predicted = activation_func(np.dot(X[i], weights) + bias)

            # Calculate delta w (depends on error function a.k.a loss function and activation function)
            weight_update = -dL_dy(Y[i], o_predicted) * g_prime(o_predicted)
            delta_w = learning_rate * weight_update

            # Update weights and bias using delta rule
            weights += learning_rate * delta_w * X[i]
            bias += learning_rate * delta_w

        # Optionally: calculate total loss for monitoring
        total_loss = np.sum((Y - activation_func(np.dot(X, weights) + bias)) ** 2) / 2
        loss_history.append(total_loss)

    return weights, bias, loss_history

# Example usage
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
Y_train = np.array([0, 0, 0, 1])  # Target outputs

learning_rate = 0.8
epochs = 2000

activation_functions = {
    'Sigmoid': activation_sigmoid,
    'Tanh': activation_tanh,
    'ReLU': activation_relu,
    'Leaky ReLU': activation_leaky_relu,
    'GELU': activation_gelu,
    'Identity': activation_identity,
    'Binary Step': activation_binary_step,
    'Soboleva': activation_soboleva,
    'Softplus': activation_softplus,
    'ELU': activation_elu,
    'SELU': activation_selu,
    'PReLU': activation_prelu,
    'RPSU': activation_rpsu,
    'SiLU': activation_silu,
    'ELiSH': activation_elish,
    'Gaussian': activation_gaussian,
    'Sinusoid': activation_sinusoid
}

loss_functions = {
    'MSE': loss_mse,
    'MAE': loss_mae,
    #'Cross Entropy': loss_cross_entropy
    'Hinge': loss_hinge,
    'Huber': loss_huber,
    'KL Divergence': loss_kl_divergence,
    'MSLE': loss_msle,
    'Poisson': loss_poisson,
    'Squared Hinge': loss_squared_hinge,
    #'VAE': loss_vae,
    'Wasserstein': loss_wasserstein,
    'Exponential': loss_exponential,
    'Quantile': loss_quantile,
    'Logistic': loss_logistic,
    'Tangent': loss_tangent,
    'Savage': loss_savage,
    'Square': loss_square
}

for loss_name, loss_func in loss_functions.items():
    loss_histories = {}
    for activation_name, activation_func in activation_functions.items():
        activation = activation_func
        loss = loss_func
        _, _, loss_history = train_perceptron(X_train, Y_train, activation_func, loss_func, learning_rate, epochs)
        loss_histories[activation_name] = loss_history

    # Plot loss history for all activation functions for the current loss function
    plt.figure(figsize=(10, 6))
    for activation_name, loss_history in loss_histories.items():
        plt.plot(loss_history, label=activation_name)

    plt.yscale('log')
    plt.ylim(1e-20, 1.e3)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(f'Total Loss as Function of Epoch for Different Activation Functions\nUsing {loss_name} Loss')
    plt.legend()
    plt.savefig(os.path.join('OUTPUT',f'loss_history_{loss_name}.png'))
    plt.close()