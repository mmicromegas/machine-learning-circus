import numpy as np

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Perceptron training with gradient descent
def train_perceptron(X, Y, activation_function,learning_rate=0.01, epochs=100):
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
                # in this case, error function or loss function is mean squared error (or squared loss) E = 1/2 * \sum_i  (y_i - o_i)^2
                # o_i = sigmoid(w_i*x_i + b)
                delta_w = (Y[i] - y_predicted)*y_predicted*(1-y_predicted)

                # Update weights and bias using delta rule
                weights += learning_rate * delta_w * X[i]
                bias += learning_rate * delta_w

                # print i, y_predicted, error, weights, bias
                print(f"i: {i}, Xi: {X[i]}, Yi: {Y[i]}, y_predicted: {y_predicted}, error: {delta_w}, weights: {weights}, bias: {bias}")

            # Optionally: calculate total loss for monitoring
            total_loss = np.sum((Y - sigmoid((np.dot(X, weights) + bias))) ** 2) / 2
            loss_history.append(total_loss)
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        else:
            for i in range(n_samples):
                # Calculate predicted output
                y_predicted = np.dot(X[i], weights) + bias

                # Calculate delta w (depends on activation function)
                delta_w = Y[i] - y_predicted

                # Update weights and bias using delta rule
                weights += learning_rate * delta_w * X[i]
                bias += learning_rate * delta_w

                # print i, y_predicted, error, weights, bias
                print(
                    f"i: {i}, Xi: {X[i]}, Yi: {Y[i]}, y_predicted: {y_predicted}, delta_w: {delta_w}, weights: {weights}, bias: {bias}")

            # Optionally: calculate total loss for monitoring
            total_loss = np.sum((Y - (np.dot(X, weights) + bias)) ** 2) / 2
            loss_history.append(total_loss)
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return weights, bias,loss_history


# Example usage
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
Y_train = np.array([0, 1, 1, 1])  # Target outputs

#X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
#Y_train = np.array([0, 0, 0, 1])  # Target outputs

learning_rate = 0.8
epochs = 15000
#activation_function = 'none'
activation_function = 'sigmoid'

weights, bias, loss_history = train_perceptron(X_train, Y_train, activation_function, learning_rate, epochs)
print(f"Trained Weights: {weights}, Bias: {bias}")

# plot loss_history
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.show(block=False)