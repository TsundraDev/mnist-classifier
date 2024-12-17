import numpy as np
import sys

class NeuralNetwork:
    def __init__(self, input_layer_dim, hidden1_layer_dim, hidden2_layer_dim, output_layer_dim):
        self.W1 = np.random.rand(input_layer_dim, hidden1_layer_dim) * 2 - 1
        self.W2 = np.random.rand(hidden1_layer_dim, hidden2_layer_dim) * 2 - 1
        self.W3 = np.random.rand(hidden2_layer_dim, output_layer_dim) * 2 - 1

    def arelu(x):
        return np.where(x > 0, x, 0.01 * x)
    def arelu_der(x):
        return np.where(x > 0, 1, 0.01)

    def tanh_der(x):
        return 1 - np.tanh(x)**2

    def softmax(x):
        return np.exp(x) / np.exp(x).sum()

    def forward(self, input):
        output = input
        output = np.matmul(output, self.W1)
        output = np.tanh(output)

        output = np.matmul(output, self.W2)
        output = NeuralNetwork.arelu(output)

        output = np.matmul(output, self.W3)
        output = NeuralNetwork.softmax(output)
        return output

    def backward(self, input, target, learning_rate=0.001):
        grad_W1 = np.zeros(self.W1.shape)
        grad_W2 = np.zeros(self.W2.shape)
        grad_W3 = np.zeros(self.W3.shape)
        for elem in range(len(input)):
            # Calculate forward pass
            hidden_11 = np.matmul(input[elem], self.W1)
            hidden_12 = np.tanh(hidden_11)
            hidden_12 += np.random.rand(*hidden_12.shape) * 0.001

            hidden_21 = np.matmul(hidden_12, self.W2)
            hidden_22 = NeuralNetwork.arelu(hidden_21)
            hidden_22 += np.random.rand(*hidden_22.shape) * 0.001

            hidden_31 = np.matmul(hidden_22, self.W3)
            output = NeuralNetwork.softmax(hidden_31)

            # Calculate gradient values
            loss_W3 = output * target[elem].sum() - target[elem]

            loss_W2 = np.matmul(self.W3, loss_W3)
            loss_W2 = loss_W2 * NeuralNetwork.arelu_der(hidden_21)

            loss_W1 = np.matmul(self.W2, loss_W2)
            loss_W1 = loss_W1 * NeuralNetwork.tanh_der(hidden_11)

            loss_W1 = np.outer(input[elem], loss_W1)
            loss_W2 = np.outer(hidden_12, loss_W2)
            loss_W3 = np.outer(hidden_22, loss_W3)


            grad_W1 += loss_W1
            grad_W2 += loss_W2
            grad_W3 += loss_W3

        self.W3 -= learning_rate * grad_W3
        self.W2 -= learning_rate * grad_W2
        self.W1 -= learning_rate * grad_W1

    def test(self, input, target):
        total = 0
        correct = 0
        for i in range(len(input)):
            # Guess and check
            guess = np.argmax(self.forward(input[i]))
            result = np.argmax(target[i])

            correct += 1 if guess == result else 0
            total += 1

        print(f"Test : {correct/total * 100:.2f} %")
        return correct/total
