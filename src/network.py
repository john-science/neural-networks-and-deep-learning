"""
The stochastic gradient descent learning algorithm for a feedforward neural network.
Gradients are calculated using backpropagation.
"""

import numpy as np
from scipy.special import expit as sigmoid
from time import time


class Network(object):

    def __init__(self, sizes):
        """ The list `sizes` contains the number of neurons in each layer of the network. The
            biases and weights for the network are initialized using a Gaussian distribution with
            mean 0, and variance 1.  Note that the first layer is assumed to be an input layer,
            and by convention we won't set any biases for those neurons.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """ Return the output of the network given the input. """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Train the neural network using mini-batch stochastic gradient descent.
            The `training_data` is a list of tuples `(x, y)` representing the
            training inputs and the desired outputs.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        original = time()
        for j in range(epochs):
            start = time()
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                if j % 3 == 0 or j == (epochs - 1):
                    print("Epoch {0}: {1} / {2} in {3} seconds".format(j,
                        self.evaluate_test(test_data), n_test, "%0.2f" % (time() - start)))
            else:
                print("Epoch {0} in {1} seconds".format(j, "%0.2f" % (time() - start)))

        print("Total Time: {0}".format("%0.2f" % (time() - original)))

    def update_mini_batch(self, mini_batch, eta):
        """ Update the network's weights and biases by applying gradient descent using
            backpropagation to a single mini batch.
            The `mini_batch` is a list of tuples `(x, y)`, and `eta` is the learning rate.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for train_in, train_out in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(train_in, train_out)
            grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, grad_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, grad_b)]

    def backprop(self, train_in, train_out):
        """ Return a tuple `(grad_b, grad_w)` representing the gradient for the cost
            function C_x.  `grad_b` and `grad_w` are layer-by-layer lists of numpy arrays,
            similar to `self.biases` and `self.weights`.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = train_in
        activations = [train_in]  # list to store all the activations, layer by layer
        zs = []                   # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], train_out) * sigmoid_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (grad_b, grad_w)

    def cost_derivative(self, output_activations, y):
        """ Return the vector of partial derivatives
        \partial C_x / \partial a
        for the output activations.
        """
        return (output_activations - y)  # TODO: How is this a partial derivative?

    def evaluate_test(self, test_data):
        """ Return the number of test inputs for which the neural network outputs the correct
            result. Note that the neural network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(train_in)), train_out)
                        for (train_in, train_out) in test_data]
        return sum(train_out[out] for (out, train_out) in test_results)


def sigmoid_prime(z):
    """ Derivative of the sigmoid function. """
    return sigmoid(z) * (1.0 - sigmoid(z))
