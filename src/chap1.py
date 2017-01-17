
"""
Build a very simple, back propogation neural network
using stochiastic gradient descent.
Train it on the MNIST hand writing dataset for numbers.

http://neuralnetworksanddeeplearning.com/chap1.html
"""


import mnist_loader
import network


def main():
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	net = network.Network([784, 30, 10])
	net.SGD(training_data, 21, 15, 3, test_data=test_data)


if __name__ == '__main__':
	main()
