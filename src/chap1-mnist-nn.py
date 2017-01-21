
'''
Build a very simple, back propogation neural network
using stochiastic gradient descent.
Train it on the MNIST hand writing dataset for numbers.

http://neuralnetworksanddeeplearning.com/chap1.html
'''

import mnist_loader
import network


def main():
    ''' These settings achieved 95% accuracy on my particular hardware setup.
    '''
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 15, 3, test_data=test_data)
    ''' And these settings achieved 97% accuracy, but took about
		4 times longer to run.

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 90, 60, 30, 10])
    net.SGD(training_data, 30, 15, 3, test_data=test_data)
    '''

if __name__ == '__main__':
    main()
