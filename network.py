# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np

from multiprocessing import Process
from sys import getsizeof
from pympler.classtracker import ClassTracker
from pympler import asizeof
from scipy.sparse import random as rnd
from multiprocessing import Pool
import multiprocessing

import pdb

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x)
        #                for x, y in zip(sizes[:-1], sizes[1:])]
        #self.weights = self.generate_weights(sizes)

        self.tracker = None
        self.current_epoch = 0
        self.current_batch = None
        self.current_row = None

        self.global_weight_size = None
        self.batch_weight_size = None
        self.row_weight_size = None

    def generate_weights(self, sparse=False):
        sizes = self.sizes
        if sparse:
            self.weights = [rnd(y, x, density=.25).A
                        for x, y in zip(sizes[:-1], sizes[1:])]
            self.biases = [rnd(y, 1, density=.25).A for y in sizes[1:]]
        else:
            self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #return self.weights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, tracker=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        start_time = time.time()
        self.generate_weights(True)
        #tracker = ClassTracker()
        #tracker.track_class(Network)
        self.tracker = tracker
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        multithreaded = 2
        self.process_batches_ll(n, epochs, training_data, test_data, mini_batch_size, eta, n_test, multithreaded)
        #with Pool(8) as p:
        #    p.apply(self.process_batches, args=(n, epochs, training_data, test_data, mini_batch_size, eta, n_test))
        #p = Process(target=self.process_batches, args=(n, epochs, training_data, test_data, mini_batch_size, eta, n_test))
        #p.start()
        #p.join()
        print('time taken: ', (time.time()-start_time))

    def process_batches_ll(self, n, epochs, training_data, test_data, mini_batch_size, eta, n_test, multithreaded):
        if multithreaded==1:
            processes = []
            for i in range(epochs):
                processes.append(Process(target=self.process_batches, 
                                 args=(n, 1, training_data, test_data, mini_batch_size, eta, n_test)))
            for p in processes:
                p.start()
            for p in processes:
                p.join()
        elif multithreaded==2:
            args = []
            for i in range(epochs):
                args.append((n, 1, training_data, test_data, mini_batch_size, eta, n_test))
            print('doing pools', multiprocessing.cpu_count())
            with Pool(multiprocessing.cpu_count()) as p:
                    p.map(self.process_batches_wrapper, args)
        else:
            self.process_batches(n, epochs, training_data, test_data, mini_batch_size, eta, n_test)

    def process_batches_wrapper(self, args):
        return self.process_batches(*args)

    def process_batches(self, n, epochs, training_data, test_data, mini_batch_size, eta, n_test):
        for j in range(epochs):
            current_epoch = j+1

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            #print('mini batch length ', len(mini_batches))
            #print('current epoch: ', self.current_epoch, ' current batch: ', self.current_batch, ' current row: ', self.current_row)
            #print('global size: ', self.global_weight_size, ' batch size: ', self.batch_weight_size, ' row size: ', self.row_weight_size)
            current_batch = 0
            for mini_batch in mini_batches:
                current_batch += 1
                self.current_batch = current_batch
                self.update_mini_batch(mini_batch, eta)
            self.current_epoch = self.current_epoch + 1
            if test_data:
                print("Epoch {} : {} / {}".format(self.current_epoch,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(self.current_epoch))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #nabla_b = [np.zeros(b.shape) for b in np.where(abs(grad) > .01)[0]]

        current_row = 0
        #self.batch_weight_size = asizeof.asizeof(nabla_w)
        #self.global_weight_size = asizeof.asized(self.weights, detail=4, limit=4).format()

        for x, y in mini_batch:
            current_row += 1
            self.current_row = current_row
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #self.row_weight_size = asizeof.asizeof(delta_nabla_w)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #print('overall weights')
        #for w in self.weights: print('dim ', w.shape, 'zeros ', w[np.where(w==0)].size)
        #print('weights size: ', getsizeof(self.weights))
        #print('weights0 size: ', getsizeof(self.weights[0]))
        #print('weights1 size: ', getsizeof(self.weights[1]))
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        '''print('<backprop>')
        self.tracker.create_snapshot()
        self.tracker.stats.print_summary()
        print('</backprop>')'''
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #pdb.set_trace()
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        #print('number of layers ', self.num_layers)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #print('l is', l)
            #print('delta is ', delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        #print('batch weights')
        #for w in nabla_w: print('dim ', w.shape, 'zeros ', w[np.where(w==0)].size)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        #print('\nactivation\n', output_activations, '\ny\n', y)
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    sp = sigmoid(z)*(1-sigmoid(z))
    #print('sp is ', sp)
    return sp
