import numpy as np
import logging

logging.basicConfig(filename="./logs/log.log",
                    level=logging.DEBUG,
                    format="%(levelname)s %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s",
                    datefmt="%Y-%b-%d[%H:%M:%S]")


def init_network():
    """
    init weight matrixes for all layers 
    """
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])

    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])

    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


def sigmoid(x):
    """
    used for hiden layer
    param x: input
    """
    return 1/(1 + np.exp(-x))


def identify_function(x):
    """
    used for output layer
    param x: input
    """
    return x


def softmax(a):
    """
    used for output layer
    """
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def forward(network, x):
    """
    build forward-cast network
    param network: weight network
    param x: input vector x
    """
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    logging.debug("layer number [{}]".format(int(len(network)/2)+1))

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)

    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

