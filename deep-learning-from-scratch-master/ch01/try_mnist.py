import sys
import os
from dataset import mnist

sys.path.append(os.pardir)

(x_train, t_train), (x_test, t_test) = mnist.load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
