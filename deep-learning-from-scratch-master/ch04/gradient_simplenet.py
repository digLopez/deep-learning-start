# coding: utf-8
import sys
import os
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        # 隐层
        z = self.predict(x)
        # 输出层
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
# 求出权重w的变化方向
dW = numerical_gradient(f, net.W)

print(dW)
