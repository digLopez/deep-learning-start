# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from ch04.gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    梯度下降
    :param f:
    :param init_x: 初始向量
    :param lr: 学习率
    :param step_num: 学习次数
    :return:
    """
    x = init_x
    # 记录每次下降后的损失值
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        # 计算梯度
        grad = numerical_gradient(f, x)
        # 梯度下降
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
