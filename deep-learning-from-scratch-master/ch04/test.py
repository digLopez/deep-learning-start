import numpy as np
from matplotlib import pylab
from matplotlib import pyplot
from mpl_toolkits import mplot3d


def function_1(_x):
    """"""
    return 0.1 * _x**3 + 0.1 * _x


def test_function_1():
    x = np.arange(0, 20, 0.1)
    y = function_1(x)
    pylab.plt.xlabel("x")
    pylab.plt.ylabel("y")
    pylab.plt.plot(x, y)
    pylab.plt.show()

    h = 10e-4
    dfx = (function_1(3 + h) - function_1(3 - h)) / 2 * h
    print(dfx)


def function_2(_x0, _x1):
    return _x0**2 + _x1**2


def test_function_2():
    fig = pyplot.figure()

    ax = mplot3d.Axes3D(fig)

    x = np.arange(-20, 20, 0.1)
    y = np.arange(-20, 20, 0.1)
    x, y = np.meshgrid(x, y)
    # r = np.sqrt(x**2 + y**2)
    z = function_2(x, y)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=pyplot.get_cmap("rainbow"))
    ax.contourf(x, y, z, zdir='z', offset=-2)
    ax.set_zlim(-2, 2)
    pyplot.show()


if __name__ == '__main__':
    test_function_2()
