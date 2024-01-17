from math import *
import numpy as np

# X0 = [7, 0, 1]
# y0, z0 = 3, 2

X0 = [7, 0, 1]
a = pi / 3
y0, z0 = 13, 12

d = 10 ** (-3)


def turn(y, z):
    yN = (y - y0) * cos(a) + (z - z0) * sin(a)
    zN = (z - z0) * cos(a) - (y - y0) * sin(a)
    return yN, zN


def f(x, y, z):
    yN, zN = turn(y, z)
    return 3 * x ** 2 + cosh(2 * yN) + exp(zN ** 2)


def grad(vec):
    eps = 10 ** (-5)
    x, y, z = vec
    func = f(x, y, z)
    return [(f(x + eps, y, z) - func) / eps,
            (f(x, y + eps, z) - func) / eps,
            (f(x, y, z + eps) - func) / eps]


point = X0


def g(t, p_):
    x = point[0] + p_[0] * t
    y = point[1] + p_[1] * t
    z = point[2] + p_[2] * t

    return f(x, y, z)


def methGS(a, b, p):
    F = ((5 ** (0.5) + 1) / 2)  # = 1,61

    x = a + (b - a) / F
    lx = a + b - x

    Fx = g(x, p)
    Flx = g(lx, p)

    while abs(b - a) > 2 * d:
        if Flx > Fx:
            a = lx
            lx = x
            Flx = Fx
        else:
            b = x

        x = a + b - lx
        Fx = g(x, p)

        if lx > x:
            lx, x = x, lx
            Flx, Fx = Fx, Flx

    return lx


def secondDer(vec):
    eps = 10 ** (-3)
    x, y, z = vec
    fXX = 6
    fYY = (f(x, y + eps, z) + (x, y - eps, z) - 2 * f(x, y, z)) / (eps ** 2)
    # 4 * cos(a)**2 * cosh(2 * cos(a) * y + (2*z - 2*z0) * sin(a) - 2 * y0 *cos(a))
    fZZ = (f(x, y, z + eps) + (x, y, z - eps) - 2 * f(x, y, z)) / (eps ** 2)
    fYZ = 4 * cos(a) * sin(a) * cosh(2 * sin(a) * z - 2 * z0 * sin(a) + (2 * y - 2 * y0) * cos(a))

    return np.array([[fXX, 0, 0],
                     [0, fYY, fYZ],
                     [0, fYZ, fZZ]])


def invMat(vec):
    return np.linalg.inv(secondDer(vec))


lastP = point.copy()

it = 1

while True:
    invM = invMat(point)
    vector = grad(point)
    if np.linalg.norm(vector) != 0:
        vector /= np.linalg.norm(vector)

    for i in range(3):
        point[i] -= (invM[i][0] * vector[0] + invM[i][1] * vector[1] + invM[i][2] * vector[2])

    if max(abs(lastP[0] - point[0]), abs(lastP[1] - point[1]), abs(lastP[2] - point[2])) < 0.001:
        break
    lastP = point.copy()
    it += 1

print('it =', it)
print('min =', point)