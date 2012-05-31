from __future__ import division
import numpy as np
import numpy.linalg as npl
from pyprox.utils import soft_thresholding

class MathFunction(object):
    # todo compute expression for grads
    def __call__(self, x):
        return self.eval(x)

    def eval(self, x):
        return None

    def prox(self, x, tau):
        raise Exception('Proximity operator does not exist/is not '
                        'implemented.')

    def grad(self, x):
        raise Exception('Function has no derivative in this point/is not '
                        'implemented.')

    def translate_by(self, z):
        mf = MathFunction()
        mf.eval = lambda x: self.eval(x - z)
        mf.prox = lambda x,tau: z + self.prox(x-z, tau)
        mf.grad = lambda x: self.grad(x)

    def scaling_by(self, rho):
        if rho == 0:
            raise Exception('Cannot scale by 1/0.')
        mf = MathFunction()
        mf.eval = lambda x: self.eval(x/rho)
        mf.prox = lambda x,tau: rho * self.prox(x/rho, 1/(rho ** 2))
        mf.grad = None

    def reflexion(self):
        mf = MathFunction()
        mf.eval = lambda x: self.eval(-x)
        mf.prox = lambda x,tau: -self.prox(-x, tau)
        mf.grad = None

    def quadratic_perturbation(self, u, alpha, gamma):
        mf = MathFunction()
        mf.eval = lambda x: self.eval(x) + alpha/2 * sum(sum(x*x)) + np.dot\
            (u,x) + gamma
        mf.prox = lambda x,tau: self.prox((x-u)/(alpha+1), 1/(alpha+1))
        mf.grad = None

    def conjugate(self):
        mf = MathFunction()
        mf.eval = None
        mf.prox = lambda x,tau: x - self.prox(x, tau)
        mf.grad = None

class LeastSquareDataFit(MathFunction):

    def __init__(self,A,y):
        self.A = A
        self.y = y

    def eval(self, x):
        return (1/2) * npl.norm(self.y - self.A(x)) ** 2

    def grad(self, x):
        return self.A.T(self.A(x) - self.y)

    #    def prox(self, x, tau):
    #        return x + self.A.T((self.A * self.A.T).inv(self.y - self.A(x)))

    def _latex_(self):
        return "\\| y - A x \\|_2^2"

class L1Norm(MathFunction):

    def __init__(self,la=1.0):
        self.la = la

    def eval(self,x):
        return npl.norm(x, 1)

    def prox(self,x,tau):
        return soft_thresholding(x,self.la*tau)

    def _latex_(self):
        return "\\lambda \\| x \\|_1"

class L2Norm(MathFunction):

    def __init__(self,la=1.0):
        self.la = la

    def eval(self,x):
        return npl.norm(x)

    def prox(self,x,tau):
        # todo compute expression
        return None

    def _latex_(self):
        return "\\lambda \\| x \\|_2"

class LpNorm(MathFunction):

    def __init__(self,la=1.0, p=1):
    # to put in a factory function ?
    #        if p == 1:
    #            return L1Norm(la)
    #        elif p == 2:
    #            return L2Norm(la)
    #        elif p == 'inf' or p == 'Inf' or p == np.Inf:
    #            return None
    #        else:
        self.p = p
        self.la = la

    def eval(self,x):
        return npl.norm(x, p)

class L1L2Norm(MathFunction):

    def __init__(self,la=1.0,blocks=None):
        self.la = la
        self.blocks = blocks

    def eval(self,x):
        res = 0.0
        for block in self.blocks:
            res += npl.norm(x[block])
        return self.la * res

    def prox(self, x, tau):
        xres = x.copy()
        for block in self.blocks:
            energy = npl.norm(x[block]) ** 2
            if energy < 1e-10:
                xres[block] = x[block]
            xres[block] = np.maximum(1 - abs(tau) / energy, 0) * x[block]
        return xres

    def _latex_(self):
        return "\\lambda \\| x \\|_{1,2}"
