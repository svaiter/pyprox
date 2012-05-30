"""
Proximal algorithms
"""

from __future__ import division
import numpy as np
import math
from .utils import operator_norm, soft_thresholding

def _output_helper(full_output, retall, x, fx, iterations, allvecs):
    if full_output:
        retlist = x, fx
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist

def douglas_rachford(prox_f, prox_g, x0,
                     maxiter=1000, mu = 1, gamma = 1,
                     full_output=0, retall=0, callback=None):
    """
    dr
    """
    def rProx_f(x, tau):
        return 2*prox_f(x, tau) - x
    def rProx_g(x, tau):
        return 2*prox_g(x, tau) - x

    x = x0.copy()
    y = x0.copy()
    allvecs = [x]
    fx = []
    iterations = 1

    while iterations < maxiter:
        y = (1-mu/2)*y + mu/2*rProx_f(rProx_g(y, gamma), gamma)
        x = prox_g(y, gamma)

        if callback is not None:
            fx.append(callback(x))
        iterations += 1
        if retall:
            allvecs.append(x)

    return _output_helper(full_output, retall, x, fx, iterations, allvecs)

def forward_backward(prox_f, grad_g, x0, L,
                     maxiter=1000, method='fb', fbdamping=1.8,
                     full_output=0, retall=0, callback=None):
    t = 1
    tt = 2/L
    gg = 0
    A = 0
    y = x0.copy()
    x = x0.copy()

    allvecs = [x]
    fx = []
    iterations = 1

    while iterations <= maxiter:
        if method == 'fb':
            x = prox_f(x - fbdamping/L * grad_g(x), fbdamping/L)
        elif method == 'fista':
            xnew = prox_f(y - 1/L * grad_g(y), fbdamping/L)
            tnew = (1+math.sqrt(1 + 4* t ** 2))/2
            y = xnew + (t-1)/tnew * (xnew-x)
            x = xnew
            t = tnew
        elif method == 'nesterov':
            a = (tt + math.sqrt(tt ** 2 + 4*tt*A))/2
            v = prox_f(x0 - gg, A)
            z = (A*x + a*v)/(A+a)
            x = prox_f(z - 1/L * grad_g(z), 1/L)
            gg += a * grad_g(x)
            A += a
        else:
            raise Exception('ex a def in fb')

        if callback is not None:
            fx.append(callback(x))
        iterations += 1
        if retall:
            allvecs.append(x)

    return _output_helper(full_output, retall, x, fx, iterations, allvecs)

def admm(prox_fs, prox_g, K, KS, x0,
         maxiter=100, theta = 1, sigma=None, tau=None,
         full_output=0, retall=0, callback=None):
    """
    dr
    """
    if not(sigma and tau):
        L = operator_norm(
            lambda x : KS(K(x)),
            np.random.randn(x0.shape[0],1)
        )
        sigma = 10
        tau = .9 / (sigma * L)

    x = x0.copy()
    x1 = x0.copy()
    xold = x0.copy()
    y = K(x)
    allvecs = [x]
    fx = []
    iterations = 1

    while iterations < maxiter:
        xold = x.copy()
        y = prox_fs(y + sigma*K(x1), sigma)
        x = prox_g(x - tau*KS(y), tau)
        x1 = x + theta * (x-xold)

        if callback is not None:
            fx.append(callback(x))
        iterations += 1
        if retall:
            allvecs.append(x)

    return _output_helper(full_output, retall, x, fx, iterations, allvecs)

def iterative_soft_thresholding(A,y,x0=None,
                                maxiter=1000, method='fb', fbdamping=1.8,
                                full_output=0, retall=0, callback=None):
    ProxF = soft_thresholding
    GradG = lambda x : np.dot(A.T,np.dot(A,x) - y)
    L = np.linalg.norm(A, 2) ** 2
    if x0 is None:
        x0 = np.zeros((A.T*y).size)
    return forward_backward(ProxF, GradG, x0, L, maxiter=maxiter,
        method=method, fbdamping=fbdamping,full_output=full_output,
        retall=retall, callback=callback)