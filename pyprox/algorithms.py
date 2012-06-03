"""
The :mod:`pyprox.algorithms` module includes the proximal schemes of pyprox.
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>

from __future__ import division
import numpy as np
import math
from pyprox.utils import operator_norm

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
    """Minimize the sum of two functions using the Douglas Rachford splitting.
    scheme.

    This algorithm assumes that F, G are both "proximable" where the
    optimization objective reads::

        F(x) + G(x)

    Parameters
    ----------
    prox_f : callable
        should take two arguments : an ndarray and a float.
    prox_g : callable
        same as prox_f.
    x0 : ndarray
        initial guess for the solution.
    maxiter : int, optional
        maximum number of iterations.
    mu : float, optional
    gamma : float, optional
    full_output : bool, optional
        non-zero to return all optional outputs.
    retall : bool, optional
        Return a list of results at each iteration if non-zero.
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    xrec: ndarray
    fx: list

    References
    ----------
    Proximal Splitting Methods in Signal Processing,
    Patrick L. Combettes and Jean-Christophe Pesquet, in:
    Fixed-Point Algorithms for Inverse Problems in Science and Engineering,
    New York: Springer-Verlag, 2010.
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
    """Minimize the sum of two functions using the Forward-backward splitting.
    scheme.

    This algorithm assumes that F, G is "proximable" and L has a
    L-Lipschitz gradient where the optimization objective reads::

        F(x) + G(x)

    Parameters
    ----------
    prox_f : callable
        should take two arguments : an ndarray and a float.
    grad_g : callable
        same as prox_f.
    x0 : ndarray
        initial guess for the solution.
    L : float
        Module of Lipschitz of nabla G.
    maxiter : int, optional
        maximum number of iterations.
    method : string, optional,
        can be 'fb', 'fista' or 'nesterov'
    fbdamping : float, optional
    full_output : bool, optional
        non-zero to return all optional outputs.
    retall : bool, optional
        Return a list of results at each iteration if non-zero.
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    xrec: ndarray
    fx: list

    References
    ----------
    P. L. Combettes and V. R. Wajs, Signal recovery by proximal
    forward-backward splitting,
    Multiscale Model. Simul., 4 (2005), pp. 1168-1200
    """
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
            xnew = prox_f(y - 1/L * grad_g(y), 1/L)
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

def forward_backward_dual(grad_fs, prox_gs, K, KS, x0, L,
                          maxiter = 100, method='fb', fbdamping=1.8,
                          full_output=0, retall=0, callback=None):
    """Minimize the sum of the strongly convex function and a proper convex
    function.

    This algorithm minimizes

        F(x) + G(K(x))

    where F is strongly convex, G is a proper convex function and K is a
    linear operator by a duality argument.

    Parameters
    ----------
    grad_fs : callable
        should take one argument : an ndarray.
    prox_gs : callable
        should take two arguments : an ndarray and a float.
    K : callable or ndarray
        a linear operator
    KS : callable or ndarray
        the dual linear operator
    x0 : ndarray
        initial guess for the solution.
    L : float
        Module of Lipschitz of nabla G.
    maxiter : int, optional
        maximum number of iterations.
    method : string, optional,
        can be 'fb', 'fista' or 'nesterov'
    fbdamping : float, optional
    full_output : bool, optional
        non-zero to return all optional outputs.
    retall : bool, optional
        Return a list of results at each iteration if non-zero.
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    xrec: ndarray
    fx: list

    Notes
    -----
    This algorithm use the equivalence of

        min_x F(x) + G(K(x))        (*)

    with

        min_u F^*(-K(u)) + G^*(u)   (**)

    using x = grad(F^*)(-K(u)) where the convex dual function is

        F^*(y) = sup_x = <x,y> - F(x)

    It uses `forward_backward` as a solver of (**)
    """
    new_callback = lambda u : callback(grad_fs(-KS(u)))
    new_grad = lambda u : - K(grad_fs(-KS(u)))
    u0 = K(x0)
    res = forward_backward(prox_gs, new_grad, u0, L, maxiter=maxiter,
        method=method, fbdamping=fbdamping, full_output=full_output,
        retall=retall, callback=new_callback)

    res[0] = grad_fs(-KS(res[0]))
    return res

def admm(prox_fs, prox_g, K, KS, x0,
         maxiter=100, theta = 1, sigma=None, tau=None,
         full_output=0, retall=0, callback=None):
    """Minimize an optimization problem using the Preconditioned Alternating
     direction method of multipliers

    This algorithm assumes that F, G are both "proximable" where the
    optimization objective reads::

        F(K(x)) + G(x)

    where K is a linear operator.

    Parameters
    ----------
    prox_fs : callable
        should take two arguments : an ndarray and a float.
    prox_g : callable
        same as prox_f.
    K : callable or ndarray
        a linear operator
    KS : callable or ndarray
        the dual linear operator
    x0 : ndarray
        initial guess for the solution.
    maxiter : int, optional
        maximum number of iterations.
    theta : float, optional
    sigma : float, optional
        parameters of the method.
        They should satisfy sigma * tay * norm(K)^2 < 1
    full_output : bool, optional
        non-zero to return all optional outputs.
    retall : bool, optional
        Return a list of results at each iteration if non-zero.
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    xrec: ndarray
    fx: list

    References
    ----------
    A. Chambolle and T. Pock,
    A First-Order Primal-Dual Algorithm for Convex Problems
    with Applications to Imaging,
    JOURNAL OF MATHEMATICAL IMAGING AND VISION
    Volume 40, Number 1 (2011)
    """
    if isinstance(K, np.ndarray):
        return admm(prox_fs, prox_g, lambda u : np.dot(K,u),
            lambda v : np.dot(K.T,v), x0, maxiter=maxiter, theta=theta,
            sigma=sigma, tau=tau, full_output=full_output, retall=retall,
            callback=callback)
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