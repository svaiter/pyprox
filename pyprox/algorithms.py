"""
The :mod:`pyprox.algorithms` module includes the proximal schemes of pyprox.
"""
# Author: Samuel Vaiter <samuel.vaiter@gmail.com>

from __future__ import division
import numpy as np
import math
from .utils import operator_norm
from .context import Context, defaultContext


def douglas_rachford(prox_f, prox_g, x0,
                     mu=1, gamma=1, context=defaultContext):
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
    mu : float, optional
    gamma : float, optional
    context: Context
        the context (default to defaultContext)

    Returns
    -------
    x_rec: ndarray
    fx: list

    References
    ----------
    Proximal Splitting Methods in Signal Processing,
    Patrick L. Combettes and Jean-Christophe Pesquet, in:
    Fixed-Point Algorithms for Inverse Problems in Science and Engineering,
    New York: Springer-Verlag, 2010.
    """
    def rProx_f(x, tau):
        return 2 * prox_f(x, tau) - x

    def rProx_g(x, tau):
        return 2 * prox_g(x, tau) - x

    x = x0.copy()
    y = x0.copy()

    def step(x, y):
        y = (1 - mu / 2) * y + mu / 2 * rProx_f(rProx_g(y, gamma), gamma)
        x = prox_g(y, gamma)
        return [x, y]

    return context.execute([x, y], step)


def forward_backward(prox_f, grad_g, x0, L,
                     method='fb', fbdamping=1.8, context=defaultContext):
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
    method : string, optional,
        can be 'fb', 'fista' or 'nesterov'
    fbdamping : float, optional
    context: Context
        the context (default to defaultContext)

    Returns
    -------
    x_rec: ndarray
    fx: list

    References
    ----------
    P. L. Combettes and V. R. Wajs, Signal recovery by proximal
    forward-backward splitting,
    Multiscale Model. Simul., 4 (2005), pp. 1168-1200
    """
    # FISTA
    t = 1

    # Nesterov
    tt = 2 / L
    gg = 0
    A = 0

    y = x0.copy()
    x = x0.copy()

    def step_fb(x):
        x = prox_f(x - fbdamping / L * grad_g(x), fbdamping / L)
        return [x]

    def step_fista(x, y, t):
        xnew = prox_f(y - 1 / L * grad_g(y), 1 / L)
        tnew = (1 + math.sqrt(1 + 4 * t ** 2)) / 2
        y = xnew + (t - 1) / tnew * (xnew - x)
        x = xnew
        t = tnew
        return [x, y, t]

    def step_nesterov(x, tt, gg, A):
        a = (tt + math.sqrt(tt ** 2 + 4 * tt * A)) / 2
        v = prox_f(x0 - gg, A)
        z = (A * x + a * v) / (A + a)
        x = prox_f(z - 1 / L * grad_g(z), 1 / L)
        gg += a * grad_g(x)
        A += a
        return [x, tt, gg, A]

    if method == "fb":
        return context.execute([x], step_fb)
    elif method == "fista":
        return context.execute([x, y, t], step_fista)
    elif method == "nesterov":
        return context.execute([x, tt, gg, A], step_nesterov)
    else:
        raise Exception('ex a def in fb')


def forward_backward_dual(grad_fs, prox_gs, K, x0, L,
                          method='fb', fbdamping=1.8, context=defaultContext):
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
    method : string, optional,
        can be 'fb', 'fista' or 'nesterov'
    fbdamping : float, optional
    context: Context
        the context (default to defaultContext)

    Returns
    -------
    x_rec: ndarray
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
    if isinstance(K, np.ndarray):
        op = lambda u: np.dot(K, u)
        op.T = lambda u: np.dot(K.T, u)
        return forward_backward_dual(
            grad_fs, prox_gs, op, x0, L,
            method=method, fbdamping=fbdamping,
            context=context)

    if context.callback:
        old_callback = context.callback
        context.callback = lambda u: old_callback(grad_fs(-K.T(u)))
    new_grad = lambda u: - K(grad_fs(-K.T(u)))
    u0 = K(x0)
    res = forward_backward(
        prox_gs, new_grad, u0, L,
        method=method, fbdamping=fbdamping,
        context=context)

    try:
        res[0] = grad_fs(-K.T(res[0]))
    except:
        res = grad_fs(-K.T(res))
    return res


def admm(prox_fs, prox_g, K, x0,
         theta=1, sigma=None, tau=None, context=defaultContext):
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
    theta : float, optional
    sigma : float, optional
        parameters of the method.
        They should satisfy sigma * tay * norm(K)^2 < 1
    context: Context
        the context (default to defaultContext)

    Returns
    -------
    x_rec: ndarray
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
        op = lambda u: np.dot(K, u)
        op.T = lambda u: np.dot(K.T, u)
        return admm(prox_fs, prox_g, op, x0, theta=theta,
                    sigma=sigma, tau=tau, context=context)
    if not(sigma and tau):
        L = operator_norm(
            lambda x: K.T(K(x)),
            np.random.randn(x0.shape[0], 1)
        )
        sigma = 10.0
        if sigma * L > 1e-10:
            tau = .9 / (sigma * L)
        else:
            tau = 0.0

    x = x0.copy()
    x1 = x0.copy()
    xold = x0.copy()
    y = K(x)

    def step(x, x1, xold, y):
        xold = x.copy()
        y = prox_fs(y + sigma * K(x1), sigma)
        x = prox_g(x - tau * K.T(y), tau)
        x1 = x + theta * (x - xold)
        return [x, x1, xold, y]

    return context.execute([x, x1, xold, y], step)
