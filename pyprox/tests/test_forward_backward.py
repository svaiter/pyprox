from __future__ import division
from numpy.testing import assert_array_almost_equal

import numpy as np
from pyprox.algorithms import forward_backward
from pyprox.utils import soft_thresholding

methods = ['fb', 'fista', 'nesterov']

def test_fb_virtual_zero():
    # Virtual 0-prox
    prox_f = lambda u, la : 0 * 0
    grad_g = lambda u : u * 0

    # observations of size (5,1)
    y = np.zeros((5,1))
    for method in methods:
        xRec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, xRec)

    # observations of size (5,2)
    y = np.zeros((5,2))
    for method in methods:
        xRec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, xRec)

def test_fb_zero():
    prox_f = lambda u, la : u
    grad_g = lambda u : u * 0

    # observations of size (5,1)
    y = np.zeros((5,1))
    for method in methods:
        xRec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, xRec)

    # observations of size (5,2)
    y = np.zeros((5,2))
    for method in methods:
        xRec = forward_backward(prox_f, grad_g, y, 1, method=method)
        assert_array_almost_equal(y, xRec)

def test_fb_l1_denoising():
    n = 1000
    # Use a very sparse vector for the test
    x = np.zeros((n,1))
    x[1,:] = 100
    y = x + 0.06 * np.random.randn(n,1)

    la = 0.2
    prox_f = lambda x,tau: soft_thresholding(x, la*tau)
    grad_g = lambda x: x - y

    for method in methods:
        xRec = forward_backward(prox_f, grad_g, y, 1, method=method)
        #TODO ugly test to change
        assert_array_almost_equal(x, xRec, decimal=0)