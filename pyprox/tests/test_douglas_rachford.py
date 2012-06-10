from __future__ import division
from numpy.testing import assert_array_almost_equal

import numpy as np
from pyprox.algorithms import douglas_rachford
from pyprox.operators import soft_thresholding


def test_dr_virtual_zero():
    # Virtual 0-prox
    prox_f = lambda u, la: 0 * 0
    prox_g = lambda u, la: u * 0

    # observations of size (5,1)
    y = np.zeros((5, 1))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)


def test_dr_zero():
    # Prox of F, G = 0
    prox_f = lambda u, la: u
    prox_g = lambda u, la: u

    # observations of size (5,1)
    y = np.zeros((5, 1))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    x_rec = douglas_rachford(prox_f, prox_g, y)
    assert_array_almost_equal(y, x_rec)


def test_dr_l1_cs():
    # Dimension of the problem
    n = 200
    p = n // 4

    # Matrix and observations
    A = np.random.randn(p, n)
    # Use a very sparse vector for the test
    x = np.zeros((n, 1))
    x[1, :] = 1
    y = np.dot(A, x)

    # operator callbacks
    prox_f = soft_thresholding
    prox_g = lambda x, tau: x + np.dot(A.T, np.linalg.solve(np.dot(A, A.T),
        y - np.dot(A, x)))

    x_rec = douglas_rachford(prox_f, prox_g, np.zeros((n, 1)), maxiter=1000)
    assert_array_almost_equal(x, x_rec)
