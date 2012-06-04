from __future__ import division
from numpy.testing import assert_array_almost_equal

import numpy as np
from pyprox.algorithms import admm

def test_admm_virtual_zero():
    # Virtual 0-prox
    prox_fs = lambda u, la : u * 0
    prox_g = lambda u, la : u * 0

    # ndarray
    k_nd = np.zeros((5,5))
    # explicit
    k_exp = lambda u : 0 * u
    k_exp.T = lambda u : 0 * u

    # observations of size (5,1)
    y = np.zeros((5,1))
    xRec = admm(prox_fs, prox_g, k_nd, y)
    assert_array_almost_equal(y, xRec)
    xRec = admm(prox_fs, prox_g, k_exp, y)
    assert_array_almost_equal(y, xRec)

    # observations of size (5,2)
    y = np.zeros((5,2))
    xRec = admm(prox_fs, prox_g, k_nd, y)
    assert_array_almost_equal(y, xRec)
    xRec = admm(prox_fs, prox_g, k_exp, y)
    assert_array_almost_equal(y, xRec)