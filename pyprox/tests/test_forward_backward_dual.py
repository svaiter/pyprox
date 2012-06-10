from __future__ import division
from numpy.testing import assert_array_almost_equal

import numpy as np
from pyprox.algorithms import forward_backward_dual


def test_fb_dual_virtual_zero():
    # Virtual 0-prox
    grad_fs = lambda u: u * 0
    prox_gs = lambda u, la: u * 0

    # ndarray
    k_nd = np.zeros((5, 5))
    # explicit
    k_exp = lambda u: 0 * u
    k_exp.T = lambda u: 0 * u

    # observations of size (5,1)
    y = np.zeros((5, 1))
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_nd, y, 1)
    assert_array_almost_equal(y, x_rec)
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_exp, y, 1)
    assert_array_almost_equal(y, x_rec)

    # observations of size (5,2)
    y = np.zeros((5, 2))
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_nd, y, 1)
    assert_array_almost_equal(y, x_rec)
    x_rec = forward_backward_dual(grad_fs, prox_gs, k_exp, y, 1)
    assert_array_almost_equal(y, x_rec)
