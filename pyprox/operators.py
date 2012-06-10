"""
Proximal operators
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>

from __future__ import division
import numpy as np


def soft_thresholding(x, gamma):
    return np.maximum(0, 1 - gamma / np.maximum(np.abs(x), 1E-10)) * x


def dual_prox(prox):
    return lambda u, sigma: u - sigma * prox(u / sigma, 1 / sigma)
