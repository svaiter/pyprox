"""
==============================================
Total variation denoising using Chambolle Pock
==============================================

Test the use of ADMM for a denoising scenario with anistropic TV
"""
# Author: Samuel Vaiter <samuel.vaiter@gmail.com>
from __future__ import print_function, division

print(__doc__)

import time
import numpy as np
from scipy import misc
import scipy.linalg as lin
import pylab as plt

from pyprox import dual_prox, admm
from pyprox.operators import soft_thresholding
from pyprox.context import Context

# Load image, downsample and convert to a float
im = misc.face()[:,:,0]
im = misc.imresize(im, (256, 256)).astype(np.float) / 255.

n = im.shape[0]

# Noisy observations
sigma = 0.06
y = im + sigma * np.random.randn(n, n)

# Regularization parameter
alpha = 0.1

# Gradient and divergence with periodic boundaries


def gradient(x):
    g = np.zeros((x.shape[0], x.shape[1], 2))
    g[:, :, 0] = np.roll(x, -1, axis=0) - x
    g[:, :, 1] = np.roll(x, -1, axis=1) - x
    return g


def divergence(p):
    px = p[:, :, 0]
    py = p[:, :, 1]
    resx = px - np.roll(px, 1, axis=0)
    resy = py - np.roll(py, 1, axis=1)
    return -(resx + resy)

# Minimization of F(K*x) + G(x)
K = gradient
K.T = divergence
amp = lambda u: np.sqrt(np.sum(u ** 2, axis=2))
F = lambda u: alpha * np.sum(amp(u))
G = lambda x: 1 / 2 * lin.norm(y - x, 'fro') ** 2

# Proximity operators
normalize = lambda u: u / np.tile(
    (np.maximum(amp(u), 1e-10))[:, :, np.newaxis],
    (1, 1, 2))
prox_f = lambda u, tau: np.tile(
    soft_thresholding(amp(u), alpha * tau)[:, :, np.newaxis],
    (1, 1, 2)) * normalize(u)
prox_fs = dual_prox(prox_f)
prox_g = lambda x, tau: (x + tau * y) / (1 + tau)


# context
ctx = Context(full_output=True, maxiter=300)
ctx.callback = lambda x: G(x) + F(K(x))

t1 = time.time()
x_rec, cx = admm(prox_fs, prox_g, K, y, context=ctx)
t2 = time.time()
print("Performed 300 iterations in " + str(t2 - t1) + " seconds.")


plt.subplot(221)
plt.imshow(im, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(222)
plt.imshow(y, cmap='gray')
plt.title('Noisy')
plt.axis('off')
plt.subplot(223)
plt.imshow(x_rec, cmap='gray')
plt.title('TV Regularization')
plt.axis('off')
plt.subplot(224)
fplot = plt.plot(cx)
plt.title('Objective versus iterations')
plt.show()
