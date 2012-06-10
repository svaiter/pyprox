"""
==============================================
Total variation denoising using Chambolle Pock
==============================================
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>
from __future__ import division
from pyprox import dual_prox
from pyprox.operators import soft_thresholding

print __doc__

import time

import numpy as np
import scipy as sp
import pylab as pl

import pyprox as pp

# Load image, downsample and convert to a float
im = sp.misc.lena()
im = sp.misc.imresize(im, (256, 256)).astype(np.float) / 255.

n = im.shape[0]

# Noisy observations
sigma = 0.06
y = im + sigma * np.random.randn(n,n)

# Regularization parameter
alpha = 0.1

# Gradient and divergence with periodic boundaries
def gradient(x):
    g = np.zeros((x.shape[0],x.shape[1],2))
    g[:,:,0] = np.roll(x,-1,axis=0) - x
    g[:,:,1] = np.roll(x,-1,axis=1) - x
    return g

def divergence(p):
    px = p[:,:,0]
    py = p[:,:,1]
    resx = px - np.roll(px,1,axis=0)
    resy = py - np.roll(py,1,axis=1)
    return -(resx + resy)

# Minimization of F(K*x) + G(x)
K = gradient
K.T = divergence
amp = lambda u : np.sqrt(np.sum(u ** 2,axis=2))
F = lambda u : alpha * np.sum(amp(u))
G = lambda x : 1/2 * np.linalg.norm(y-x,'fro') ** 2

# Proximity operators
normalize = lambda u : u/np.tile(
    (np.maximum(amp(u), 1e-10))[:,:,np.newaxis],
    (1,1,2))
proxF = lambda u,tau : np.tile(
    soft_thresholding(amp(u), alpha*tau)[:,:,np.newaxis],
    (1,1,2) )* normalize(u)
proxFS = dual_prox(proxF)
proxG = lambda x,tau : (x + tau*y) / (1+tau)

callback = lambda x : G(x) + F(K(x))

t1 = time.time()
xRec, cx = pp.admm(proxFS, proxG, K, y,
         maxiter=300, full_output=1, callback=callback)
t2 = time.time()
print "Performed 300 iterations in " + str(t2-t1) + " seconds."


pl.subplot(221)
imgplot = pl.imshow(im)
imgplot.set_cmap('gray')
pl.title('Original')
pl.axis('off')
pl.subplot(222)
imgplot = pl.imshow(y)
imgplot.set_cmap('gray')
pl.title('Noisy')
pl.axis('off')
pl.subplot(223)
imgplot = pl.imshow(xRec)
imgplot.set_cmap('gray')
pl.title('TV Regularization')
pl.axis('off')
pl.subplot(224)
fplot = pl.plot(cx)
pl.title('Objective versus iterations')
pl.show()