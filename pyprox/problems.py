"""
Classic problems in sparsity
"""
from __future__ import division
import numpy as np
from pyprox.algorithms import forward_backward
from pyprox.disciplined import LeastSquareDataFit, L1Norm

def probe_dimensions(Phi, y):
    if callable(Phi) and hasattr(Phi,'T'):
        x = Phi.T(y)
    elif isinstance(np.ndarray):
        x = np.dot(Phi.T, y)
    else:
        raise Exception('Cannot determine the transpose of the operator.')
    return (x.shape[0],y.shape[0])

class Lasso():
    """Lasso
    """

    valid_solvers = ['fb']

    def __init__(self, la=1.0, solver='fb', maxiter=1000, warm_start=None):
        """
        init
        """
        self.la = la
        if solver in valid_solvers:
            self.solver = solver
        else:
            raise Exception('Lasso initialization: '
                            + solver + 'is not a valid solver')
        self.maxiter = maxiter
        self.warm_start = warm_start

    def solve(self, Phi, y):
        """
        solve
        """
        self.datafit = LeastSquareDataFit(Phi, y)
        self.regularization = L1Norm(la=self.la)
        (q,n) = probe_dimensions(Phi, y)
        if self.warm_start:
            init_vector = self.lastsol
        else:
            init_vector = np.zeros((n,1))
        self.lastsol = forward_backward(
            self.regularization.prox,
            self.datafit.grad,
            init_vector, operator_norm(Phi, n), maxiter=maxiter)