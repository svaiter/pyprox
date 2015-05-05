"""
The :mod:`pyprox.context` module includes the definition of severals contexts.
"""
# Author: Samuel Vaiter <samuel.vaiter@gmail.com>


def maxiter_criterion(values, iteration, allvecs, fx, maxiter=100):
    return iteration < maxiter


class Context(object):
    """A Context object
    maxiter : int, optional
        maximum number of iterations.
    full_output : bool, optional
        non-zero to return all optional outputs.
    retall : bool, optional
        Return a list of results at each iteration if non-zero.
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as callback(xk), where xk is the current parameter vector.
    """
    def __init__(self, criterion=maxiter_criterion,
                 full_output=False, retall=False, callback=None, **kwargs):
        self.criterion = criterion
        if 'maxiter' in kwargs:
            self.criterion = lambda v, i, a, f: maxiter_criterion(v, i, a, f, maxiter=kwargs['maxiter'])
        self.full_output = full_output
        self.retall = retall
        self.callback = callback

    def execute(self, values, step):
        allvecs = [values[0]]
        fx = []
        iteration = 0
        while self.criterion(values, iteration, allvecs, fx):
            values = step(*values)
            x = values[0]
            if self.full_output:
                pass
            if self.retall:
                allvecs.append(x)
            if self.callback:
                fx.append(self.callback(x))
            iteration += 1
        return self._output_helper(x, fx, allvecs)

    def _output_helper(self, x, fx, allvecs):
        if self.full_output:
            retlist = x, fx
            if self.retall:
                retlist += (allvecs,)
        else:
            retlist = x
            if self.retall:
                retlist = (x, allvecs)

        return retlist


defaultContext = Context()
