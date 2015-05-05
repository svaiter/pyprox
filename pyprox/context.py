class Context(object):
    """Blabla
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
    def __init__(self):
        self.maxiter = 100
        self.full_output = 0
        self.retall = 0
        self.callback = None

    def execute(self, values, step):
        allvecs = [values[0]]
        fx = []
        for i in range(self.maxiter):
            values = step(*values)
            x = values[0]
            if self.full_output:
                pass
            if self.retall:
                allvecs.append(x)
            if self.callback:
                fx.append(callback(x))
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
