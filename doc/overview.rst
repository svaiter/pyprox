====================
Overview of `pyprox`
====================

`pyprox` is a free Open Source proximal algorithm library for Python
programming language.

Simple example
--------------

`pyprox` aims to bring simplicity to software using proximal algorithms.
To start using it, install the package, open the the Python interactive
shell and type:

    >>> import pyprox as pp
    >>> import numpy as np
    >>> x0 = np.zeros((100,1))
    >>> y = x0 + 0.01 * np.random.randn(100,1)
    >>> ProxF = pp.soft_thresholding
    >>> GradG = lambda x : x - y
    >>> L = np.linalg.norm(A, 2) ** 2
    >>> xrec = forward_backward(ProxF, GradG, y, L)

Requirements
------------
`pyprox` requires `Python <http://python.org/>` 2.5-2.7 installed.
The only external requirement is a recent version of
`NumPy <http://numpy.scipy.org/>`_ numeric array module.

Download
--------

The most recent *development* version can be found on Github at:

  * Github - https://github.com/svaiter/pyprox

Latest release is available for download from the Python Package Index at
http://pypi.python.org/pypi/pyprox/.

License
-------

`pyprox` is a free Open Source software available under the New BSD license
terms.

Contact
-------

Post your suggestions and guestions to `pyprox discussions
group <http://groups.google.com/group/pyprox>`_ (pyprox@googlegroups.com).

Comments, bug reports and fixes are welcome through `Github <https://github
.com/svaiter/pyprox>`_
