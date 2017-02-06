
"""
This is just a quick script to cythonize the neural network code,
to improve performance. The code is, by no means, well
optimized otherwise though.

To run this script:

python setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize


setup(
    ext_modules = cythonize(["mnist_loader.pyx", "network.pyx", "network2.pyx"])
)
