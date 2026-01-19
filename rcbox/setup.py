from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("rmds_cythoned", ["rmds_cythoned.pyx"]),
    Extension("rtdoa_cythoned", ["rtdoa_cythoned.pyx"]),
    Extension("toa_processor", ["toa_processor.pyx"])
]

setup(
    name="rcbox",
    version="0.1",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)

# Run this line for compilation:
# python setup.py build_ext --inplace

# Run this line to clean generated binaries and C
# rm -r *.c *.so build

