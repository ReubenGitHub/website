from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(name="custom_metric",
      ext_modules=[Extension('custom_metric',
                   sources=['custom_metric.pyx'],
                   extra_compile_args=['-O3', '-march=native', '-ffast-math', '-flto'])
                  ],
      cmdclass = {'build_ext': build_ext},
      include_dirs=[numpy.get_include()])