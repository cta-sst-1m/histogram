from distutils.core import setup, Extension
import sys

histo_module = Extension('histogram.histogram_c',
                         sources=['histogram/histogram_c.c'])


if sys.argv[1] == 'install':

    sys.argv += ['build_ext']

setup(
    name='histogram',
    version='0.3.1',
    packages=['histogram'],
    url='https://github.com/calispac/histogram',
    license='GNU GPL 3.0',
    author='Cyril Alispach',
    author_email='cyril.alispach@gmail.com',
    long_description=open('README.md').read(),
    description='A package for Histogramming',
    requires=['Cython', 'numpy', 'matplotlib', 'scipy', 'iminuit'],
    ext_modules=[histo_module],
)