from distutils.core import setup, Extension
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
C_HISTOGRAM_FILE = os.path.join(dir_path, 'histogram/histogram_c.c')
REQUIREMENT_FILE = os.path.join(dir_path, 'requirements.txt')
README_FILE = os.path.join(dir_path, 'README.md')

histo_module = Extension('histogram.histogram_c',
                         sources=[C_HISTOGRAM_FILE])


if sys.argv[1] == 'install':

    sys.argv += ['build_ext']


with open(REQUIREMENT_FILE) as f:
    content = f.readlines()

requires = [x.strip() for x in content]
setup(
    name='histogram',
    version='0.5.0',
    packages=['histogram'],
    url='https://github.com/calispac/histogram',
    license='GNU GPL 3.0',
    author='Cyril Alispach',
    author_email='cyril.alispach@gmail.com',
    long_description=open(README_FILE).read(),
    description='A package for Histogramming',
    requires=requires,
    ext_modules=[histo_module],
)
