from distutils.core import setup, Extension
import sys

histo_module = Extension('histogram.histogram_c',
                         sources=['histogram/histogram_c.c'])


if sys.argv[1] == 'install':

    sys.argv += ['build_ext']


REQUIREMENT_FILE = 'requirements.txt'
with open(REQUIREMENT_FILE) as f:
    content = f.readlines()

requires = [x.strip() for x in content]

setup(
    name='histogram',
    version='0.3.3',
    packages=['histogram'],
    url='https://github.com/calispac/histogram',
    license='GNU GPL 3.0',
    author='Cyril Alispach',
    author_email='cyril.alispach@gmail.com',
    long_description=open('README.md').read(),
    description='A package for Histogramming',
    requires=requires,
    ext_modules=[histo_module],
)
