#!/usr/bin/env python
import setuptools 

__version__ = '0.1'

setuptools.setup(name = 'astrologs',
      version = __version__,
      author='ChangHoon Hahn', 
      author_email='hahn.changhoon@gmail.com', 
      description = 'accessing astro catalogs',
      install_requires = ['numpy', 'h5py'], 
      provides = ['astrologs'],
      packages = setuptools.find_packages(),
      python_requires='>3.5.2'
      )
