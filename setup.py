#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

export_packages = find_packages(where='./src', include=('lm1b_wrapper',))

setup(name='lm_1b_wrapper',
      version='0.1.0',
      description='Wrapper for Google\'s Billion Word Language Model',
      packages=['lm1b_wrapper'],
      package_dir = {'': 'src'},
      # 3.6 and up, but not Python 4
      python_requires='~=3.6',
      install_requires=[
          # we deliberately leave out tensorflow so we don't mess with your existing
          # tensorflow install
          'attrs>=17.3.0',
          'vistautils>=0.4.0'
      ],
      scripts=['scripts/dump_sentence_probs.py']
      )
