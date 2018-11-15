#!/usr/local/bin/python
# coding: utf-8
from setuptools import setup

AUTHOR = dict(
    name='Tomas Pereira de Vasconcelos',
    email='tomasvasconcelos1@gmail.com',
    )

LICENCE = open("LICENSE").read()

LONG_DESCRIPTION = 'mdsea is a stand-alone Python molecular dynamics ' \
                   'library equipped with a flexible simulation engine ' \
                   'and multiple analysis tools, including integrated ' \
                   'beautiful visualization in 1, 2, and 3 dimensions.'

setup(name='mdsea',
      version='0.01',

      description='Molecular Dynamics Library',
      long_description=LONG_DESCRIPTION,

      author=AUTHOR['name'],
      author_email=AUTHOR['email'],
      maintainer=AUTHOR['name'],
      maintainer_email=AUTHOR['email'],

      url='https://github.com/TPVasconcelos/mdsea',
      # download_url='TODO',

      packages=['mdsea',
                'mdsea.constants',
                'mdsea.vis'],
      # py_modules=['mdsea', 'mdsea.vis', 'mdsea.constants'],
      # scripts='TODO',
      # ext_modules='TODO',

      install_requires=['numpy', 'scipy', 'h5py', 'matplotlib'],

      # classifiers='TODO',
      # distclass='TODO',

      # script_name='TODO',
      # script_args='TODO',
      # options='TODO',

      license=LICENCE,
      # keywords='TODO',
      # platforms='TODO',

      # cmdclass='TODO',
      # data_files='TODO',
      # package_dir='TODO',
      )
