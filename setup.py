#!/usr/bin/env python

from setuptools import setup, find_packages


with open('README.rst') as file:
    long_description = file.read()

setup(name='rndensity',
      version='1.0',
      description=('Risk-neutral density-based option pricing'),
      long_description=long_description,
      author='Stanislav Khrapov',
      license='MIT',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/rndensity',
      py_modules=['rndensity'],
      packages=find_packages(),
      keywords=['forecasting', 'derivatives',
                'implied volatility surface',
                'no-arbitrage restrictions',
                'pricing', 'options', 'model',
                'term structure'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
      ],
      )
