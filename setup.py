#!/usrbin/env python
from setuptools import setup

description = "BBPower  the C_ell-based pipeline for BB"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(name="bbpower",
      version="0.0.0",
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simonsobs/BBPower",
      author="Kevin Wolz",
      author_email="kevin.wolz93@gmail.com",
      install_requires=requirements,
      packages=['bbpower'],
)