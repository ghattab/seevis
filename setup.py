#!/usr/bin/env python
import os
import io
import re
import shutil
import sys
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


readme = open("README.md").read()


requirements = [
    "numpy >= 1.14.3",
    "pillow >= 5.1.0",
    "matplotlib >= 2.2.2",
    "opencv-contrib-python >= 3.4.1",
    "pyqtgraph >= 0.10.0",
    "trackpy >= 0.4.1",
    "pims >= 0.4.1",
    "pandas >= 0.23.0",
    "scipy >= 1.1.0",
    "Pillow >= 5.1.0"
]

setup(
    # Metadata
    name="SeeVIS",
    version="1.0",
    author="Georges Hattab",
    author_email="ghattab@cebitec.uni-bielefeld.de",
    url="https://github.com/ghattab/seevis",
    description="Segmentation free approach visualization of cell colony movies \
      in a space-time feature cube",
    long_description=readme,
    license="MIT",
    install_requires=requirements,
)
