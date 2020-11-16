# -*- coding: utf-8 -*-

import setuptools
import distutils.text_file
from pathlib import Path
from typing import List


with open("README.md", "r") as fh:
    long_description = fh.read()


def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    # Ref: https://stackoverflow.com/a/42033122/
    return distutils.text_file.TextFile(filename=str(Path(__file__).with_name(filename))).readlines()


setuptools.setup(
    name="lcksvd",
    version="0.0.1",
    author="Giussepi Lopez",
    author_email="giussepexy@gmail.com",
    description="LC-KSVD (1 & 2) python implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giussepi/LC-KSVD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: APACHE License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=_parse_requirements('requirements.txt'),
)
