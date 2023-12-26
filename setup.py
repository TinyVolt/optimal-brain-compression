import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="optimal-brain-compression",
    py_modules=["optimal_brain_compression"],
    version="0.1",
    description="Unofficial implementation of the Optimal Brain Compression algorithm",
    author="Vinay Sisodia",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["torch", "pydantic"],
    include_package_data=True,
)