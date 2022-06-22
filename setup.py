# Simple setup file for qfinance
from setuptools import setup

setup(
    name="qfinance",
    version="0.3.5",
    description="A simple financial database management package",
    url="https://github.com/965311532/qfinance",
    author="Gabriele Armento",
    author_email="gabriele.armento@gmail.com",
    license="MIT",
    py_modules=["qfinance"],
    install_requires=[
        "pandas",
        "fastparquet",
    ],
)
