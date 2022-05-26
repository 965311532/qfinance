# Simple setup file for qfinance
from distutils.core import setup

setup(
    name="qfinance",
    version="0.1.5",
    description="A simple financial database management package",
    url="https://github.com/965311532/qfinance",
    author="Gabriele Armento",
    author_email="gabriele.armento@gmail.com",
    license="MIT",
    install_requires=[
        "pandas",
        "fastparquet",
    ],
    py_modules=["qfinance"],
)
