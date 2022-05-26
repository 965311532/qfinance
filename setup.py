# Simple setup file for qfinance
from setuptools import setup, find_packages

setup(
    name="qfinance",
    version="0.2.5",
    description="A simple financial database management package",
    url="https://github.com/965311532/qfinance",
    author="Gabriele Armento",
    author_email="gabriele.armento@gmail.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "fastparquet",
    ],
)
