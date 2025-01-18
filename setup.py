from setuptools import setup, find_packages

setup(
    name="lucass_tools",
    version="0.1",
    packages=find_packages(),
    install_requires=[
    "matplotlib==3.10.0",
    "numpy==2.2.1",
    "scipy==1.15.1",
],  # Add dependencies here
)
