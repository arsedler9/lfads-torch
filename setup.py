# Runs the installation. See the following for more detail:
# https://docs.python.org/3/distutils/setupscript.html

from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="lfads_torch",
    author="Andrew Sedler",
    author_email="arsedler9@gmail.com",
    description="A PyTorch implementation of "
    "Latent Factor Analysis via Dynamical Systems (LFADS)",
    url="https://github.com/arsedler9/lfads-torch",
    install_requires=requirements,
    packages=find_packages(),
)
