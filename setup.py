from setuptools import setup


with open("requirements.txt") as f:
    requires = list(x for x in f.readlines())

with open("dev-requirements.txt") as f:
    requires = list(x for x in f.readlines())

setup(
    name="movenet",
    version="0.0.0+dev0",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    packages=["movenet"],
    install_requires=requires,
)
