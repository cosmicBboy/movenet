from setuptools import setup


EXCLUDE_REQUIRES = {
    "dask",
    "joblib",
    "tensorboard",
    "tensorflow",
    "typing-extensions==3.7.4.3",
}


with open("requirements.txt") as f:
    requires = [
        x for x in f.readlines() if x.split("==")[0] in EXCLUDE_REQUIRES
    ]


setup(
    name="movenet",
    version="0.0.0+dev0",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    packages=["movenet"],
    install_requires=requires,
)
