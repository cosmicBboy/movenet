from setuptools import setup


setup(
    name="movenet",
    version="0.0.0+dev0",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    packages=["movenet"],
    install_requires=[
        "av",
        "dataclasses_json",
        "torch",
        "torchvision",
        "torchaudio",
        "wandb",
    ],
)
