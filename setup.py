from setuptools import setup


setup(
    name="movenet",
    version="0.0.0+dev0",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    packages=["movenet"],
    install_requires=[
        "av",
        "dask",
        "dataclasses==0.6",
        "dataclasses-json==0.5.2",
        "joblib",
        "librosa==0.8.1",
        "numpy==1.20.3",
        "opencv-python",
        "opencv-python-headless==4.5.3.56",
        "tensorboard",
        "tensorflow",
        "tqdm",
        "torch==1.9.0",
        "torchvision==0.10.0",
        "torchaudio==0.9.0",
        "torchtyping==0.1.2",
        "typeguard",
        "typing-extensions==3.7.4.3",
        "ipdb==0.13.7",
        "pytorchvideo==0.1.2",
        "wandb",
    ],
)
