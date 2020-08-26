import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="alcokit",
    version="0.0.9",
    author="Antoine Daurat",
    author_email="antoinedaurat@gmail.com",
    description="package to do MIR and learn generative models on top of librosa and pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antoinedaurat/alcokit",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)