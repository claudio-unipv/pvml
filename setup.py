import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pvml",
    version="0.3.0",
    author="Claudio Cusano",
    author_email="claudio.cusano@unipv.it",
    description="A small and simple machine learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/claudio-unipv/pvml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.13.0",
        "matplotlib>=2.1.0",
        "Pillow>=5.1.0"
    ],
    python_requires='>=3.6',
)


# python3 setup.py sdist bdist_wheel
# python3 -m twine upload dist/*
