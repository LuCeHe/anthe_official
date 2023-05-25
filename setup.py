from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

__version__ = '1.0.0'

setup(
    name='anthe_official',
    version=__version__,
    license='Apache License',
    author='Luca Herranz-Celotti, Ermal Rrapaj',
    author_email='luca.herrtti@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='Anthe improves performance of Transformers with less parameters.',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
