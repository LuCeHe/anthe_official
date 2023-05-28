from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    'h5py>=3.1.0',
    'numpy>=1.23.3',
    'pandas>=1.4.0',
    'scikit_learn>=1.1.3',
    'sentencepiece>=0.1.97',
    'tensorflow>=2.10.0',
    'tqdm>=4.64.0',
    'pyaromatics==0.0.2',
]
__version__ = '1.0.1'

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
