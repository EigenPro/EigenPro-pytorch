from setuptools import setup, find_packages
import eigenpro2

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='eigenpro2',
    version=eigenpro2.__version__,
    author='Siyuan Ma, Adityanarayanan Radhakrishnan, Parthe Pandit',
    author_email='parthe1292@gmail.com',
    description='Fast solver for Kernel Regression using GPUs with linear space and time complexity',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/EigenPro/EigenPro-pytorch/tree/pytorch',
    project_urls = {
        "Bug Tracker": "https://github.com/EigenPro/EigenPro-pytorch/issues"
    },
    license='Apache-2.0 license',
    packages=find_packages(),
    install_requires=[],
)
