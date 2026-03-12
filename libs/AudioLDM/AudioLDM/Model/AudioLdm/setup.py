# pip install -e ./
from setuptools import setup, find_packages

setup(
    name='audioldm_train',
    version='0.1',
    package_dir={'': './'},
    packages=find_packages( where = './'),
    install_requires=[],
    # additional metadata about your project
    author='',
    author_email='',
    description='',
    license='',
    keywords='',
)