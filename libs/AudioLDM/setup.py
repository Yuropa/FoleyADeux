# pip install -e ./
from setuptools import setup, find_packages

setup(
    name='AudioLDM',
    version='0.1',
    package_dir={'': './'},
    packages=find_packages( where = './'),
    install_requires=[
        'pyyaml',
        'einops',
        'soundfile',
        'h5py',
        'tqdm',
        'torchlibrosa',
        'transformers',
        'ftfy',
        'regex',
        'braceexpand',
        'pandas',
        'webdataset',
        'wget',
        'timm',
        'matplotlib',
        'taming-transformers'
    ],
    # additional metadata about your project
    author='Jaekwon Im',
    author_email='jakeoneijk@kaist.ac.kr',
    description='',
    license='',
    keywords='',
)