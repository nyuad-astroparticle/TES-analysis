from setuptools import setup, find_packages

setup(
    name='tes',
    version='0.1',
    packages=find_packages(),
    description='A package for performing PCA + KMeans on TES data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Artem Knyazev, Isaac Sarnoff, Panos Economou, Tengiz Ibrayev',
    author_email='tengiz.ibrayev@nyu.edu',
    url='https://github.com/nyuad-astroparticle/TES-analysis/',
    # Add additional fields as necessary
)
