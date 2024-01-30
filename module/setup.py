from setuptools import setup

setup(
    name='tes',
    version='0.1.0',
    packages=['tes'],
    include_package_data=True,
    package_data={'tes':['reference.dat']},
    description='A package for performing PCA + KMeans on TES data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Artem Knyazev, Isaac Sarnoff, Panos Economou, Tengiz Ibrayev',
    author_email='tengiz.ibrayev@nyu.edu',
    url='https://github.com/nyuad-astroparticle/TES-analysis/',
    # Add additional fields as necessary
)
