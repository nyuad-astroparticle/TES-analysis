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
    install_requires=[
        'numpy==1.23.5',
        'scipy==1.10.1',
        'pandas==2.0.3',
        'scikit-learn==1.3.0',
        'ipython==8.12.2',
        'notebook==7.0.6',
        'matplotlib==3.7.2',
        'seaborn==0.12.2',
        'cupy==8.3.0',
        'tqdm==4.65.0',
        'ipympl==0.9.3',
        'mplcursors==0.5.3',
        'pyperclip==1.8.2'
    ]
    # Add additional fields as necessary
)
