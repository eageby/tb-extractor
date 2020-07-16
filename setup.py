from setuptools import setup, find_packages

setup(name='tb-extractor', version='0.1', packages=find_packages(),
       entry_points='''[console_scripts]
        tb-extractor=extractor:main
    ''',
    install_requires=[
        'click',
        'tensorboard',
        'pandas',
        'pathlib')
