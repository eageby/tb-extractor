from setuptools import find_packages, setup

setup(
    name="tb-extractor",
    version="0.1",
    packages=find_packages(),
    entry_points="""[console_scripts]
        tb-extractor=extractor.cli:main
    """,
    install_requires=["click", "tensorboard", "pandas", "pathlib"],
)
