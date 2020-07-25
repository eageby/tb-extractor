from setuptools import find_packages, setup

setup(
    name="tb-extractor",
    version="0.1",
    packages=find_packages(),
    entry_points="""[console_scripts]
        tb-extractor=tb_extractor.cli:main
    """,
    install_requires=["click", "tensorflow", "tensorboard", "pandas", "pathlib"],
)
