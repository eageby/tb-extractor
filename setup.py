from setuptools import find_packages, setup

setup(
    name="tb-extractor",
    version="0.1",
    py_modules=["extractor"],
    entry_points="""[console_scripts]
        tb-extractor=extractor:main
    """,
    install_requires=["click", "tensorboard", "pandas", "pathlib"],
)
