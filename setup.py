import logging
from setuptools import setup, find_packages


PACKAGE_NAME = "monolayout"
VERSION = "0.1.0"
DESCRIPTION = "MonoLayout: Amodal scene layout from a single image"
URL = "https://hbutsuak95.github.io/monolayout/"
AUTHOR = "Kaustubh Mani"
LICENSE = "MIT (TBD)"
DOWNLOAD_URL = ""
LONG_DESCRIPTION = """
Code release for the WACV 2020 paper
"MonoLayout: Amodal scene layout estimation from a single image"
"""
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    # TODO: Add Windows OS
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: MIT",
    "Topic :: Software Development :: Libraries",
]

# Minimum required pytorch version (TODO: check if versions below this work).
TORCH_MINIMUM_VERSION = "1.0.0"

logger = logging.getLogger()
logging.basicConfig(format="%(levelname)s - %(message)s")


def get_requirements():
    return [
        "matplot",
        "numpy",
        "pillow",
        "torch>=1.0",
        "torchvision",
    ]


if __name__ == "__main__":

    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        licence=LICENSE,
        python_requires=">3.6",
        # Package info
        packages=find_packages(exclude=("docs", "test", "examples")),
        install_requires=get_requirements(),
        zip_safe=True,
        classifiers=CLASSIFIERS,
    )
