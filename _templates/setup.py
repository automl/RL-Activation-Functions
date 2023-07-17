import os

import setuptools

from <<package-name>> import (
    author,
    author_email,
    description,
    package_name,
    project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        <<requires::testing
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        endrequires::testing>>
        <<requires::docs
        # Docs
        "automl_sphinx_theme",
        endrequires::docs>>
        # Others
<<<<<<< Updated upstream:_templates/setup.py
        <<requires::mypy "mypy", endrequires::mypy>>
        <<requires::isort "isort", endrequires::isort>>
        <<requires::black "black", endrequires::black>>
        <<requires::pydocstyle "pydocstyle", endrequires::pydocstyle>>
        <<requires::flake8 "flake8", endrequires::flake8>>
        <<requires::pre-commit "pre-commit", endrequires::pre-commit>>
=======
        "isort",
        "black",
        "pydocstyle",
        "flake8",
>>>>>>> Stashed changes:setup.py
    ]
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    <<requires::license license="Apache-2.0", endrequires::license>>
    url=url,
    project_urls=project_urls,
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "jax",
        "flax",
        "distrax",
        "gymnax",
        "optax",
        "numpy",
        "matplotlib",
        "hydra-core",
        "omegaconf",
        "wandb",
        "jaxpruner @ git+https://github.com/google-research/jaxpruner.git"
    ],
    extras_require=extras_require,
    <<requires::testing test_suite="pytest", endrequires::testing>>
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
