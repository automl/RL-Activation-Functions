import os
import setuptools

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Others
        "isort",
        "black",
        "pydocstyle",
        "flake8",
    ]
}

setuptools.setup(
    name="safs-rl",
    author="Your Name",
    author_email="your.email@example.com",
    description="Your project description",
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yourrepository",
    version="0.1.0",
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
    test_suite="pytest",
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
