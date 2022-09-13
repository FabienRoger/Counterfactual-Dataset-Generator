import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent  # The directory containing this file
README = (HERE / "README.md").read_text()

setup(
    name="countergen",
    version="0.1.2",
    description="A counterfactual dataset generator to evaluate language model.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="SaferAI",
    author_email="saferai.audit@gmail.com",
    packages=["countergen"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/FabienRoger/Counterfactual-Dataset-Generator",
    python_requires=">=3.9",
)
