from setuptools import setup, find_packages

setup(
    name="counterfactual_dataset_generator",
    version="0.1",
    description="A counterfactual dataset generator to evaluate language model failures.",
    author="SaferAI",
    author_email="saferai.audit@gmail.com",
    packages=find_packages(),
)