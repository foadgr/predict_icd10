from setuptools import setup, find_packages

setup(
    name="Predict ICD10",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'scikit-learn',
        'spacy'
    ]
)
