#!/bin/bash

# Create the conda environment
conda create -n predict_icd10 python=3.11 -y

# Activate the conda environment
conda activate predict_icd10

# Install the required packages
pip install --upgrade --editable .

# Add conda environment to Jupyter
python -m ipykernel install --user --name=predict_icd10

# Update Jupyter and ipywidgets
pip install --upgrade jupyter ipywidgets

#Download `en_core_web_sm` SpaCy model
python -m spacy download en_core_web_sm