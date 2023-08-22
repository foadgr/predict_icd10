## Quick Start Guide

### Setup

To set up the environment and install the necessary packages, run the provided shell script:

```bash
./setup.sh
```

### Prediction

Once the setup is complete, you can predict ICD-10 code categories by using the `predict_icd10.py` script. Pass in the text you want to predict. For example:

```bash
python predict_icd10.py --text "Lt peroneal tendon tear"
```

This will return the predicted ICD-10 code for the input text.