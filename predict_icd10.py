import spacy
import argparse
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def predict_icd10(text):

    nlp = spacy.load('frac_0.5')
    doc = nlp(text)
    predicted_label = max(doc.cats, key=doc.cats.get)
    
    return predicted_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict ICD 10 code category.')
    parser.add_argument('--text', required=True, help='Input text for prediction')
    args = parser.parse_args()
    prediction = predict_icd10(args.text)
    print(f'predicted code is: {prediction}')
