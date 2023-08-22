import spacy
import argparse

def predict_icd10(text):

    nlp = spacy.load('./model/frac_0.2')
    doc = nlp(text)
    predicted_label = max(doc.cats, key=doc.cats.get)
    
    return predicted_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict ICD 10 code category.')
    parser.add_argument('--text', required=True, help='Input text for prediction')
    args = parser.parse_args()
    prediction = predict_icd10(args.text)
    print(f'predicted code is: {prediction}')
