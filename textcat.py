from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import logging
import warnings
import sklearn.exceptions

import random
import pandas as pd
import numpy as np

import spacy
from spacy.training.example import Example
from spacy.util import minibatch

from datetime import datetime
from typing import List, Tuple, Union
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class TextClassificationModel:
    def __init__(self, experiment_id: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.experiment_id = experiment_id if experiment_id else timestamp
        log_filename = f"training_metrics_{self.experiment_id}_{timestamp}.log"
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        logging.info('Training and Validation Metrics:')
        
        self.nlp = spacy.load("en_core_web_sm")
        self.textcat = self.nlp.add_pipe("textcat")
        
    def process_data(self, data: List[Tuple[str, str]], labels: List[str]) -> List[Tuple[str, dict]]:
        processed_data = []
        for text, label in data:
            cats = {lbl: (1.0 if lbl == label else 0.0) for lbl in labels}
            processed_data.append((text, {"cats": cats}))
        return processed_data
    
    def scoring(self, y_true: List[str], y_pred: List[str], phase: str, epoch: int, loss: float, labels: List[str]) -> None:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', labels=labels, zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)

        metrics_line = (f"Epoch {epoch}\t"
                        f"Loss {loss:.4f}\t"
                        f"{phase} Accuracy {accuracy * 100:.2f}%\t"
                        f"{phase} F1 (weighted) {f1 * 100:.2f}%\t"
                        f"{phase} Micro F1 {micro_f1 * 100:.2f}%\t"
                        f"{phase} Precision {precision * 100:.2f}%\t"
                        f"{phase} Recall {recall * 100:.2f}%")
        print(metrics_line)
        logging.info(metrics_line)
        
    def train(self, train_data: List[Tuple[str, dict]], valid_data: List[Tuple[str, dict]], labels: List[str], epochs: int = 10) -> None:
        train_examples = [Example.from_dict(self.nlp.make_doc(text), cats) for text, cats in train_data]
        valid_examples = [Example.from_dict(self.nlp.make_doc(text), cats) for text, cats in valid_data]

        # Initialize the model
        self.nlp.initialize(lambda: train_examples)

        # Training loop
        for epoch in range(epochs):
            random.shuffle(train_examples)
            losses = {}
            for batch in minibatch(train_examples, size=8):
                self.nlp.update(batch, drop=0.5, losses=losses)
            
            y_true_valid = [max(example.reference.cats, key=example.reference.cats.get) for example in valid_examples]
            y_pred_valid = [max(self.nlp(example.text).cats, key=self.nlp(example.text).cats.get) for example in valid_examples]
            
            # Metrics for validation
            self.scoring(y_true_valid, y_pred_valid, f"Validation", epoch + 1, losses['textcat'], labels=labels)
            
    def test(self, test_data: List[Tuple[str, dict]], labels: List[str]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        log_filename = f"testing_metrics_{self.experiment_id}_{timestamp}.log"
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        logging.info('Testing Metrics:')

        test_examples = [Example.from_dict(self.nlp.make_doc(text), cats) for text, cats in test_data]
        test_y_true = [max(example.reference.cats, key=example.reference.cats.get) for example in test_examples]
        test_y_pred = [max(self.nlp(example.text).cats, key=self.nlp(example.text).cats.get) for example in test_examples]
        
        # Metrics for testing
        self.scoring(test_y_true, test_y_pred, "Test", epoch=float('inf'), loss=float('inf'), labels=labels)

    def save_model(self, path: str) -> None:
        self.nlp.to_disk(path)
    
    def train_model(self, data_path: str, test_size: float = 0.3, valid_size: float = 0.5, frac: float = 0.5, epochs: int = 10) -> None:
        data_df = pd.read_csv(data_path).sample(frac=frac, random_state=42)
        data = list(zip(data_df['chf cmplnt'] + ' ' + data_df['A/P'], data_df['icd10encounterdiagcode'].str[:3]))

        labels = list(set([str(label) for _, label in data]))
        for label in labels:
            self.textcat.add_label(label)

        # Split data into training, validation, and test sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        valid_data, test_data = train_test_split(test_data, test_size=valid_size, random_state=42)

        train_data, valid_data, test_data = self.process_data(train_data, labels), self.process_data(valid_data, labels), self.process_data(test_data, labels) # Process the data
        self.train(train_data, valid_data, labels, epochs)
        self.test(test_data, labels)
        self.save_model(f'./frac_{frac}')
