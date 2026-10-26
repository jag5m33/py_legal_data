import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score

class ComplementNaiveBayes:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, alpha=1.0, norm=False):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        #class_prior allows you to set custom weights manually. You tell the model exactly how common each class should be considered
#fit_prior tells the model whether to figure out those weights automatically

        self.model = ComplementNB(alpha=alpha, norm=norm)
        print(f"Settings: alpha={alpha}, norm={norm}")

    def train(self):
        print("Training complement naive bayes model on imbalanced class x train dataset")
        self.model.fit(self.x_train, self.y_train)
        print(f" Number of classes: {len(self.model.classes_)}") # check the number of classes

    def _score(self, x, y, name):
        preds = self.model.predict(x)
        # error = 1 - MACRO F1
        f1 = f1_score(y, preds, average = 'macro')
        scores = {
            "accuracy": accuracy_score(y, preds),
            "macro_f1": f1_score(y, preds, average = 'macro'),
            "micro_f1": f1_score(y, preds, average = 'micro'),
            "error" : 1- f1

        }               
        print(f"{name:<10} accuracy {scores['accuracy']:.4f} | "
              f"macro F1 {scores['macro_f1']:.4f} | error {scores['error']:.4f}")
        return preds, scores

#run the actual prediction
    def evaluate_train(self):
        return self._score(self.x_train, self.y_train, 'train')
    def validate(self):
        return self._score(self.x_val, self.y_val, 'validation')
    def eval_test(self):
        return self._score(self.x_test, self.y_test, 'test')
    
