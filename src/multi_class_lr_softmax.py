from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# sklearn's LogisticRegression automatically uses the multinomial (Softmax) formulation
# Encapsulation: Keeps data loading, preprocessing, training, and predicting in one place.
# Reusability: Allows you to create multiple distinct instances of models easily.
# Per-class scores, confusions and mistakes live in src/error_analysis.py (ErrorAnalysis).

class MultiClassLogisticRegression:

    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test,
                 C=1.0, class_weight=None):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


        self.model = LogisticRegression(C=C,
                                        solver="lbfgs",
                                        max_iter=1000,
                                        random_state=42,
                                        class_weight=class_weight,
                                        verbose= 2)
        print(f"Settings: C={C}, class_weight={class_weight}")

    def train(self):
        print("Training multiclass softmax logistic regression...")
        self.model.fit(self.x_train, self.y_train)
        print(f"Number of classes: {len(self.model.classes_)}")

    #score function; predicts the input x (test, val, train) then calculates corresponding score and performance metrics 
    def _score(self, x, y, name):
        preds = self.model.predict(x)
        scores = {
            "accuracy": accuracy_score(y, preds),
            "macro_f1": f1_score(y, preds, average="macro"),   # main score: every class counts equally
            "micro_f1": f1_score(y, preds, average="micro"),   # equals accuracy for single-label multiclass
        }
        print(f"{name:<10} accuracy {scores['accuracy']:.4f} | "
              f"macro F1 {scores['macro_f1']:.4f} | micro F1 {scores['micro_f1']:.4f}")
        return preds, scores

    def evaluate_train(self):
        return self._score(self.x_train, self.y_train, "train")

    def validate(self):
        return self._score(self.x_val, self.y_val, "validation")

    def eval_test(self):
        return self._score(self.x_test, self.y_test, "test")



## NOTE FOR FUTURE:
    ## High Value of C (Low Regularization): Tells the model to give high weight to fitting the training data closely. 
        #  This risks overfitting if the training data contains noise or is not fully representative of real-world data.
    ## Low Value of C (Strong Regularization): Tells the model to prioritize a penalty on large coefficients (keeping the model simple) even if it means missing some training patterns. 
        # This helps prevent overfitting and improves generalisation to new data