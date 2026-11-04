import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Scores ANY trained supervised classifier (logistic regression, naive bayes, SVM, trees...).
# Works because every scikit-learn classifier has the same .predict() method.
# The data splits are stored once; then you hand it whichever model you want to score.
#
# ALL validating and testing happens HERE, not in models.py (models.py only trains).

# Validating and testing are the same job - predict, then score - only the data changes = so keeping it in one place means every model is scored in exactly the same way.
    
    # validate / evaluate / compare -> validation data, as often as you like (to choose settings)
    # eval_test -> test data, ONCE at the very end (the final exam)

class SupervisedEvaluator:

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    #score function; predicts the input x (test, val, train) then calculates corresponding score and performance metrics
    def _score(self, model, x, y, name):
        preds = model.predict(x)
        f1 = f1_score(y, preds, average="macro")
        scores = {
            "accuracy": accuracy_score(y, preds),
            "macro_f1": f1,                                    # main score: every class counts equally
            "micro_f1": f1_score(y, preds, average="micro"),   # equals accuracy for single-label multiclass
            "error": 1 - f1,                                   # error = 1 - MACRO F1
        }
        print(f"{name:<10} accuracy {scores['accuracy']:.4f} | "
              f"macro F1 {scores['macro_f1']:.4f} | error {scores['error']:.4f}")
        return preds, scores

    #run the actual prediction
    def evaluate_train(self, model):
        return self._score(model, self.x_train, self.y_train, "train")

    def validate(self, model):
        return self._score(model, self.x_val, self.y_val, "validation")

    # train + validation together, plus the gap (bias / variance check)
    #   training error high          -> bias (underfitting)
    #   gap (val error - train error) big -> variance (overfitting)
    def evaluate(self, model):
        train_preds, train_scores = self.evaluate_train(model)
        val_preds, val_scores = self.validate(model)
        gap = val_scores["error"] - train_scores["error"]
        print(f"{'gap':<10} {gap:.4f}  (validation error - train error)")
        return val_preds, {"train": train_scores, "validation": val_scores, "gap": gap}

    # FINAL EXAM: run ONCE at the very end, after choosing settings on validation
    def eval_test(self, model):
        return self._score(model, self.x_test, self.y_test, "test")

    # side-by-side table of several models on train + validation (never test)
    # models: dict of name -> trained model, e.g. {"Logistic regression": lr.model, "Complement NB": nb.model}
    def compare(self, models):
        rows = []
        for name, model in models.items():
            print(f"--- {name} ---")
            _, s = self.evaluate(model)
            rows.append({"model": name,
                         "train macro F1": s["train"]["macro_f1"],
                         "val macro F1": s["validation"]["macro_f1"],
                         "val accuracy": s["validation"]["accuracy"],
                         "gap": s["gap"]})
        return pd.DataFrame(rows).set_index("model").round(4)
