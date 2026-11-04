from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

# MODELS ONLY: each class builds and trains one model - on TRAIN data only.
# This script does NOT validate or test.
    # models.py           -> .train(x_train, y_train) -> trained model
    # sup_eval_metrics.py -> validate / evaluate / compare as often as you like,
    
    # eval_test ONCE at the very end

# Train once, keep the trained model in a variable (or save it with joblib),

# then pass lr.model / nb.model to the evaluator - no retraining between validation and test.

# Only retrain if you change a setting (C, class_weight, alpha, norm).

# Scoring lives in src/sup_eval_metrics.py (SupervisedEvaluator).

# Per-class scores, confusions and mistakes live in src/classification_error_analysis.py (ErrorAnalysis).



# 1. Multiclass softmax logistic regression

# sklearn's LogisticRegression automatically uses the multinomial (Softmax) formulation
# Encapsulation: keeps the model's settings and training in one place.
# Reusability: allows you to create multiple distinct instances of models easily.

class MultiClassLogisticRegression:

    def __init__(self, C=1.0, class_weight=None):
        self.model = LogisticRegression(C=C,
                                        solver="lbfgs",
                                        max_iter=1000,
                                        random_state=42,
                                        class_weight=class_weight,
                                        verbose= 2)
        print(f"Settings: C={C}, class_weight={class_weight}")

    def train(self, x_train, y_train):
        print("Training multiclass softmax logistic regression...")
        self.model.fit(x_train, y_train)
        print(f"Number of classes: {len(self.model.classes_)}")
        return self   # lets you write: lr = MultiClassLogisticRegression().train(x_train, y_train)



## NOTE FOR FUTURE:
    ## High Value of C (Low Regularization): Tells the model to give high weight to fitting the training data closely. 
        #  This risks overfitting if the training data contains noise or is not fully representative of real-world data.
    ## Low Value of C (Strong Regularization): Tells the model to prioritize a penalty on large coefficients (keeping the model simple) even if it means missing some training patterns. 
        # This helps prevent overfitting and improves generalisation to new data



# 2. Complement Naive Bayes

class ComplementNaiveBayes:
    def __init__(self, alpha=1.0, norm=False):
        # alpha: smoothing - stops an unseen word giving a class probability of zero
        #        (like 1/C in logistic regression: higher alpha = more bias, lower = more variance)
        # norm:  extra normalisation of the learned weights (made no real difference on LEDGAR)
        #class_prior allows you to set custom weights manually. You tell the model exactly how common each class should be considered
        #fit_prior tells the model whether to figure out those weights automatically
        # (ComplementNB ignores the prior when predicting, by design)

        self.model = ComplementNB(alpha=alpha, norm=norm)
        print(f"Settings: alpha={alpha}, norm={norm}")

    def train(self, x_train, y_train):
        print("Training complement naive bayes model on imbalanced class x train dataset")
        self.model.fit(x_train, y_train)
        print(f" Number of classes: {len(self.model.classes_)}") # check the number of classes
        return self   # allows you to save any number of trained models in variables
    
    
