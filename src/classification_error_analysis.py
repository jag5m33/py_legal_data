import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             multilabel_confusion_matrix, ConfusionMatrixDisplay)


class ErrorAnalysis:
    # Everything needed to study one model's mistakes on ONE split (normally validation).
    # y_true      : the real labels (numbers)
    # y_pred      : the model's guesses (numbers), e.g. val_preds
    # label_names : dict, number -> clause name, e.g. 47 -> "Governing Laws"
    # classes     : all class numbers in a fixed order (lr.model.classes_),
    #               so every table has the same 100 classes in the same order
    # texts       : the clause text for each row, so mistakes can be read
    def __init__(self, y_true, y_pred, label_names, classes, texts=None):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.label_names = label_names
        self.classes = list(classes)
        self.texts = None if texts is None else np.asarray(texts)

    # 1. PER-CLASS SCORES: precision, recall, F1 and support for every clause type, worst F1 first
    def report_df(self):
        report = classification_report(self.y_true, self.y_pred, labels=self.classes,
                                       output_dict=True, zero_division=0)
        df = pd.DataFrame(report).T               # flip so each class is a row
        df = df.iloc[:len(self.classes)]          # keep the class rows, drop the summary rows
        df.index = [self.label_names[c] for c in self.classes]   # numbers -> clause names
        df["support"] = df["support"].astype(int) # how many real clauses of this type
        return df.sort_values("f1-score")

    # 2. TP / FP / FN / TN for every clause type ("this type vs everything else")
    def counts(self):
        cms = multilabel_confusion_matrix(self.y_true, self.y_pred, labels=self.classes)
        return pd.DataFrame({
            "TP": cms[:, 1, 1],    # correctly called this type
            "FP": cms[:, 0, 1],    # called this type, but wasn't
            "FN": cms[:, 1, 0],    # was this type, but called something else
            "TN": cms[:, 0, 0],    # correctly called not this type
        }, index=[self.label_names[c] for c in self.classes])

    # 3. MISTAKES TABLE: one row per wrong clause - its text, true type and predicted type
    def mistakes(self):
        df = pd.DataFrame({"true": self.y_true, "predicted": self.y_pred})
        if self.texts is not None:
            df.insert(0, "text", self.texts)
        df = df[df["true"] != df["predicted"]]                # keep only the wrong ones
        df["true"] = df["true"].map(self.label_names)
        df["predicted"] = df["predicted"].map(self.label_names)
        return df

    # 4. TOP CONFUSED PAIRS: which true type gets predicted as which, most often
    #    one row per (true, predicted) PAIR, with how many times it happened
    def top_confusions(self, n=15):
        wrong = self.mistakes()
        pairs = (wrong.groupby(["true", "predicted"]).size()   # count each pair
                 .sort_values(ascending=False).head(n)          # biggest first
                 .reset_index(name="count"))                    # pair back into columns
        return pairs

    # 5. ONE CLASS IN DETAIL: where did the real clauses of this type end up?
    #    top row is usually the class itself (correct); rows below are what it was confused with
    def class_confusions(self, class_name, n=5):
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)
        num = [k for k, v in self.label_names.items() if v == class_name][0]
        row = cm[self.classes.index(num)]                       # this class's row = its real clauses
        names = [self.label_names[c] for c in self.classes]
        return pd.Series(row, index=names).sort_values(ascending=False).head(n)

    # 6. CONFUSION MATRIX PLOT: rows = true type, columns = predicted type
    #    normalize="true" -> each row shows shares (0-1), so small classes are readable
    def plot_confusion_matrix(self, size=30):
        fig, ax = plt.subplots(figsize=(size, size))
        ConfusionMatrixDisplay.from_predictions(
            self.y_true, self.y_pred, labels=self.classes,
            normalize="true", include_values=False,   # 100 x 100 numbers would be unreadable
            ax=ax, cmap="Blues", colorbar=True)
        ax.set_title("Validation confusion matrix")
        plt.show()