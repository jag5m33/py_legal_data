import shap

# visualise the model performance and inner workings

# how is the model performing the classifications?

'''
SHAP Notes (LEDGAR: 100 classes, TF-IDF features)

- Original dataset (x_train): 2D, rows = clauses, columns = TF-IDF terms, cells = TF-IDF scores (model INPUT)
- shap_values: 3D stack of spreadsheets explaining the model OUTPUT
    - one spreadsheet per clause, rows are now = TF-IDF terms, columns = all 100 classes
    - shape (100 clauses, n_terms, 100 classes)
- Each cell = SHAP value: how much that TF-IDF term pushed that class's score for that clause
    - + (red) = more likely that class, - (blue) = less likely, ~0 = didn't matter
    - units are logits, not probabilities or TF-IDF
- Base values (shap_values.base_values): model's average score per class (100 of them)
    - base value + sum of a class's column = final score for that class; highest score = prediction
- Slicing: shap_values[1] = clause 1's spreadsheet, shap_values[1, :, 26] = its class 26 column
- Waterfall plots ONE column -> needs a clause AND a class: shap_values[i, :, c]
'''

class ShapVisualizer:
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def visualize(self):
        # create a shap explainer: calcaultes shap values for each row of your matrix
        # NOTE: sklearn LogisticRegression likely needs background data for the base values:
        #       explainer = shap.LinearExplainer(self.model, self.x_train)
        #       (ComplementNB is not supported by LinearExplainer)
        explainer = shap.Explainer(self.model)

        shap_values = explainer(self.x_train)

        # TESTER NOTE:
        # if the feature matrix is large: use a subset of the feature matrix instead:
        #shap_values = explainer(self.x_train[0:100])

        # result is the 3D stack of spreadsheets: np.shape(shap_values.values) -> (100 clauses, n_terms, 100 classes)
        # base values separately:            np.shape(shap_values.base_values) -> (100 clauses, 100 classes)
        return shap_values

    def waterfall_plot(self, shap_values, row_predictor_index, class_index, max_display=10):
        # waterfall plot for a single prediction
        # multi-indexing: shap_values[clause, TF-IDF term, class] - one index per axis, separated by commas
        #   shap_values[row_predictor_index] -> the spreadsheet (one clause): all terms x all 100 classes
        #   shap_values[row_predictor_index, :, class_index] -> one column of it: all terms (":" = every row) for one class
        #   shap_values[row_predictor_index, 500, class_index] -> one cell: one term's push on one class for one clause
        
        # class_index = POSITION in model.classes_ (with labels 0-99 this is the same as the label)
        # max_display picks the terms shown: the (max_display - 1) biggest pushes, the rest folded into "other features"
        shap.plots.waterfall(shap_values[row_predictor_index, :, class_index], max_display = max_display)
        
    def force_model_predictions_plot(self, shap_values, class_index, start=0, end=50):
        # stacked force plot: MANY clauses, ONE class
        #   shap_values[start:end, :, class_index] -> clauses start..end-1, all terms, one class
        #   each clause becomes one thin vertical strip, placed side by side
        #   red = terms pushing towards class_index, blue = pushing away, y-axis = model score (logits)
        #   hover a strip to see that clause; dropdowns re-sort the clauses (by similarity, by output score, ...)
        # interactive JavaScript only (no matplotlib=True for many clauses) -> needs shap.initjs()
        shap.initjs()
        return shap.plots.force(shap_values[start:end, :, class_index])  # return it so the notebook cell displays it
