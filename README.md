# LEDGAR: Legal Contract Clause Classification (NLP)

Task: read a clause from a commercial contract and predict its type (Governing Laws, Indemnifications, Confidentiality and so on) out of 100 clause types.

Uses the LEDGAR subset of [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue): contract clauses taken from SEC filings, labelled by their section headings. The project covers EDA, text preprocessing, a TF-IDF and multiclass logistic regression model, and error analysis.

## Results (validation set)

| Split | Accuracy | Macro F1 | Micro F1 |
|---|---|---|---|
| Train | 0.899 | 0.887 | 0.899 |
| Validation | 0.839 | 0.780 | 0.839 |

- Macro F1 is the main metric. It weights all 100 clause types equally, so rare types count as much as common ones. Accuracy and micro F1 are the same thing for this kind of problem and are dominated by the large classes.
- The 11-point macro F1 gap between train and validation shows some overfitting, mostly in rare classes.
- 1,607 of the 10,000 validation clauses were misclassified.
- The test set (10,000 clauses) is held back and has not been scored yet.

## Error analysis

Most errors are between clause types that mean nearly the same thing. The most common mistakes on validation:

| True type | Predicted as | Count |
|---|---|---|
| Governing Laws | Applicable Laws | 73 |
| Applicable Laws | Governing Laws | 33 |
| Amendments | Modifications | 25 |
| Governing Laws | Venues | 24 |
| Authorizations | Authority | 21 |
| Indemnifications | Indemnity | 18 |
| Tax Withholdings | Withholdings | 18 |
| Definitions | Defined Terms | 16 |
| Waivers | No Waivers | 15 |

LEDGAR's labels come from whatever heading a contract's author chose, so near-identical wording can carry different labels. The EDA found this directly (see below), so a large share of the remaining error is label noise rather than model error.

Weakest clause types (validation F1): Applicable Laws 0.30, Jurisdictions 0.35, Venues 0.36, General 0.46. All except General belong to the governing law, jurisdiction and venue group. Two classes are too rare in validation to judge: Books has no examples and Assigns has three.

## EDA

- Size: 60,000 train, 10,000 validation and 10,000 test clauses, with no missing values.
- Class imbalance: the largest class (Governing Laws) has 3,167 examples and the smallest (Books) has 23, a ratio of 138 to 1. Accuracy alone would be misleading.
- Clause length: right-skewed. The median is 86 words, the 95th percentile is 305, and some clauses are over 1,000 words (possibly several clauses stuck together). 8,629 training clauses are longer than 200 words.
- Duplicates: 64 duplicate clauses in train, 3 of them with conflicting labels (No Waivers / Waivers, Applicable Laws / Governing Laws, Qualifications / Existence). The conflicting rows and the extra copies of the rest were removed, leaving 59,933 training clauses.
- Leakage: no test clause also appears in train.

The conflicting-label pairs found in the EDA are the same pairs the model confuses most.

## Pipeline

Raw clause, then regex cleaning, then spaCy lemmatisation, then custom stop words, then TF-IDF, then softmax logistic regression.

- Cleaning (src/clean.py): lowercase; replace emails, URLs, money and numbers with placeholder tokens (emailtoken, moneytoken, numtoken); remove list markers like (a) and (ii); remove punctuation.
- Lemmatisation (src/lemmetizer.py): spaCy en_core_web_sm, with the parser and named-entity recogniser turned off for speed. For example "parties" becomes "party".
- Stop words: scikit-learn's English list with two changes. Negations (no, not, nor, never, without, cannot and others) were kept, because removing them flips a clause's meaning, as were words from clause names (further, interest). Legal words that appear in almost every clause were added (shall, party, agreement, hereof, hereunder, herein).
- TF-IDF (src/tf_idf.py): single words and phrases up to 3 words, min_df=4, max_df=0.9, sublinear_tf=True (so a word repeated ten times isn't weighted ten times as heavily). Fitted on train only. 95,559 features.
- Model (src/multi_class_lr_softmax.py): multinomial logistic regression, C=1.0, class_weight='balanced' to account for the class imbalance.
- Error analysis (src/classification_error_analysis.py): per-class precision, recall and F1; TP/FP/FN/TN per class; most confused pairs; a table of the wrong clauses to read; a confusion matrix.

## Repository structure

- notebooks/eda.ipynb: loads LEDGAR, class balance, clause length, duplicates, leakage check
- notebooks/model.ipynb: cleaning, lemmatisation, stop words, TF-IDF, model and error analysis
- src/: the cleaning, lemmatiser, TF-IDF, model and error analysis code used by the notebooks
- requirements.txt

## How to run

```
python -m venv legal_venv
legal_venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On Mac or Linux, activate with `source legal_venv/bin/activate`.

Run eda.ipynb, then model.ipynb. The first notebook downloads LEDGAR from Hugging Face and saves it to data/, which is not committed to the repo. The lemmatised text is saved to data/processed/, so the slow spaCy step only runs once.

## Next steps

- Tune C on validation, and compare class_weight='balanced' with no weighting.
- Try merging near-identical labels (for example Governing Laws and Applicable Laws) to measure how much of the error is label noise.
- Compare the TF-IDF model with sentence-transformer embeddings and a fine-tuned legal transformer. About 14% of training clauses are over 200 words and would need truncating.
- Score the test set once, after the model is final.
