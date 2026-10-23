# LEDGAR pipeline notes: from raw data to model

How the project goes from raw contract clauses to a trained model, one stage at a time.
Each stage shows **what the object looks like**, **what one row / column / cell means**, and **which file it lives in**.

---

## The whole flow at a glance

```
 data/raw/*.parquet                     (Hugging Face LEDGAR: text + label)
        │
        ▼   eda.ipynb  ── explore, de-duplicate, lowercase train, add word_count
 data/interim/train_eda.parquet
        │
        ▼   preprocess.ipynb  ── RUN ONCE
        │     clean_text()        → clean_text column
        │     Lemmatizer          → lemmatized_text column
 data/processed/proc_{train,validation,test}.parquet
        │     stop words + TfidfVectorizer (fit on train, transform all 3)
        ▼
 data/features/x_train.npz  x_val.npz  x_test.npz  vectorizer.joblib
        │
        ├──▶  model_mcc.ipynb   softmax logistic regression → scores → ErrorAnalysis
        └──▶  model_nb.ipynb    Naive Bayes                 → scores → ErrorAnalysis
```

| Notebook | Job | How often |
|---|---|---|
| `eda.ipynb` | understand the data, fix duplicates / conflicts | once |
| `preprocess.ipynb` | text → numbers, save to disk | once (re-run only if preprocessing changes) |
| `model_mcc.ipynb` | train + evaluate logistic regression | as often as you like |
| `model_nb.ipynb` | train + evaluate Naive Bayes | as often as you like |

---

## Stage 0: raw data (`data/raw/`)

One row = one contract clause. Three files: train, validation, test.

```
                         text                                        label
0  "Except as otherwise set forth in this Debenture, the Company…"     97
1  "No ERISA Event has occurred or is reasonably expected to occur…"   39
2  "This Amendment may be executed by one or more of the parties…"     26
```

| Column | Type | Meaning |
|---|---|---|
| `text` | string | the clause, exactly as in the contract |
| `label` | int 0–99 | which of the 100 clause types it is |

Sizes: train 60,000 · validation 10,000 · test 10,000.

---

## Stage 1: EDA output (`data/interim/train_eda.parquet`)

Made by `eda.ipynb`. **Train only** (validation and test are not changed by the EDA).

```
                         text                         label   label_name     word_count
0  "except as otherwise set forth in this debenture…"   97    Waivers             47
1  "no erisa event has occurred or is reasonably…"      39    Erisa               74
2  "this amendment may be executed by one or more…"     26    Counterparts        53
```
Shape: **59,933 × 4**

What changed vs raw:
- text **lowercased**
- `label_name` = the readable name of the label (97 → "Waivers")
- `word_count` = words per clause (median 86)
- 6 rows with **conflicting labels** dropped, 61 **duplicates** removed

---

## Stage 2: cleaning (`clean_text` column)

`src/clean.py → clean_text()` runs on **one clause at a time**:

```
BEFORE:  "Notices: (a) send to legal@acme.com within 30 days; (ii) pay $1,250,000.00 per Section 5.2"
AFTER:   "notices send to emailtoken within numtoken days pay moneytoken per section numtoken"
```

| Step (order matters) | Example |
|---|---|
| emails → `emailtoken` | legal@acme.com → emailtoken |
| money → `moneytoken` (before numbers!) | $1,250,000.00 → moneytoken |
| remove list markers | (a), (ii) → gone |
| numbers → `numtoken` | 30, 5.2 → numtoken |
| lowercase, strip punctuation, collapse spaces | "Notices:" → notices |

New column added: `clean_text`.

---

## Stage 3: lemmatising (`lemmatized_text` column)

`src/lemmetizer.py → Lemmatizer` (spaCy, parser + ner switched off).

```
clean_text:       "notices send to emailtoken within numtoken days pay moneytoken"
                        │ spaCy tokenises → tags each word (noun/verb…) → finds base form
lemmatized_text:  "notice send to emailtoken within numtoken day pay moneytoken"
```

| Word | Lemma |
|---|---|
| parties | party |
| notices | notice |
| terminated | terminate |
| given | give |

Saved as **`data/processed/proc_*.parquet`**:

```
proc_train (59,933 × 6)
   text | label | label_name | word_count | clean_text | lemmatized_text
proc_val / proc_test (10,000 × 5)          (no word_count column)
   text | label | label_name | clean_text | lemmatized_text
```

---

## Stage 4: stop words (`stop_words_set`)

A **Python set of words to throw away** before counting.

```
stop_words_set = ( scikit-learn's 318 English stop words
                   + {shall, party, agreement, hereof, hereunder, herein} )
                   − {no, not, nor, never, neither, none, nothing, nobody,
                      without, cannot, further, interest}
               → 311 words
```

Must be passed to TF-IDF as a **list**: `tfid(list(stop_words_set))`.

---

## Stage 5: the vectoriser (TF-IDF)

### What it is
`vectorizer = tfid(list(stop_words_set))`: a **TfidfVectorizer**, the machine that turns text into numbers.

Settings: `ngram_range=(1,3)`, `min_df=4`, `max_df=0.9`, `sublinear_tf=True`, `norm="l2"`.

### What it does, in two steps

```
                    ┌──────────────── FIT (train only) ────────────────┐
proc_train          │ 1. split each clause into n-grams                 │
.lemmatized_text ──▶│ 2. drop stop words, rare (<4 clauses) and very     │
                    │    common (>90% of clauses) n-grams                │
                    │ 3. remember the VOCABULARY (the column list)       │
                    │ 4. remember each n-gram's IDF (how rare it is)     │
                    └───────────────────────────────────────────────────┘
                                         │
                    ┌───────────── TRANSFORM (train, val, test) ────────┐
any clauses ───────▶│ for each clause, score every vocabulary n-gram:   │
                    │   TF  (how often in THIS clause)                  │
                    │ × IDF (how rare across ALL train clauses)         │──▶ x matrix
                    │ then scale each row to length 1 (L2)              │
                    └───────────────────────────────────────────────────┘
```

**Why fit on train only:** validation and test stand in for future clauses the model has never seen. If the vocabulary or IDF were learned from them, they would leak information into training.

### What an n-gram is

From `"party may terminate agreement"` (after stop words → `"terminate"` remains, plus pairs/triples that survived):
```
unigram (1 word):  "terminate", "notice", "presentment"
bigram  (2 words): "terminate agreement", "waive presentment"
trigram (3 words): "waive presentment protest"
```

### The output: `x_train`

```
                     columns = n-grams (feature_names)
              "aa"  "notice"  "presentment"  "waive presentment"  …  (213,105 columns)
rows    0  [ 0.00    0.00       0.215          0.184            …  ]   ← clause 0 (Waivers)
=       1  [ 0.00    0.00       0.000          0.000            …  ]   ← clause 1 (Erisa)
clauses 2  [ 0.00    0.06       0.000          0.000            …  ]   ← clause 2
        …
    59,932 [ …                                                       ]
```

| Part | Meaning |
|---|---|
| **one row** | one clause, as 213,105 numbers |
| **one column** | one n-gram |
| **one cell** | TF-IDF score of that n-gram in that clause |
| **cell = 0** | n-gram not in this clause (almost every cell) |
| **cell high** | n-gram frequent in this clause AND rare overall → distinctive |
| range | **0 to 1**, never negative |

Real numbers: **59,933 × 213,105**, **6,034,686** non-zero cells (~101 per clause), >99.9% zeros, so it is stored as a **sparse matrix** (only the non-zeros are kept).

Clause 0 (a Waivers clause), its top scores:
```
presentment               0.215   ← rare banking word, very distinctive
protest                   0.211
maturity                  0.192
waive presentment protest 0.184   ← a trigram
…
notice                    lower   ← appears in many clause types
```

### The other objects

| Object | Shape | What it is |
|---|---|---|
| `x_train` | 59,933 × 213,105 | TF-IDF of train clauses |
| `x_val` | 10,000 × 213,105 | same columns, validation clauses |
| `x_test` | 10,000 × 213,105 | same columns, test clauses |
| `feature_names` | 213,105 | `vectorizer.get_feature_names_out()`: the n-gram for each column, same order |
| `vectorizer` | – | saved fitted machine; can `.transform()` brand-new clauses later |

**Labels are never in `x`.** The vectoriser only ever sees `lemmatized_text`.

Saved as `data/features/x_train.npz`, `x_val.npz`, `x_test.npz`, `vectorizer.joblib`.

---

## Stage 6: labels (`y`) and the name lookup

```
y_train = proc_train.label        →  [97, 39, 26, 45, 11, …]   (59,933 numbers)
y_val   = proc_val.label          →  (10,000 numbers)
y_test  = proc_test.label         →  (10,000 numbers)

label_names = {97: "Waivers", 39: "Erisa", 47: "Governing Laws", 4: "Applicable Laws", …}
```

Row *i* of `x_train` ↔ position *i* of `y_train`: the clause and its correct answer.

```
x_train row 0  →  y_train[0] = 97 (Waivers)
x_train row 1  →  y_train[1] = 39 (Erisa)
```

---

## Stage 7a: model, softmax (multinomial) logistic regression

`src/multi_class_lr_softmax.py → MultiClassLogisticRegression`

### How it predicts one clause

```
clause row x (213,105 TF-IDF numbers)
        │
        ▼  1. SCORE every class:  z_k = x · w_k + b_k     (one weight row per class)
   z = [ Waivers: 4.1,  No Waivers: 2.3,  Notices: -0.8,  …  100 scores ]
        │
        ▼  2. SOFTMAX: turn scores into probabilities that add to 1
   p = [ Waivers: 0.81, No Waivers: 0.13, Notices: 0.01,  …  ]   sum = 1
        │
        ▼  3. PICK the biggest (argmax)
   prediction = Waivers (97)
```

Softmax: `p_k = e^(z_k) / sum of e^(z_j)`. A big score becomes a big probability, and raising one class lowers the others.

### What it learns: `lr.model.coef_`

```
                     "aa"   "presentment"  "waive presentment"  "govern"  …  (213,105)
class 0  (Adjust.)  [ 0.0     -0.4            -0.3              -0.1    … ]
…
class 47 (Gov Laws) [ 0.0     -0.2            -0.1              +3.8    … ]
…
class 97 (Waivers)  [ 0.0     +5.2            +4.9              -0.6    … ]
        (100 rows)
```

| Part | Meaning |
|---|---|
| `coef_` | 100 × 213,105 weights |
| **one cell** | how much that n-gram pushes a clause towards (+) or away from (−) that class |
| **can be negative** | yes. TF-IDF is 0–1, but **weights** are any number |
| `intercept_` | 100 numbers, the starting score per class (`b_k`) |
| `classes_` | [0 … 99], the row order of `coef_` |

### Settings

| Setting | Meaning |
|---|---|
| `C` | 1/λ. Small C = strong regularisation (simpler); large C = weak (fits train closer) |
| `class_weight="balanced"` | rare classes count more, helps macro F1 |
| `max_iter=1000` | how many rounds of nudging the weights |

### What the methods return

```
lr.train()                            → fits coef_ and intercept_
train_preds, train_scores = lr.evaluate_train()
val_preds,   val_scores   = lr.validate()
test_preds,  test_scores  = lr.eval_test()        (ONCE, at the end)

val_preds  = array([97, 39, 4, 47, …])            one guess per clause (10,000)
val_scores = {"accuracy": 0.874, "macro_f1": 0.816, "micro_f1": 0.874}
```

### Current results (trigrams, C = 30, balanced)

| Split | Accuracy | Macro F1 |
|---|---|---|
| Train | 0.995 | 0.995 |
| Validation | 0.874 | 0.816 |
| Test | 0.868 | 0.820 |

Train is almost perfect but validation is ~18 points lower in macro F1. That big gap means **variance** (memorising train). Worth comparing with a smaller C.

---

## Stage 7b: model, Naive Bayes (for `model_nb.ipynb`)

Same `x_train` / `y_train`, a different way of deciding.

### The idea
"For each class, how likely is each n-gram?" Then for a new clause: "which class makes these n-grams most likely?"

```
LEARN (per class, from train):
   prior  P(Waivers)              = share of train clauses that are Waivers
   P(n-gram | Waivers)            = how much of Waivers' total TF-IDF goes to that n-gram

PREDICT one clause:
   score_k = log P(class k)  +  Σ  (TF-IDF of n-gram) × log P(n-gram | class k)
   prediction = class with the highest score
```

"Naive" = it assumes n-grams are independent of each other. That's not true, but it still works well for text.

### What it learns

```
nb.feature_log_prob_     100 × 213,105   log P(n-gram | class), always ≤ 0
nb.class_log_prior_      100             log P(class)
```

| Variant | When |
|---|---|
| `MultinomialNB` | the standard text version |
| `ComplementNB` | built for **imbalanced** classes (yours: 137 to 1); usually better here |

Key settings: `alpha` (smoothing, so an unseen n-gram doesn't give probability 0).
Needs **non-negative** inputs, which TF-IDF is.

### LR vs NB at a glance

| | Logistic regression | Naive Bayes |
|---|---|---|
| Learns | weights by gradient descent | counts/probabilities in one pass |
| Speed | minutes | seconds |
| Handles overlapping n-grams | yes | assumes independence |
| Usually | more accurate | a fast, strong baseline |

---

## Stage 8: error analysis (`src/classification_error_analysis.py`)

`ea = ErrorAnalysis(y_true=y_val, y_pred=val_preds, label_names, classes, texts)`

### `ea.report_df()`: one row per class, worst first
```
                   precision  recall  f1-score  support
Books                0.00      0.00     0.00       0     ← no examples in validation
Applicable Laws      0.39      0.38     0.39      69
Assigns              0.50      0.33     0.40       3
Venues               0.38      0.63     0.47      19
```
- **precision**: of the clauses the model *called* this type, how many really were
- **recall**: of the clauses that *really are* this type, how many it found
- **support**: how many real clauses of this type are in validation

### `ea.counts()`: TP / FP / FN / TN per class
```
                 TP    FP   FN    TN
Governing Laws  431    50   63  9456
Applicable Laws  26    40   43  9891
```
- TP = correctly called this type · FP = called this type, wrongly
- FN = missed it · TN = correctly said "not this type"

### `ea.mistakes()`: one row per wrong clause
```
     text                                               true             predicted
4    "This Amendment shall be interpreted and…"       Governing Laws   Applicable Laws
98   "No delay, omission or failure by Landlord…"     No Waivers       Waivers
```
1,263 wrong out of 10,000 (12.6%).

### `ea.top_confusions()`: one row per (true → predicted) pair
```
     true              predicted          count
0    Governing Laws    Applicable Laws      39
1    Applicable Laws   Governing Laws       37
2    Defined Terms     Definitions          18
3    Withholdings      Tax Withholdings     17
```

### `ea.class_confusions("Assigns")`: where one class's real clauses went
```
Assigns            1    ← correct
Binding Effects    1
Assignments        1
```

### `ea.plot_confusion_matrix()`
```
               predicted →
             c0   c1   c2  …  c99
true   c0  [ ██   ·    ·      ·  ]     dark diagonal = correct
 ↓     c1  [ ·    ██   ░      ·  ]     ░ off-diagonal = a mix-up
       …
```
Rows = true class, columns = predicted, each row shows shares (0–1).

---

## Stage 9: why Governing vs Applicable Laws is hard

Using the model's weights: `diff = coef_[Governing Laws] − coef_[Applicable Laws]`

```
positive (→ Governing Laws):  "amendment construe", "letter govern", "plan govern law"
negative (→ Applicable Laws): "law thereof jurisdiction", "apply without reference", "performable"
```

Checking the text: **~74% of Applicable Laws clauses are choice-of-law clauses** worded like Governing Laws. LEDGAR labels come from the contract's own heading, so the two classes overlap. This is **label noise**, a ceiling on accuracy for this pair, not a model bug.

---

## One-line glossary

| Term | One line |
|---|---|
| clause / document | one row |
| n-gram | 1, 2 or 3 words in a row; one column |
| vocabulary | all kept n-grams = all columns |
| TF | how often an n-gram appears in this clause |
| IDF | how rare the n-gram is across train clauses |
| TF-IDF | TF × IDF, 0 to 1, the number in each cell |
| sparse matrix | stores only non-zero cells |
| fit / transform | learn from train / apply to any data |
| `x` | the numbers the model reads |
| `y` | the correct answers (labels) |
| coefficient / weight | what the model learned per n-gram per class, can be negative |
| softmax | turns 100 scores into probabilities that sum to 1 |
| macro F1 | F1 averaged with every class counted equally, your main score |
