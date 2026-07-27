# Introduction to Cross-Validation: overfitting, held-out data, and K folds

**Module 14 Cliff Notes** | Source: lecture transcript "Introduction to Cross-Validation"

---

## TL;DR

- Scoring a model on the **same data it was trained on** answers the wrong question. Training R-squared tells you how well the model memorized; you usually want to know how well it **generalizes** to data it has never seen.
- **Overfitting** is the failure mode: keep adding variables and interaction features and the training R-squared keeps climbing (training RMSE keeps falling) while the model is really learning **idiosyncrasies of that particular training set**, not generalizable patterns.
- The lecture's example: modeling respiratory virus spread from 2020 to 2022 data mostly models **whichever COVID-19 variant was around in those seasons**, not respiratory viruses in general.
- **Fix 1, a holdout test set:** shuffle, carve off 10 to 30 percent, never let the model train on it. Simple and good, but on a small dataset you **lose those rows from training** and the estimate rests on very few test points (Verified: a 10 percent holdout of 60 rows is **6 rows**).
- **Fix 2, cross-validation:** split the shuffled data into **K folds** (K is usually 3, 5, or 10), then run **K iterations**: each iteration trains on the other K-1 folds and tests on the held-out fold. You train **K separate models** and get **K scores** (Verified: `cross_val_score` over a 5-fold `KFold` performs exactly 5 fits, and 10 fits over a 10-fold one).
- Every row gets used: each row is **tested exactly once** and **used for training K-1 times** (Verified on 150 rows, K=5: 5 iterations, 120 train rows and 30 test rows each).
- **The estimate is the average of the K test-fold scores.** Because it is an average, you can attach a standard error and a confidence interval to it, and use that to compare models.
- Why bother: the average is **far steadier** than any one split. On a 60-row demo, a single 20 percent holdout gave test R-squared anywhere from **0.459 to 0.965** across 200 random splits, while the 5-fold average stayed in **0.829 to 0.915** (Verified: standard deviation **0.063** versus **0.015**, about **4.3x** tighter).

> **The one takeaway:** cross-validation turns one lucky-or-unlucky train/test split into K of them, so every row is tested exactly once and the model's score becomes an **average over K held-out folds**, which is both a fairer estimate of generalization and a number you can put a confidence interval around.

---

## Why training scores lie: overfitting

Everything up to this point in the course scored a model on the **training set**: fit the line on the data, then ask how well the fitted line matches that same data (R-squared, RMSE, residuals). That measures fit, not generalization.

The problem is that the training data is **always partial**. It is a sample, never the whole phenomenon, and there is always more data out there that represents the same thing. If you keep adding variables and interaction features to chase a better training score, the model starts fitting the **weird little quirks of your particular sample**.

The lecture's example: if you model respiratory virus spread using data from 2020 to 2022, you are mostly modeling **whatever COVID-19 variant dominated those seasons** (the Delta wave, say), not respiratory viruses in general. Part of the answer is to gather better data. The other part is to **hold some data back** and test on rows the model has never seen.

The lecture's slide makes the point with two pictures. On the **right** is overfitting: a model finding all the little patterns, contorting itself to minimize error on the training points (the "bubbles"). On the **left** is a nicer, smoother, more generalizable pattern that is not so overfit to the data.

**The training score really does keep climbing.** For **nested** models fit by least squares (each model contains all of the previous model's terms), training R-squared can never go down when you add a term, so it climbs or stays flat by construction. Demonstration on 60 seeded points, fitting polynomials of increasing degree (an added example; the lecture itself shows no code):

```python
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(2100)              # seeded, so these numbers reproduce
x = np.sort(rng.uniform(-3, 3, 60))
y = 0.8 * x + np.sin(x) + rng.normal(0, 0.6, 60)
X = x.reshape(-1, 1)

for deg in [1, 2, 3, 5, 9, 15]:
    pipe = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
    pipe.fit(X, y)
    train_r2 = r2_score(y, pipe.predict(X))                      # scored on training data
    cv_r2 = cross_val_score(pipe, X, y,
                            cv=KFold(5, shuffle=True, random_state=0),
                            scoring='r2').mean()                 # scored on held-out folds
```

Verified results:

```text
degree  1:  train R2 = 0.874    5-fold CV R2 = 0.855
degree  2:  train R2 = 0.880    5-fold CV R2 = 0.853
degree  3:  train R2 = 0.922    5-fold CV R2 = 0.899
degree  5:  train R2 = 0.925    5-fold CV R2 = 0.901   <- best honest score
degree  9:  train R2 = 0.934    5-fold CV R2 = 0.841
degree 15:  train R2 = 0.943    5-fold CV R2 = 0.271   <- training score says "best model"
```

Training R-squared rises **monotonically** across all six models, right up to degree 15 (Verified). The cross-validated score peaks at degree 5 and then falls off a cliff. That gap is overfitting, made visible.

> **Not in the lecture, but worth knowing:** adjusted R-squared is not a substitute for held-out data. It penalizes extra terms, so unlike plain R-squared it *can* go down, but it is still computed on the training data. Only data the model has not seen answers "does this generalize?"

## Fix 1: pull out a test set

The straightforward approach: take the full dataset, **shuffle the rows**, then split off a **test set** that the model never uses to find its coefficients. (Fitting is the step where the model is handed the training data and works out the slope, the coefficient for every variable. The lecture notes the course did not go into that process itself.) The lecture suggests holding out **10, 20, or 30 percent**. Train on the rest, score once on the held-out portion, and that score estimates performance beyond the training data.

This is a good approach, and the lecture says so. It has two costs, both worse on small datasets:

1. **You lose those rows from training.** Less data for the model to learn from.
2. **The estimate rests on very few points.** Verified on a 60-row dataset:

```text
test_size=0.1  ->  train 54 rows, test  6 rows
test_size=0.2  ->  train 48 rows, test 12 rows
test_size=0.3  ->  train 42 rows, test 18 rows
```

Six rows is not a trustworthy estimate of anything. And that particular handful of rows "might be kind of weird itself," as the lecture puts it. You get one number and no sense of how much it would have moved had the shuffle landed differently.

## Fix 2: cross-validation

Cross-validation is a way to run **multiple train/test splits** and combine them. The recipe:

1. **Shuffle** the rows.
2. **Split into K folds** of (as near as possible) equal size. K is typically **3, 5, or 10**.
3. For each fold i in 1..K: **train on all folds except i**, **test on fold i**.
4. **Average the K test scores.**

With K=5 the first iteration trains on folds 1 to 4 (80 percent) and tests on fold 5 (20 percent); the next trains on folds 1, 2, 3 and 5 and tests on fold 4; and so on until every fold has taken one turn as the test set.

The important consequence: **you train K models, not one.** Five folds means five separate linear regressions, each fit on a slightly different 80 percent, each scored on the 20 percent it never saw.

Verified mechanics on 150 rows with K=5:

```text
iter 1: train 120  test 30
iter 2: train 120  test 30
iter 3: train 120  test 30
iter 4: train 120  test 30
iter 5: train 120  test 30

each row tested exactly once      (Verified)
each row used for training 4 times = K-1   (Verified)
fits performed by cross_val_score: 5 for K=5, 10 for K=10   (Verified)
```

Under the hood there is no magic. A hand-rolled loop reproduces `cross_val_score` **exactly** (Verified: identical to 6 decimal places on the demo data):

```python
from sklearn.base import clone

scores = []
for train_idx, test_idx in KFold(5, shuffle=True, random_state=0).split(X):
    m = clone(model)                       # a fresh, untrained copy each iteration
    m.fit(X[train_idx], y[train_idx])      # train on K-1 folds
    scores.append(r2_score(y[test_idx], m.predict(X[test_idx])))   # test on the held-out fold
```

## Why the average is worth the extra work

Same model, same 60 rows, two ways of estimating its held-out R-squared, each repeated 200 times with seeds 0 to 199 (the seed drives `train_test_split` in one case and the `KFold` shuffle in the other). Verified:

| Method | mean R2 | sd | min | max | range |
|---|---|---|---|---|---|
| single 20 percent holdout | 0.888 | 0.063 | 0.459 | 0.965 | 0.506 |
| 5-fold CV average | 0.893 | 0.015 | 0.829 | 0.915 | 0.085 |

Both land in the same place on average, but a single split is about **4.3x more volatile** (Verified). Draw one unlucky split and you would report 0.46 for a model that is really around 0.89. That instability is precisely what cross-validation averages away, which is the lecture's argument for it.

## From K scores to one number (plus an interval)

You "simply average the performance score across all test folds." And because the result is an **average**, everything the course taught about averages applies: the **standard error of the mean**, **confidence intervals**, and therefore **model comparison**. (The transcript renders this as "the standard area, the mean," which is a transcription slip for "the standard error of the mean.")

Worked on the demo data with RMSE (Verified):

```text
RMSE per fold:  [0.554, 0.528, 0.501, 0.756, 0.442]
mean RMSE:      0.556
sd (ddof=1):    0.119        sem: 0.053        t* (df = K-1 = 4): 2.776
95% CI:         [0.408, 0.704]
```

That interval is the point of the exercise. A single number cannot tell you whether model A really beats model B or whether the difference is fold-to-fold noise; a mean with an interval can. The lecture closes by promising exactly this: code that runs the process "for evaluating models and for being able to select which model performs well, generalizes well to unseen data."

## Shuffle first, and mean it

The lecture flags this in passing ("you'll want to randomize your data before this, so shuffle the rows") and then draws the folds as contiguous blocks. Treat the shuffle as **mandatory**, not cosmetic. If the rows arrive sorted or grouped (by date, by category, by the predictor itself), contiguous folds are systematically unrepresentative and every model gets tested on a region it never trained on.

Verified on the demo data, which is sorted by x:

```text
no shuffle:  per-fold R2 = [-3.852, 0.420, 0.356, 0.444, -8.145]   mean = -2.155
shuffled:    per-fold R2 = [ 0.894, 0.893, 0.943, 0.822, 0.941]    mean =  0.899
```

Same data, same model, same K. Without shuffling, the estimate is not merely worse, it is meaningless (a negative R-squared means the model does worse than just predicting the mean). Shuffling is exactly why the companion scikit-learn lecture builds an explicit `KFold(n_splits=5, shuffle=True, random_state=...)` object rather than taking the shortcut of passing `cv=5`, which does not shuffle.

## Fine print the lecture glosses

- **"K equal folds" is approximate.** When the row count does not divide evenly by K, scikit-learn makes the folds as equal as possible: the first `n % K` folds get one extra row. Verified: n=100 with K=3 gives test folds of **34, 33, 33**; n=150 with K=4 gives **38, 38, 37, 37**. *Exam safety: the course says "equal folds," so answer it that way; the remainder detail is implementation trivia.*
- **"Use all of our data for training and testing" describes the procedure, not any one model.** Across all K iterations, yes, every row serves both duties. Within a single one of the K models, never: a model is not tested on any row it trained on, which is the whole point. (Verified: each row is tested once and used for training K-1 times.)
- **"Test set" here is really a validation set.** The lecture calls each held-out fold a test fold. In the broader convention, cross-validation folds give a **validation** estimate used to *choose* a model, and a separate untouched **test set** is reserved for the final unbiased number, because once you use CV scores to pick a model you have used that data to make a decision. *Exam safety: this course says "test set" and "test fold"; use its wording on a quiz.*
- **The confidence interval built from fold scores is approximate.** The usual "mean plus or minus t times SEM" assumes the K scores are independent draws. They are not, because the K training sets overlap heavily: at n=150 with K=5, any two of the five training sets share **90 of the 150 rows** (60 percent of the data, 75 percent of each training set) (Verified). It is a known result in the machine learning literature that there is **no unbiased estimator of the variance of K-fold cross-validation**, so treat the interval as a useful rule of thumb rather than a guarantee. In this demo it happened to land on the **conservative** side: the fold-based 95 percent half-width averaged **0.059**, while the CV mean's actual spread was sd **0.022** across 200 freshly generated 60-row datasets (1.96 times sd = **0.043**) and sd **0.015** across reshuffles of the same rows (1.96 times sd = **0.029**) (Verified). *Exam safety: the course wants the confidence interval computed from the fold scores' standard error of the mean; compute it that way.*
- **Fold order does not matter.** The lecture's slide walks the test fold backward (last fold first, then the fourth, and so on). Scikit-learn walks forward. Nothing about the averaged result depends on the order.

---

## Quiz-ready facts

- **Why not score on training data:** it measures memorization, not generalization. The goal is to estimate performance on **unseen data from the same phenomenon**.
- **Overfitting** = the model captures idiosyncrasies of the particular training sample instead of generalizable patterns. Symptom: training R-squared keeps rising (training RMSE keeps falling) as you add variables and interactions, while held-out performance stalls or collapses (Verified: degree 15 hit train R2 **0.943** but CV R2 **0.271**).
- For **nested** least-squares models, training R-squared **cannot decrease** when you add a term. Adjusted R-squared can, but it is still a training-data statistic.
- **Lecture example of overfitting:** 2020 to 2022 respiratory virus data mostly models the COVID-19 variant of those seasons, not respiratory viruses in general.
- **Holdout split:** shuffle, then reserve **10 to 30 percent** as a test set. Downsides on small data: fewer training rows, and a test set too small to trust (Verified: 10 percent of 60 rows = **6 rows**).
- **Cross-validation:** shuffle, split into **K folds** (typically **3, 5, or 10**), and for each fold **train on the other K-1 folds and test on that fold**.
- **K folds means K models trained and K scores produced** (Verified: `cross_val_score` runs exactly K fits). Each row is **tested exactly once** and used for **training K-1 times**.
- **With n=150 and K=5** (Verified): 5 iterations, **120 training rows** and **30 test rows** each; the test fold is 1/K = **20 percent**.
- **The CV estimate is the mean of the K test-fold scores.** Because it is a mean, you can compute a standard error and a **confidence interval** (Verified demo: fold RMSEs mean **0.556**, sem **0.053**, t* at df=4 is **2.776**, 95 percent CI **[0.408, 0.704]**), and use those to **compare models**.
- **Degrees of freedom for the fold-score t interval is K-1** (4 when K=5).
- **CV is much more stable than a single split** (Verified over 200 seeds on 60 rows): single-holdout R2 ranged **0.459 to 0.965** (sd 0.063); the 5-fold average ranged **0.829 to 0.915** (sd 0.015), about **4.3x** tighter.
- **Shuffle before folding.** On x-sorted demo data, unshuffled 5-fold gave a mean R2 of **-2.155** versus **0.899** shuffled (Verified).
- **Folds are equal-sized only when K divides n**; otherwise the first `n % K` folds take one extra row (Verified: n=100, K=3 gives 34, 33, 33).
- **Fold scores are not independent** (Verified: at n=150, K=5, two training sets share 60 percent of the data), so the fold-based confidence interval is an approximation.

---

> **See also:** "Cross-Validation with Scikit-learn (Cliff Notes)" and "Cross-Validation with Scikit-learn - Notebook (Cliff Notes)" are the direct companions, turning this walkthrough into `KFold(n_splits=5, shuffle=True, random_state=...)` plus `cross_val_score` with `neg_root_mean_squared_error` and `r2`. "Comparing Models with Cross-Validation (Cliff Notes)" is where the averaged scores and their intervals actually decide between candidate models. "Linear Regression Residuals and R-Squared (Cliff Notes)" covers the training-set metrics whose limits motivate all of this, and "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" covers the error metric most often averaged across folds.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
