# Cross-Validation with Scikit-learn: design matrices, KFold, and cross_val_score

**Module 14 Cliff Notes** | Source: lecture transcript "Cross-Validation with Scikit-learn"

---

## TL;DR

- **statsmodels cannot do cross-validation well.** It is great for coefficients, p-values, and the R-style formula interface, but it "doesn't really have much functionality for cross-validation." **scikit-learn** does, so the lecture writes the model in statsmodels' formula language and then hands it to scikit-learn.
- The bridge between the two is a **design matrix** (also called the **feature array**): the formula gets expanded into an explicit numeric matrix with **one column per fitted coefficient**, including the intercept column, every dummy column for a categorical, and every interaction column.
- The expansion is done by a third package, **`patsy`**, via **`dmatrices(formula, data=...)`**, which returns **`y_feature, x_features`** in that order (outcome first, inputs second).
- Cross-validation itself is three objects: **`KFold`** (how to split), **`LinearRegression`** (what to train), **`cross_val_score`** (run it and score every fold). Set `shuffle=True` and a `random_state` on `KFold`, and `fit_intercept=False` on `LinearRegression` because the design matrix already carries an intercept column.
- scikit-learn scorers are **always "higher is better"**, so RMSE is exposed as **`'neg_root_mean_squared_error'`**. Put a **minus sign in front of `cross_val_score`** to flip it back to real RMSE.
- Verified run (`price_usd ~ size_m2 * city_center`, 150 rows, `KFold(n_splits=5, shuffle=True, random_state=9)`): per-fold RMSE **31275.61, 39953.92, 40810.29, 29354.24, 35929.57**, **mean RMSE 35464.73**, and with `scoring='r2'` a **mean R-squared of 0.8690**.

> **The one takeaway:** to cross-validate a formula-defined regression, expand the formula into a patsy design matrix, then feed that matrix plus a `KFold` splitter to `cross_val_score`, remembering that scikit-learn reports error metrics negated so that higher always means better.

---

## Why the tools switch here

The course's regression work up to this point ran through `statsmodels.formula.api.ols`, which is comfortable: you write `price_usd ~ size_m2 * city_center` and it silently builds dummy variables and interaction columns for you, then hands back coefficients and p-values. The problem is that cross-validation, repeatedly holding out one chunk of the data as a test fold while training on the rest, is not something statsmodels automates.

scikit-learn automates it completely. As the lecture puts it, "you just set it up and then it just, boom, it does it." The cost is that scikit-learn does **not** accept formulas. It wants plain numeric arrays.

> **Cross-validation in one paragraph (refresher):** rather than judging a model on the same rows it was trained on (which flatters it), split the data into k roughly equal **folds**. Train on k-1 folds, score on the held-out fold, and repeat k times so every row gets to be test data exactly once. You end up with k scores, and the **average** of those is your estimate of how the model performs on data it has not seen.

## The data

The running dataset is the apartment real estate one from earlier modules: **150 rows**, `size_m2` (float64, size in square meters), `city_center` (bool), and `price_usd` (int64, the target), with **no missing values** (Verified). The instructor waves off the realism ("it's pretty rare to buy apartments instead of renting" and "it's sort of made-up data anyway"), so treat it as synthetic.

```python
import pandas as pd

data_url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(data_url)
```

Verified split: **80 city-center rows, 70 not**.

## Step 1: formula in, design matrix out (`patsy`)

`dmatrices` takes the same formula string you would give statsmodels and returns two arrays: the outcome and the inputs.

```python
from patsy import dmatrices

y_feature, x_features = dmatrices('price_usd ~ size_m2 * city_center', data=data)
type(x_features)
```

Verified: the type is **`patsy.design_info.DesignMatrix`**, and the shapes are **`x_features` (150, 4)** and **`y_feature` (150, 1)**.

## Step 2: read the design matrix

Displaying `x_features` shows what statsmodels normally hides:

```text
DesignMatrix with shape (150, 4)
Intercept  city_center[T.True]  size_m2  size_m2:city_center[T.True]
        1                    0     75.9                          0.0
        1                    1     84.3                         84.3
        1                    0    100.2                          0.0
        1                    1     78.8                         78.8
        1                    0     55.6                          0.0
```

(Verified on patsy 1.0.1. The notebook's saved output is identical except for two extra leading spaces per line, a cosmetic formatting change between patsy versions, not a difference in the numbers.) Column by column:

- **`Intercept`**: always `1`, "because it doesn't matter about the input for the intercept."
- **`city_center[T.True]`**: the dummy for the categorical. **`False` is the reference level**, so this column is `1` for city-center rows and `0` otherwise. With an intercept in the model, a categorical with n levels contributes n-1 such columns.
- **`size_m2`**: the continuous input, passed through as is.
- **`size_m2:city_center[T.True]`**: the **interaction**, literally the product of the two preceding columns. Verified: this column equals `city_center[T.True] * size_m2` for all 150 rows, and it is **0.0 on all 70 non-city-center rows** because "if it's zero times the meters squared, well, it's just gonna be zero."

That matrix, not the formula, is what scikit-learn consumes. (Practical aside from the lecture: when a Jupyter cell prints a very long output, right-click it or use the left-hand gutter to **enable or disable scrolling** for that output.)

> **Sanity check worth knowing:** fitting scikit-learn's `LinearRegression(fit_intercept=False)` on this design matrix reproduces statsmodels' coefficients **exactly**: Intercept 31586.41, city_center[T.True] 92869.53, size_m2 2996.99, interaction 870.57 (Verified). The design-matrix detour changes the plumbing, not the model.

## Step 3: the splitter (`KFold`)

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=9)
kf.get_n_splits()
```

Verified: returns **5**. Three arguments, all deliberate:

- **`n_splits=5`**: five folds, so with 150 rows each fold trains on **120 rows and tests on 30** (Verified).
- **`shuffle=True`**: "crucially, we're gonna be instructed to shuffle the data." Without shuffling the test folds are contiguous blocks of rows (Verified: rows 0-29, 30-59, 60-89, 90-119, 120-149), so any ordering in the file leaks straight into the split.
- **`random_state=9`**: the seed, "whatever number you want," which makes the shuffle reproducible.

**Note that no data has been passed yet.** `kf` only describes *how* to split.

The lecture mentions an easier route "without having to define k-fold, but it doesn't shuffle things" and then moves on without showing it. That easier route is passing an integer straight to `cross_val_score` as `cv=5`. Verified: for a regressor that is exactly `KFold(n_splits=5)` with **`shuffle=False`** (identical fold scores), and on this data it gives a **mean RMSE of 36113.69** versus **35464.73** for the shuffled version. Same model, different splits, different estimate.

## Step 4: the estimator and `cross_val_score`

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

clf = LinearRegression(fit_intercept=False)
scores = cross_val_score(clf, x_features, y_feature.ravel(), cv=kf,
                         scoring='neg_root_mean_squared_error')
```

The arguments, in the order the lecture walks them:

1. **`clf`**, the **untrained** estimator. `cross_val_score` clones it and fits the clones once per fold, so you never call `.fit()` yourself and `clf` itself is still unfitted afterwards (Verified: it has no `coef_` attribute when the call returns).
2. **`x_features`**, the design matrix.
3. **`y_feature.ravel()`**, the target. `patsy` hands back a 2-D `(150, 1)` array and scikit-learn wants 1-D, so `.ravel()` flattens it to `(150,)` (Verified; see the caveat in *Three places the lecture is loose*, below).
4. **`cv=kf`**, the splitter object, which supplies the shuffling and the five splits.
5. **`scoring=...`**, the metric name. scikit-learn ships **56 named scorers** in version 1.7.2 (Verified); the lecture uses `'neg_root_mean_squared_error'` and later `'r2'`.

**Why `fit_intercept=False`:** the design matrix already has an `Intercept` column of ones, so scikit-learn must not add a second one. Verified detail: with `fit_intercept=False` the intercept lands in **`coef_[0] = 31586.41`** and `intercept_` is `0.0`; with `fit_intercept=True` it lands in **`intercept_ = 31586.41`** and `coef_[0]` becomes `0.0`. Predictions and cross-validated scores come out identical either way on this matrix, which is why the lecture calls it "kind of just a detail," but the `False` setting is the one that keeps `coef_` lined up with the design matrix columns.

## Reading the scores, and the minus sign

```python
scores
```

Verified output:

```text
array([-31275.60510053, -39953.91989162, -40810.29391583, -29354.23882684,
       -35929.56993949])
```

A NumPy array of five values, one per test fold, from **five separately trained linear regression models**, each of which held out a different fold. All of that "in seconds."

They are negative because **scikit-learn's convention is that higher is always better**, and lower error is better, so error metrics are exposed negated. To get real RMSE, negate the whole call:

```python
scores = -cross_val_score(clf, x_features, y_feature.ravel(), cv=kf,
                          scoring='neg_root_mean_squared_error')
```

Verified:

```text
array([31275.60510053, 39953.91989162, 40810.29391583, 29354.23882684,
       35929.56993949])
```

## The number you actually report: the mean

"We usually don't care that much about the individual folds though with cross validation. We want the average, we want the mean."

```python
scores.mean()
```

Verified: **35464.7255**, that is, an average out-of-sample error of about **\$35,465** on a target whose values run in the hundreds of thousands. Because it is a plain NumPy array, `.mean()` is all you need.

## Swapping the metric: R-squared

R-squared is a goodness-of-fit measure where higher is better already, so **no negation is needed** and no `neg_` prefix exists:

```python
cross_val_score(clf, x_features, y_feature.ravel(), cv=kf, scoring='r2').mean()
```

Verified: **0.8689691581676767**, from per-fold R-squared values of **0.9102, 0.8133, 0.8421, 0.9045, 0.8747**. That is the cross-validated average, not an in-sample number: fitting the same model on all 150 rows at once gives a slightly rosier **in-sample R-squared of 0.8785** (Verified), which is exactly the optimism cross-validation exists to strip out. The lecture's verdict: "it's a pretty good model."

## Three places the lecture is loose

> **1. The R-squared it reads out.** The lecture says "it's out of from zero to one, we're at 0.86 R squared. So pretty good correlation." Three small slips. The value on screen is **0.8690**, which rounds to **0.87** (truncated, not rounded). R-squared is the **proportion of variance explained**, not a correlation coefficient. And a **cross-validated** R-squared is not bounded below by zero: scikit-learn computes it as `1 - SS_res/SS_tot` on each held-out fold, so a bad model scores negative (Verified: a noise-only model on this data returns fold scores near -7 to -12). None of this changes the verdict that the model is good. If a quiz keys to the course's spoken wording, answer its way and say 0.86.

> **2. "Classifier."** The lecture calls the model "a linear regression classifier" and the notebook names the variable `clf`. **`LinearRegression` is a regressor, not a classifier**: classifiers predict discrete labels, regressors predict continuous numbers. `clf` is just a habitual scikit-learn variable name here, and `cross_val_score` takes any estimator. If the exam uses the course's word, answer its way, but know the distinction.

> **3. Why `.ravel()`.** The lecture says scikit-learn "likes it in a 1D array" and that `.ravel()` "just makes it work." Verified: on this model, passing the raw 2-D `(150, 1)` array produces **identical scores with no warning** in scikit-learn 1.7.2. It is not always harmless, though: the same 2-D target passed to `RandomForestRegressor` raises a **`DataConversionWarning`** (Verified). Keep using `.ravel()`, since 1-D is the shape scikit-learn documents for a single-target problem.

## How much does the seed matter?

The lecture shrugs at `random_state` ("whatever number you want"), which is fine for reproducibility but hides real variance. Verified across **30 seeds** (`random_state` 0 through 29, everything else unchanged): mean RMSE ranges from **35465 to 36733** and mean R-squared from **0.8467 to 0.8733**. The seed the course uses, `random_state=9`, gives the **lowest mean RMSE of all 30** (Verified; its mean R-squared ranks 5th of 30, so it is a flattering draw rather than a typical one). Quote the seeded number as the course's number, but do not read meaning into its last digits. For a stabler estimate you would repeat the whole cross-validation over several seeds and average.

---

## Quiz-ready facts

- **statsmodels** gives formulas, coefficients, and p-values but little cross-validation support; **scikit-learn** gives cross-validation but takes **no formulas**. The bridge is a **design matrix**.
- A **design matrix** (feature array) has **one column per estimated coefficient**: the intercept column of ones, one dummy column per non-reference categorical level, the continuous columns, and one column per interaction (literally the product of the involved columns).
- **`from patsy import dmatrices`**, then **`y_feature, x_features = dmatrices(formula, data=data)`**. Outcome comes back **first**, inputs second. `type(x_features)` is **`patsy.design_info.DesignMatrix`**; here its shape is **(150, 4)** (Verified).
- **`KFold(n_splits=5, shuffle=True, random_state=9)`** defines the split only; **no data is passed at construction**. `kf.get_n_splits()` returns **5**. With 150 rows that is **120 train / 30 test per fold** (Verified).
- **`KFold` defaults to `shuffle=False`**, and for a regressor a bare **`cv=5`** integer passed to `cross_val_score` is that same unshuffled split, with contiguous blocks of rows as test folds (Verified). Shuffling is the whole reason the lecture builds a `KFold` object at all.
- **`LinearRegression(fit_intercept=False)`** because the design matrix already contains the intercept column. With `False`, the intercept appears as **`coef_[0]`**; with `True` it appears as `intercept_` (Verified, same predictions here).
- **`cross_val_score(estimator, X, y, cv=..., scoring=...)`** takes an **untrained** estimator, fits and scores it once per fold, and returns a **NumPy array** of one score per fold.
- Give the target as **1-D**: `y_feature.ravel()` turns `patsy`'s `(150, 1)` output into `(150,)`. `LinearRegression` happens to tolerate the 2-D version, but other estimators warn (Verified).
- **All scikit-learn scorers are "higher is better."** Error metrics therefore carry a **`neg_` prefix**: **`'neg_root_mean_squared_error'`**. Multiply by -1 (a minus sign in front of `cross_val_score`) to recover RMSE. **`'r2'` needs no negation** because higher R-squared is already better.
- **Verified results** for `price_usd ~ size_m2 * city_center`, `KFold(5, shuffle=True, random_state=9)`, `fit_intercept=False`: per-fold RMSE **31275.61, 39953.92, 40810.29, 29354.24, 35929.57**; **mean RMSE 35464.73**; per-fold R-squared **0.9102, 0.8133, 0.8421, 0.9045, 0.8747**; **mean R-squared 0.8690**.
- Report the **mean across folds**, not individual fold scores. Cross-validated R-squared (**0.8690**) is below the in-sample R-squared (**0.8785**), as expected (Verified).
- A **cross-validated R-squared is not capped at zero from below**. It is `1 - SS_res/SS_tot` computed on held-out rows, so a bad model returns **negative** scores (Verified). Only the in-sample R-squared of a model with an intercept is confined to 0 through 1.
- The scikit-learn fit on the design matrix reproduces **statsmodels' coefficients exactly**: 31586.41, 92869.53, 2996.99, 870.57 (Verified).
- Terminology trap: the lecture and notebook call the model a **"classifier"** and name it `clf`, but `LinearRegression` is a **regressor**.

---

> **See also:** "Cross-Validation with Scikit-learn - Notebook (Cliff Notes)" (the companion notebook this lecture narrates cell by cell), "Introduction to Cross-Validation (Cliff Notes)" (why folds and held-out testing exist at all, the concept this note operationalizes), "Comparing Models with Cross-Validation (Cliff Notes)" (using these mean scores to pick among competing formulas, balanced against a preference for simpler models), "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" (what the error metric being averaged here actually measures), and "Scikit-learn Pipeline Cross-Validation - Notebook (Cliff Notes)" (which keeps this exact `patsy` plus `KFold` plus `cross_val_score` recipe and slots a `make_pipeline(StandardScaler(), LinearRegression())` in where `clf` sits, so data-dependent preprocessing is refit inside each training fold).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
