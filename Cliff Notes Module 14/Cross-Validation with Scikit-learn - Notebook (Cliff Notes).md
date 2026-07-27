# Cross-Validation with Scikit-learn (Lab Notebook)

**Module 14 Cliff Notes** | Source: lab notebook `cross-validation_with_scikit-learn_start.ipynb`

---

## TL;DR

- **Cross-validation** estimates how well a model does on data it has not seen: split the rows into *k* folds, train on *k-1* of them, score on the held-out one, repeat so every row gets held out exactly once, then average the scores.
- statsmodels fits formula models nicely but **has almost no cross-validation tooling**, so this lab hands the job to **scikit-learn**, which cannot read formulas. The bridge is **`patsy.dmatrices`**, which turns a formula plus a DataFrame into **design matrices**: plain numeric arrays scikit-learn does understand.
- A **design matrix** has one column per *feature that gets a coefficient*, not one per raw variable: dummy columns for categorical levels (all but the reference) and separate columns for interaction terms. Here `price_usd ~ size_m2 * city_center` yields a **(150, 4)** matrix with columns `Intercept`, `city_center[T.True]`, `size_m2`, `size_m2:city_center[T.True]` (Verified).
- Splitter: **`KFold(n_splits=5, shuffle=True, random_state=9)`**. Shuffling plus a fixed seed makes the folds reproducible; each fold trains on **120** rows and tests on **30** (Verified).
- Scorer: **`cross_val_score(...)`** returns **one score per fold**. scikit-learn's rule is *higher is always better*, so error metrics come back **negated**; put a `-` in front of the call to read ordinary RMSE.
- Verified RMSE per fold: **31275.61, 39953.92, 40810.29, 29354.24, 35929.57**, mean **35464.73** dollars of prediction error. Swap in `scoring='r2'` and the mean **R-squared is 0.8690** (Verified).
- Two setup details do the plumbing: the design matrix already has an `Intercept` column, so the estimator is **`LinearRegression(fit_intercept=False)`**, and patsy's 2-D target is flattened with **`.ravel()`**.

> **The one takeaway:** to cross-validate a formula model, use `patsy.dmatrices` to freeze the formula into a numeric design matrix, hand that matrix to `cross_val_score` with a seeded `KFold`, remember that scikit-learn negates error scorers, and report the **mean across folds** as your generalization estimate.

---

## Why this lab leaves statsmodels

Everything up to here fitted models with `smf.ols('y ~ x1 * x2', data=df).fit()`, because the **formula interface** is compact and readable. The catch: statsmodels does not ship the cross-validation machinery (splitters, scorers, a loop that manages folds). scikit-learn does, but scikit-learn's estimators take **arrays in, arrays out**; they have no idea what a formula string means.

So the lab splits the work in two:

- **`patsy`** turns the formula into features. It is the same engine statsmodels already uses to read formulas, so nothing about the model changes (Verified: `smf.ols(...).fit().model.exog` is **bit-identical** to the matrix `dmatrices` hands back, same four column names, same values).
- **`scikit-learn`** splits rows into folds, fits, and scores. It owns `KFold`, `cross_val_score`, and the scorer catalog.

## Load the data

Same 150-row apartment dataset used in prior modules, read straight from a URL:

```python
import pandas as pd

data_url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(data_url)
data.info()
```

Verified: **150 rows, 3 columns, no missing values**, with `size_m2` `float64` (continuous), `city_center` `bool` (categorical), and `price_usd` `int64` (the target). The split is **80 city-center rows and 70 not** (Verified), and mean price is about **\$301,927** (Verified), which is the yardstick to hold the RMSE against later.

## Step 1: build the design matrix with patsy

```python
from patsy import dmatrices

y_feature, x_features = dmatrices('price_usd ~ size_m2 * city_center', data=data)
```

`dmatrices` returns the **outcome matrix first, the feature matrix second**. Verified shapes: `y_feature` is **(150, 1)** and `x_features` is **(150, 4)**. Its type prints as `patsy.design_info.DesignMatrix`, which is a thin subclass of `numpy.ndarray` (Verified: `isinstance(x_features, np.ndarray)` is `True`), so scikit-learn accepts it directly with no conversion.

The four columns are exactly the terms the formula implies:

```text
Intercept  city_center[T.True]  size_m2  size_m2:city_center[T.True]
        1                    0     75.9                          0.0
        1                    1     84.3                         84.3
        1                    0    100.2                          0.0
        1                    1     78.8                         78.8
        1                    0     55.6                          0.0
```

Read the row structure and the whole design-matrix idea falls out:

- **`Intercept`** is a hard-coded column of 1s, which patsy adds by default. (Suppressing it with `- 1` or `+ 0` is not just a column deletion: patsy compensates by switching the categorical to full dummy coding, `city_center[False]` **and** `city_center[True]`, Verified.)
- **`city_center[T.True]`** is the **dummy** for the categorical input: `False` is the dropped reference level, so the column is 1 only for city-center rows.
- **`size_m2`** is the raw continuous value, copied through.
- **`size_m2:city_center[T.True]`** is the **interaction**, literally the product of the previous two columns, so it is `size_m2` for city-center rows and `0.0` everywhere else.

That last column is the point of the exercise: an interaction is not a modeling instruction that scikit-learn has to understand, it is **just another numeric column**. Once patsy materializes it, any array-eating estimator can fit it.

> **Sanity check:** fitting `LinearRegression(fit_intercept=False)` on this matrix over all 150 rows reproduces the statsmodels formula fit to within 4.4e-11: **Intercept 31586.41, city_center[T.True] 92869.53, size_m2 2996.99, interaction 870.57** (Verified). Same model, different steering wheel.

## Step 2: define the fold splitter with KFold

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=9)
kf.get_n_splits()   # 5
```

- **`n_splits=5`** is *k*: five folds, so with 150 rows each fold tests on **30** and trains on **120** (Verified).
- **`shuffle=True`** randomizes row order before slicing. Without it, `KFold` cuts the data in the order it arrived, which is a real hazard if the file is sorted by anything meaningful.
- **`random_state=9`** freezes that shuffle so the folds are identical on every rerun (Verified: two back-to-back `cross_val_score` calls return bit-identical arrays).

> **Gotcha:** `random_state` only makes sense with `shuffle=True`. Set a seed with `shuffle=False` and scikit-learn raises `ValueError: Setting a random_state has no effect since shuffle is False` (Verified).

## Step 3: score every fold with `cross_val_score`

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

clf = LinearRegression(fit_intercept=False)
scores = cross_val_score(clf, x_features, y_feature.ravel(), cv=kf,
                         scoring='neg_root_mean_squared_error')
```

`cross_val_score` runs the whole loop: for each of the 5 folds it **clones** the estimator, fits the clone on the 120 training rows, scores it on the 30 held-out rows, and collects the numbers. Verified output:

```text
array([-31275.60510053, -39953.91989162, -40810.29391583, -29354.23882684,
       -35929.56993949])
```

Note the clone: after the call, `clf` itself is still **unfitted** (Verified: `hasattr(clf, 'coef_')` is `False`). `cross_val_score` gives you scores, never a trained model.

## The minus sign: why the scores come back negative

scikit-learn's scorer catalog is built on one convention: **a higher score is always better**. RMSE is an error, so lower is better, which would break the rule. The fix is to expose it **negated**, as `'neg_root_mean_squared_error'`. Flip the sign to read it the way a human does:

```python
scores = -cross_val_score(clf, x_features, y_feature.ravel(), cv=kf,
                          scoring='neg_root_mean_squared_error')
```

Verified:

```text
array([31275.60510053, 39953.91989162, 40810.29391583, 29354.23882684,
       35929.56993949])
```

## Step 4: average the folds

Individual fold scores are noisy (here they range from **29354.24** to **40810.29**, a spread of **11456.06**, Verified). The generalization estimate is the **mean**:

```python
scores.mean()   # 35464.73  (Verified)
```

So this model is off by roughly **\$35.5k** on unseen apartments, about **11.7%** of the mean price (Verified). For comparison, refitting on all 150 rows and scoring on those same rows gives an in-sample RMSE of **35290.12** (Verified), a bit rosier than the cross-validated number, which is exactly the optimism cross-validation exists to strip out.

> **On reproducing the saved outputs:** the per-fold arrays match the notebook's stored output digit for digit, but the two aggregates drift in their last digits. The notebook's saved mean prints `35464.72553486105` and its saved R-squared prints `0.8689691581676767`; a fresh run here gives `35464.72553486111` and `0.8689691581676765` (Verified). The two agree to at least **14 significant figures** (relative difference about 1.6e-15 and 2.6e-16, Verified), which is floating-point noise from library builds, not a different model. Quote these to a couple of decimals and the drift disappears.

## Swapping the scorer: R-squared

Any name from scikit-learn's scoring catalog (the notebook links its "scoring functions" documentation) can go in the `scoring` argument. For R-squared it is `'r2'`:

```python
cross_val_score(clf, x_features, y_feature.ravel(), cv=kf, scoring='r2').mean()
# 0.8690  (Verified)
```

Per-fold R-squared: **0.9102, 0.8133, 0.8421, 0.9045, 0.8747** (Verified). Notice this one is **not** negated: R-squared is already "higher is better", so it needs no sign flip. Misspell the scorer and you get an `InvalidParameterError` that lists the valid names (Verified), which doubles as a quick way to look up the catalog.

## Two setup details that are easy to get wrong

**`fit_intercept=False`.** patsy already put a column of 1s in the matrix, so scikit-learn must not add its own. If you leave the default `fit_intercept=True`, nothing visibly breaks: scikit-learn centers the data first, the constant column collapses to zeros, and the predictions come out the same to about **3e-09 dollars** (Verified), which is floating-point dust against a \$300k scale, and the cross-validated mean RMSE is **35464.73** either way (Verified). What you lose is the clean column-to-coefficient mapping: `coef_[0]` comes back as **0.0** and the real intercept, **31586.41**, moves into `intercept_` (Verified). The scores survive, the interpretability does not.

**`.ravel()`.** patsy hands back a **(150, 1)** column vector; scikit-learn's usual expectation for a single target is a flat **(150,)** array, which `y_feature.ravel()` provides (Verified shape change).

## Where the notebook's prose is loose

The lab is short and its code is correct; a few of its sentences are not. Course wording quoted first, correction after. **Exam safety, for all four:** these are the notebook's words, so if a quiz keys to its phrasing, answer the course's way and keep the correction for real work.

- **"We'll pass this as the classifier"** (and the variable name **`clf`**). `LinearRegression` is a **regressor**, not a classifier: it predicts a continuous number, not a class label. `clf` is scikit-learn's conventional name for a classifier; the neutral names are `est`, `reg`, or `model`. Nothing breaks, but the label is wrong.
- **"Scoring options in Scikit-learn all are set to mean higher scores are better, so they use a negative RMSE."** The first half is right, the second over-generalizes: **higher-is-better applies to every scorer, but only error metrics get negated**. The notebook proves this itself a few cells later, when `'r2'` needs no minus sign. Read the sentence as "error scorers are negated", not "all scorers are negative".
- **"Scikit-learn expects a 1-dimensional output array"** (the sentence continues "so we'll use `.ravel()`"). True as a habit, loose as a rule. `LinearRegression` accepts the (150, 1) column with no error and no warning (Verified: the same five fold scores either way), because linear regression supports multi-output targets. Other estimators are stricter, and the same column vector makes `SVR` raise a `DataConversionWarning` (Verified). Use `.ravel()` anyway, just know why.
- The markdown calls the target **`y_features`** but the code assigns **`y_feature`** (singular). Trivial, but it will bite you if you retype the cells from the prose instead of copying them.

> **One more subtlety the lab does not raise:** `scores.mean()` is the **mean of five fold RMSEs**, which is not the same as pooling every held-out prediction and computing one RMSE. Pooled here gives **35756.30** versus **35464.73** for the mean of folds (Verified). With equal-sized folds the pooled value is exactly `sqrt(mean of the fold MSEs)` (Verified), and because the square root is concave, **the mean of fold RMSEs is always the smaller of the two**. Averaging fold scores is the standard practice and what `cross_val_score` gives you, but the two are different statistics, so do not mix them when comparing models.

---

## Quiz-ready facts

- **Why scikit-learn:** statsmodels has the formula interface but not the cross-validation tooling; scikit-learn has the tooling but no formula parser. **`patsy` bridges them by producing design matrices.**
- **Design matrix definition:** one column per **feature that receives a coefficient**, including **dummy columns** for categorical levels (reference level excluded) and **separate columns for interaction terms**.
- **`dmatrices(formula, data=df)` returns `(y, X)`**, outcome first. For `price_usd ~ size_m2 * city_center`: `y` is **(150, 1)**, `X` is **(150, 4)** with columns `Intercept`, `city_center[T.True]`, `size_m2`, `size_m2:city_center[T.True]` (Verified). Type is `patsy.design_info.DesignMatrix`, a `numpy.ndarray` subclass.
- The interaction column is literally the **product** of `size_m2` and the `city_center[T.True]` dummy, so it equals `size_m2` for city-center rows and `0.0` otherwise.
- **`KFold(n_splits=5, shuffle=True, random_state=9)`**: 5 folds, shuffled, reproducible. `kf.get_n_splits()` returns **5**; each fold is **120 train / 30 test** on 150 rows (Verified). A `random_state` with `shuffle=False` raises `ValueError`.
- **`cross_val_score(estimator, X, y, cv=kf, scoring=...)`** returns **one score per fold** (an array of length 5) and **clones** the estimator each fold, so the object you passed in is never left fitted (Verified).
- **Sign convention:** every scikit-learn scorer is "higher is better", so **error metrics are exposed negated** (`'neg_root_mean_squared_error'`, `'neg_mean_squared_error'`). Prefix the call with `-` to recover the plain error. `'r2'` is **not** negated.
- **Verified RMSE folds:** 31275.61, 39953.92, 40810.29, 29354.24, 35929.57; **mean 35464.73** (about 11.7% of the \$301,927 mean price). **Verified mean R-squared with `scoring='r2'`: 0.8690** (folds 0.9102, 0.8133, 0.8421, 0.9045, 0.8747).
- **Report the mean, not a single fold.** Fold RMSEs here span 29354.24 to 40810.29, a spread of 11456.06 (Verified); one fold on its own is noise.
- **`fit_intercept=False`** because patsy's `Intercept` column already supplies the constant. Leaving it `True` leaves predictions and the mean score effectively unchanged (Verified, 35464.73 either way) but zeroes out `coef_[0]` and hides the real intercept in `intercept_`.
- **`.ravel()`** flattens patsy's (150, 1) target to (150,), the shape scikit-learn expects for a single target.
- Cross-validated RMSE (**35464.73**) is worse than in-sample RMSE (**35290.12**, Verified). Held-out error being higher than training error is the normal, expected direction.

---

> **See also:** "Cross-Validation with Scikit-learn (Cliff Notes)" (the companion lecture note this lab implements, for the narrated version of the patsy-to-scikit-learn handoff), "Introduction to Cross-Validation (Cliff Notes)" (what k-fold is and why a single train/test split is not enough), "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" (what the RMSE numbers above actually measure and why they are in dollars), "Comparing Models with Cross-Validation - Notebook (Cliff Notes)" (using these same mean scores to choose between competing formulas), and "Scikit-learn Pipeline Cross-Validation - Notebook (Cliff Notes)" (wrapping preprocessing into the estimator so it is refit inside every fold).

---

*Source: CMPIF2100 lab notebook (personal study use).*
