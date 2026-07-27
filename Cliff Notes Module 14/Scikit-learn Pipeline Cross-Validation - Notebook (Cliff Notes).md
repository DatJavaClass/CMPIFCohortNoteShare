# Scikit-learn Pipeline for Preprocessing with Cross-Validation (Lecture Notebook)

**Module 14 Cliff Notes** | Source: lecture notebook `Module-14_Scikit-learn_pipeline_cross-validation.ipynb`

---

## TL;DR

- **Data-dependent preprocessing** is any transform whose settings are *estimated from the data*. z-standardization (subtract the mean, divide by the standard deviation, giving mean 0 and sd 1) is the case here. Converting grams to kilograms by dividing by 1000 is **not** data-dependent, because 1000 was not estimated from anything.
- Standardizing with statistics computed from the **whole** dataset and *then* cross-validating means the model has effectively "seen" the test folds, which can **overestimate performance on unseen data**. The fix is to estimate the scaler on the **training folds only** and apply those same training-fold statistics to the test fold.
- A scikit-learn **`Pipeline`** chains preprocessing and modeling into one estimator. Pass the pipeline to `cross_val_score` and every step is re-fit **inside each training fold** automatically. Built here with `make_pipeline(StandardScaler(), LinearRegression())`.
- Worked model: predict `flipper_length_mm` from `bill_length_mm + body_mass_g` on Palmer Penguins, 5-fold shuffled CV with `random_state=9`. Verified mean RMSE **6.5350 mm**; fold RMSEs **7.1495, 6.6674, 6.4717, 6.5710, 5.8154**.
- `scoring='neg_root_mean_squared_error'` returns **negative** RMSE (scikit-learn scorers are always higher-is-better), so a leading **minus sign** on `cross_val_score` flips them back to normal RMSE.
- The scaler transforms **only the input features**, never `y`, so the RMSE stays in the target's original units (millimetres).

> **The one takeaway:** put every data-dependent preprocessing step and the model into one `Pipeline`, then hand the pipeline (not the bare model) to `cross_val_score`, so each fold's preprocessing is estimated from that fold's training data alone and the reported score is an honest estimate of performance on unseen data.

---

## Why data-dependent preprocessing breaks cross-validation

Cross-validation exists to simulate unseen data: hold out a fold, train on the rest, score on the fold, repeat. That simulation only works if the held-out fold is genuinely unseen, and **estimating a mean and standard deviation counts as seeing it**.

The notebook spells out the correct per-run recipe:

1. Estimate the standardization mean and sd on the **training folds**.
2. Train the model on the training folds.
3. Standardize the **test fold** using the training-fold mean and sd.
4. Score the model on that standardized test fold.

That is the same discipline already applied to coefficients (fit on training folds only), extended to the preprocessing. Doing it by hand for five folds is tedious, and a `Pipeline` does it for you.

## The dataset and the scale problem

```python
import seaborn as sns
penguins = sns.load_dataset('penguins')
penguins.info()
```

Verified: **344 rows, 7 columns** (4 float64, 3 object), with missing values in the measurement columns (2 each) and `sex` (11). The columns live on wildly different scales, which is what motivates standardizing in the first place (Verified, per-column summary):

```text
                   mean      std      min      max
bill_length_mm    43.92     5.46     32.1     59.6
bill_depth_mm     17.15     1.97     13.1     21.5
flipper_length_mm 200.92   14.06    172.0    231.0
body_mass_g      4201.75  801.95   2700.0   6300.0
```

The notebook draws this with `sns.catplot(data=penguins, kind='box', aspect=2)`, which boxes all four numeric columns on one shared y axis. Verified: that axis spans roughly **-301 to 6614** in order to fit `body_mass_g`, so the three millimetre columns (largest value 231) flatten into a smear along the bottom. That single picture is the argument for standardizing.

## Step 1: build the design matrix with patsy

To use a statsmodels-style formula with scikit-learn, convert the formula into plain numeric arrays first:

```python
from patsy import dmatrices
y_feature, x_features = dmatrices('flipper_length_mm ~ bill_length_mm + body_mass_g', data=penguins)
```

Verified: `y_feature` is **(342, 1)** and `x_features` is **(342, 3)** with columns **`['Intercept', 'bill_length_mm', 'body_mass_g']`**. Two things worth noticing:

- **342, not 344.** `dmatrices` drops rows with missing values in any modeled column, and exactly **2 rows** have an NA among these three (Verified).
- **patsy adds its own `Intercept` column of 1s.** `StandardScaler` sees zero variance there, so its guard sets that column's `scale_` to **1.0** and the centered column becomes **all zeros** (Verified). Nothing breaks, because `LinearRegression` supplies its own intercept (`fit_intercept=True` by default) and simply fits a coefficient of exactly **0.0** for the dead column (Verified).

## Step 2: make the pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
```

`make_pipeline` takes the steps in order and names them automatically from the lowercased class names, so `pipeline.named_steps` has keys **`'standardscaler'` and `'linearregression'`** (Verified). Every step **except the last** must be a transformer implementing `fit` and `transform`; putting a model in a middle slot raises `TypeError: All intermediate steps should be transformers...` (Verified). The final step is the estimator, here the model, so the whole pipeline is itself an estimator with `.fit()` and `.predict()`, which is exactly why it can stand in for a model anywhere a model is expected.

> **Variant:** `Pipeline([('scale', StandardScaler()), ('lm', LinearRegression())])` is the explicit form where you choose the step names. Verified: identical CV mean (6.5350098073) to the `make_pipeline` version.

## Step 3: the KFold splitter

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=9)
kf.get_n_splits()
```

Verified output: **5**. Fold test sizes are **69, 69, 68, 68, 68** (summing to 342), train sizes **273, 273, 274, 274, 274**.

`shuffle=True` is not decoration. `KFold` defaults to `shuffle=False` and slices the data in row order, and the penguins file arrives in **three contiguous species blocks** (Adelie, then Chinstrap, then Gentoo; Verified), so unshuffled folds are badly unrepresentative. Verified contrast on the identical pipeline: `cv=5` unshuffled gives mean RMSE **7.4763**, while the shuffled `kf` gives **6.5350**. `random_state=9` fixes the shuffle so the result is reproducible.

## Step 4: cross-validate the pipeline

```python
from sklearn.model_selection import cross_val_score
scores = -cross_val_score(pipeline, x_features, y_feature.ravel(),
                          cv=kf, scoring='neg_root_mean_squared_error')
scores.mean()
```

Verified: **6.535009807315251**. The notebook's saved output reads 6.5350098073152525, a difference of **1.8e-15**, which is floating-point summation noise rather than any real disagreement. The five fold scores (Verified):

```text
[7.14948861  6.66740392  6.47172500  6.57103407  5.81539744]
scores.mean() = 6.5350     scores.std() = 0.4285
```

Three details in that one call:

- **The leading minus.** Without it the raw values are **negative** (`-7.1495, -6.6674, ...`, Verified). scikit-learn's scorer convention is higher-is-better, so error metrics ship as `neg_*` and you flip the sign yourself.
- **`.ravel()`** flattens the (342, 1) `y_feature` into a 1-D array of length 342, the shape scikit-learn expects for single-target regression. (In scikit-learn 1.7.2 the column vector happens to work too and gives the same mean with no warning, but `ravel()` is the safe habit.)
- **`y` is never scaled.** `StandardScaler` sits on the feature side of the pipeline only, so the RMSE comes back in millimetres and stays directly interpretable. The notebook flags this explicitly, and it is correct.

> **Terminology fix:** the notebook says to pass the pipeline "for the argument that usually takes the classifier," and its summary repeats "Pass the pipeline as the classifier into `cross_val_score`." `LinearRegression` is a **regressor**, not a classifier, and the parameter is actually named **`estimator`** (the first positional argument of `cross_val_score`, Verified against the 1.7.2 signature). "Estimator" is scikit-learn's umbrella term covering classifiers, regressors, transformers, and pipelines, which is exactly why a `Pipeline` fits in that slot. If a quiz keys to the course's wording, "pass the pipeline where the model/classifier goes" is the answer it wants.

## What the pipeline actually did

The pipeline is not magic, just bookkeeping. This hand-written loop reproduces its five scores **exactly** (Verified, maximum difference 0.0):

```python
from sklearn.metrics import root_mean_squared_error
y = y_feature.ravel()
manual = []
for train_idx, test_idx in kf.split(x_features):
    sc = StandardScaler().fit(x_features[train_idx])                 # 1. estimate on TRAIN only
    m = LinearRegression().fit(sc.transform(x_features[train_idx]),
                               y[train_idx])                         # 2. train on TRAIN only
    pred = m.predict(sc.transform(x_features[test_idx]))             # 3. TRAIN stats on TEST fold
    manual.append(root_mean_squared_error(y[test_idx], pred))        # 4. score
```

Note the scaler is **re-created inside the loop**: five folds means five separately estimated means and standard deviations, and each is used to transform its own test fold.

## Honest caveat: on this model the pipeline changes nothing

The notebook warns that fitting the scaler on everything gives the model "an unrealistic advantage" and "may overestimate performance." As a habit that is right, but on **this** model it makes no measurable difference, and knowing why is worth more than the rule.

Ordinary least squares is unchanged by linear rescaling of its inputs: shifting and scaling a column changes that column's coefficient but not the fitted values. So standardizing cannot move an OLS prediction no matter which rows the mean and standard deviation came from. All three versions agree (Verified, to about 1e-14):

```text
pipeline, scaler re-fit per training fold : 6.5350098073
LinearRegression alone, no scaler at all  : 6.5350098073
scaler fit on ALL the data, then CV       : 6.5350098073
```

What standardizing *does* change is the coefficients' units (Verified on the full 342 rows):

```text
standardized : Intercept 200.9152, bill_length 2.9941, body_mass 10.4508
raw units    : Intercept 121.9560, bill_length  0.5492, body_mass  0.0131
    2.9941 / 5.4516 (scale_ of bill_length) = 0.5492
   10.4508 / 800.78 (scale_ of body_mass)   = 0.0131
```

Same predictions, rescaled coefficients: each standardized coefficient divided by that column's `scale_` returns the raw-units coefficient (Verified). (The standardized intercept lands on **200.9152**, exactly the mean flipper length, as it must once every input is centered. Note `scale_` is the **population** sd, ddof=0, so 800.78 here versus the **sample** sd 801.95 that pandas `describe()` prints above.)

Where the scaler genuinely matters is **models that measure distances or apply penalties**. Verified on the same folds:

- `KNeighborsRegressor(n_neighbors=5)`: mean RMSE **6.3432** scaled versus **6.8647** unscaled, because unscaled `body_mass_g` (sd about 801) drowns out `bill_length_mm` (sd about 5.5) in the distance calculation.
- `Ridge(alpha=100)`: mean RMSE **7.0969** scaled versus **6.5347** unscaled, because the penalty shrinks coefficients by size and the raw-unit coefficients are not comparable to each other.

Either way the point stands: the moment the model or the preprocessing is scale-sensitive, *where* you fit the scaler starts to matter, and the pipeline is what makes the answer right by construction.

> **Exam note:** if a quiz asks why `StandardScaler` goes inside a pipeline, answer the course's way: so the standardization is estimated on the training folds only and performance on unseen data is not overestimated. The nuance above is about *how much* it moves this particular number, not about whether the practice is correct.

## When leakage is not harmless (invented example)

Standardization leaks gently. Feature selection leaks catastrophically. Same pipeline idea, a case built to make the damage visible:

```python
from sklearn.feature_selection import SelectKBest, f_regression

rng = np.random.default_rng(0)
Xn, yn = rng.normal(size=(60, 400)), rng.normal(size=60)   # pure noise: nothing is predictable

# WRONG: pick the 5 "best" columns using ALL the data, then cross-validate
Xn_sel = SelectKBest(f_regression, k=5).fit_transform(Xn, yn)
cross_val_score(LinearRegression(), Xn_sel, yn, cv=kf, scoring='r2')

# RIGHT: selection inside the pipeline, redone on each training fold
cross_val_score(make_pipeline(SelectKBest(f_regression, k=5), LinearRegression()),
                Xn, yn, cv=kf, scoring='r2')
```

Verified across 30 seeds: the leaky version averages **R-squared = 0.135** (range -0.143 to 0.366), the pipeline version averages **-0.636** (range -1.355 to 0.043), and the leaky version scores higher in **30 of 30** runs. Since the data is pure noise, the true out-of-sample R-squared is **0 at best**, so the honest, mostly negative estimates are the correct ones. Choosing features outside the CV loop manufactured skill that does not exist; the pipeline refused to.

## Reading the result

The notebook concludes the model is "about 6 mm off on average." Two refinements:

- RMSE is the root of the **mean squared** error, so it is not literally the average error. It punishes large misses harder and is always greater than or equal to MAE. On these same folds, MAE = **5.334 mm** versus RMSE = **6.535 mm** (Verified).
- For scale: flipper length's own standard deviation is **14.06 mm**, so predicting the mean every time would score about that. The model's **6.535 mm** is roughly **46%** of it (Verified), consistent with its mean cross-validated R-squared of **0.773** (Verified), whose implied error ratio is `sqrt(1 - 0.773) = 0.477`.

(Exam note: "about 6 mm off on average" is a fine plain-English gloss of RMSE and is what a quiz keyed to this notebook will want. Just do not equate RMSE with mean absolute error on a definition question.)

---

## Quiz-ready facts

- **Data-dependent preprocessing** estimates its parameters from data (z-standardization estimates a mean and sd). Dividing by a fixed constant (grams to kilograms) is **not** data-dependent and needs no special CV handling.
- The danger: estimating standardization parameters on the **full** dataset lets test-fold information into training, giving an **unrealistically good** performance estimate.
- Correct per-CV-run order: **estimate scaler on training folds, train model on training folds, transform the test fold with the training-fold statistics, then score**.
- **`make_pipeline(StandardScaler(), LinearRegression())`** chains steps; names are auto-generated as **`'standardscaler'`** and **`'linearregression'`** (Verified). All steps **but the last** must implement `fit` and `transform`; the last step is the estimator, here the model.
- Pass the pipeline as the **first argument (`estimator`)** of `cross_val_score` and each step is re-fit inside every training fold automatically.
- **`patsy.dmatrices`** turns a formula into arrays, drops rows with NAs (**344 rows to 342**, Verified), and prepends an **`Intercept`** column of 1s that `StandardScaler` flattens to zeros (harmless, `LinearRegression` fits its own intercept).
- **`KFold(n_splits=5, shuffle=True, random_state=9)`**; `get_n_splits()` returns **5** (Verified). `shuffle` defaults to **False**, and shuffling matters here: unshuffled mean RMSE **7.4763** versus shuffled **6.5350** (Verified).
- **`scoring='neg_root_mean_squared_error'`** returns negative values because scikit-learn scorers are higher-is-better; **negate** the result to read RMSE.
- Verified headline result: mean RMSE **6.5350 mm**, folds **7.1495 / 6.6674 / 6.4717 / 6.5710 / 5.8154**, `scores.std()` **0.4285**, mean R-squared **0.773**, MAE **5.334 mm**.
- The pipeline scales **inputs only**, never the target, so the score stays in the target's original units.
- For plain **OLS**, standardizing changes coefficients but **not** predictions, so pipeline, no-scaler, and leaky-scaler CV all return **6.5350** (Verified). Scaling does change results for distance-based or penalized models: on the same folds, KNN (k=5) gives **6.3432** scaled versus **6.8647** unscaled, and `Ridge(alpha=100)` gives **7.0969** versus **6.5347** (Verified).
- Leakage from **feature selection** is the severe case: selecting on all the data before CV scored higher than the correct pipeline in **30 of 30** seeded runs on pure noise (mean R-squared **0.135** versus **-0.636**, Verified).

---

> **See also:** "Cross-Validation with Scikit-learn - Notebook (Cliff Notes)" for the `KFold` plus `cross_val_score` mechanics this note assumes, and its lecture companion "Cross-Validation with Scikit-learn (Cliff Notes)"; "Introduction to Cross-Validation (Cliff Notes)" for why held-out folds estimate unseen-data performance at all; "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" for the metric behind `neg_root_mean_squared_error` and why it is not the mean absolute error; and "Comparing Models with Cross-Validation - Notebook (Cliff Notes)" for using these fold scores to choose between competing models.

---

*Source: CMPIF2100 lecture notebook (personal study use).*
