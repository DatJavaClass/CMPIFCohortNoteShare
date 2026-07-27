# Comparing Models with Cross-Validation (Lab Notebook)

**Module 14 Cliff Notes** | Source: lab notebook `comparing_models_with_cross-validation_start.ipynb`

---

## TL;DR

- With many features you can write **many** candidate regression models. Cross-validation gives each one an **honest out-of-sample score**, so you can rank them instead of guessing.
- But **lowest error is not automatically the winner**. You also prefer **simpler** models (fewer coefficients, simpler story). The notebook settles that trade-off by comparing models with a **1-standard-error interval** and taking the **simplest** model that performs as well as the best one. (That procedure is commonly called the **1-standard-error rule**; the notebook describes it without naming it.)
- The machinery: **patsy `dmatrices(formula, data)`** turns a statsmodels-style formula string into numeric arrays, then **scikit-learn's `cross_val_score`** runs **5-fold CV** on those arrays and returns **RMSE per test fold**. A helper (`eval_cv`) does this for one formula and returns a tidy DataFrame; a loop stacks all seven into one `results_df`.
- Because the design matrix from patsy **already has an `Intercept` column**, the estimator is `LinearRegression(fit_intercept=False)`. And because sklearn only *maximizes* scores, `scoring='neg_root_mean_squared_error'` is **negated** to recover positive RMSE.
- Seven formulas, five folds each, `KFold(shuffle=True, random_state=9)` (seeded, so exactly reproducible). Mean RMSE (Verified): intercept-only **102,193**, `size_m2` **83,688**, `city_center` **73,214**, `size_m2 + city_center` **36,225**, `size_m2 * city_center` **35,465**, squared-size additive **39,691**, squared-size interaction **38,534**.
- **Model 4 (`size_m2 * city_center`) has the lowest mean, but model 3 (`size_m2 + city_center`) is what the rule selects** (Verified): best mean 35,464.73 plus its standard error 2,278.50 gives a threshold of **37,743.22**, and model 3's 36,225.02 sits under it with **one fewer coefficient**. The notebook stops at the plot, so that last step is its own stated rule applied to its own numbers.

> **The one takeaway:** cross-validate every candidate formula, plot mean RMSE with a 1-standard-error bar, and pick the **simplest** model whose score is within one standard error of the best, because a difference smaller than the fold-to-fold noise is not a real difference.

---

## Why compare models at all

Once you can add features, transform them, and multiply them together, the number of possible linear models explodes. Two competing goals decide which one ships:

1. **Performance on unseen data.** Training error can only get better, never worse, as you add features on top of a model, so it cannot referee. (Verified in-sample RMSE down the nested chain here: **101,232** for `~ 1`, **82,357** adding `size_m2`, **36,204** adding `city_center`, **35,290** adding their interaction.) Cross-validation can referee: it repeatedly holds out a slice of the data, fits on the rest, and scores the held-out slice.
2. **Simplicity.** Fewer coefficients means a simpler explanation, less overfitting risk, and less to maintain. The notebook's framing: prefer "simpler models that have fewer features and offer simpler explanations of the phenomenon we're modeling."

The tie-breaker is an **error bar**. If two models' scores are within noise of each other, you have no evidence to pay for the extra complexity.

### Refresher: k-fold CV and RMSE

**k-fold cross-validation** splits the rows into k equal parts (folds). Each fold takes one turn as the test set while the other k-1 folds train the model. You end up with **k out-of-sample scores** instead of one, which is what makes a spread (and therefore an error bar) possible. Here k = 5 on 150 rows, so **each test fold holds 30 rows** (Verified).

**RMSE** (root mean squared error) is the square root of the mean squared prediction error. It lands back in the **units of the target**, so an RMSE of 36,225 means the model's price errors run on the order of **36 thousand USD**. Lower is better.

## Load data

The same 150-row apartment dataset used across the module, read straight from a course URL:

```python
import pandas as pd

data_url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(data_url)
data.info()
```

Verified: **150 rows, 3 columns, no missing values**.

```text
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   size_m2      150 non-null    float64   <- continuous, apartment size in square meters
 1   city_center  150 non-null    bool      <- categorical, True/False
 2   price_usd    150 non-null    int64     <- the target
```

Verified extras: **80 rows are `city_center == True`, 70 are `False`**, and the target has **mean 301,927 / std 101,571 USD**. Keep that standard deviation in mind, because it is roughly what a model that ignores the features can achieve.

## Step 1: list the candidate models as formula strings

Models are stored as **plain strings**, so the whole comparison becomes a loop over a list:

```python
formulas = ['price_usd ~ 1',                                     # intercept only (baseline)
            'price_usd ~ size_m2',
            'price_usd ~ city_center',
            'price_usd ~ size_m2 + city_center',                 # additive
            'price_usd ~ size_m2 * city_center',                 # additive plus interaction
            'price_usd ~ np.power(size_m2, 2) + city_center',
            'price_usd ~ np.power(size_m2, 2) * city_center']
len(formulas)   # -> 7  (Verified)
```

Three formula facts this list depends on:

- **`~ 1`** is the **intercept-only** model: no features, predicts the training mean for everything. It is the floor any real model must beat.
- **`a * b`** is shorthand for `a + b + a:b`, so it adds the **interaction** on top of both main effects.
- **`np.power(size_m2, 2)` replaces `size_m2` with its square**; it does **not** also keep the linear term. Models 5 and 6 therefore contain **no plain `size_m2`** (Verified from the design-matrix column names).

The `import numpy as np` in this cell is **load-bearing, not decorative**: patsy evaluates `np.power(...)` in your namespace, so dropping the import raises `PatsyError: Error evaluating factor: NameError: name 'np' is not defined` (Verified).

> **Source note:** the same cell runs `import statsmodels.formula.api as smf`, but **`smf` is never called anywhere in this notebook** (Verified). The formula parsing is done by **patsy**, the engine statsmodels borrows, and the fitting is done by scikit-learn. Harmless, but do not expect an `smf.ols` call to show up.

## Step 2: the `eval_cv` helper

One function per model: formula in, tidy DataFrame of per-fold RMSE out. The moving parts:

```python
from patsy import dmatrices
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

y_feature, x_features = dmatrices(formula, data=data)          # formula -> numeric arrays
kf  = KFold(n_splits=5, shuffle=True, random_state=9)          # the same 5 splits every time
clf = LinearRegression(fit_intercept=False)                    # patsy already supplied Intercept
scores = -cross_val_score(clf, x_features, y_feature.ravel(),
                          cv=kf, scoring='neg_root_mean_squared_error')
```

Line by line:

- **`dmatrices`** returns two arrays: the target `y_feature` (a patsy `DesignMatrix` of shape `(150, 1)`) and the design matrix `x_features` (shape `(150, k)`). `.ravel()` flattens the target to the conventional 1-D `(150,)` that sklearn expects. (Verified: handing over the raw `(150, 1)` array works too, scores identically, and raises no warning. Ravel anyway; 1-D is the shape sklearn documents.)
- **`shuffle=True`** matters when rows arrive in a meaningful order (sorted, grouped, time-ordered); without it the folds are contiguous blocks. **`random_state=9`** freezes the shuffle. A fresh `KFold` is built on every call, but the identical seed and identical row count mean **all seven models are scored on exactly the same folds**, which makes the comparison paired and fair. (Verified aside: dropping `shuffle` moves model 4's mean RMSE from 35,464.73 to 36,113.69 here.)
- **`fit_intercept=False`** avoids fitting a *second* intercept on top of patsy's `Intercept` column. (Verified curiosity: leaving it `True` gives **identical** RMSE for all seven models, because a duplicate constant column does not change the span of the design matrix; only the split between the two intercepts becomes non-unique. Still set it to `False`, so the coefficients stay interpretable.)
- **The leading minus sign** flips sklearn's sign convention. Its scorers are "higher is better," so RMSE is served as `neg_root_mean_squared_error` and comes back negative (Verified raw fold values for model 4: `-31275.61, -39953.92, -40810.29, -29354.24, -35929.57`). Negating restores real RMSE.

The function then packs `model_name`, `model_formula`, `num_coefs` (`x_features.shape[1]`), `fold_id`, and `rmse` into a DataFrame, which is what makes the results **stackable and plottable**.

Verified smoke test, `eval_cv(formulas[0], 0, data)`:

```text
   model_name  model_formula  num_coefs  fold_id           rmse
0           0  price_usd ~ 1          1        0  104715.328722
1           0  price_usd ~ 1          1        1   96785.749927
2           0  price_usd ~ 1          1        2  102733.480694
3           0  price_usd ~ 1          1        3   97636.407429
4           0  price_usd ~ 1          1        4  109093.823343
```

Sanity check on that baseline: predicting the mean should give an RMSE near the target's standard deviation, and indeed the five folds average **102,193** against a full-sample std of **101,571** (Verified).

> **Bug in the source (harmless):** the function contains
> `num_features = x_features.shape[0]  # number of columns is the number of features`.
> **`shape[0]` is the number of rows (150), not columns.** The comment describes `shape[1]`. The variable is **never used again** (Verified: the name appears exactly once in the notebook) and the DataFrame correctly uses `x_features.shape[1]`, so no result is affected. Note also that `shape[1]` counts the **Intercept**, which is why the honest column name is **`num_coefs`**, not "features."
> *Exam safety:* if a quiz asks what that line was meant to do, answer the course's way, that it counts the model's features/coefficients.

## Step 3: run all seven and stack the results

```python
results_list = []
for i, formula in enumerate(formulas):
    results_list.append(eval_cv(formula, i, data))

results_df = pd.concat(results_list)     # 7 models x 5 folds = 35 rows
```

Aggregating those 35 rows (Verified: every per-fold value reproduces the notebook's saved output **exactly**, to all printed digits):

| # | formula | `num_coefs` | mean RMSE | std err (1 SE) |
|---|---|---|---|---|
| 0 | `price_usd ~ 1` | 1 | 102,192.96 | 2,283.38 |
| 1 | `price_usd ~ size_m2` | 2 | 83,687.90 | 2,453.91 |
| 2 | `price_usd ~ city_center` | 2 | 73,213.84 | 4,721.70 |
| 3 | `price_usd ~ size_m2 + city_center` | 3 | **36,225.02** | 2,152.16 |
| 4 | `price_usd ~ size_m2 * city_center` | 4 | **35,464.73** | 2,278.50 |
| 5 | `price_usd ~ np.power(size_m2, 2) + city_center` | 3 | 39,691.27 | 3,482.33 |
| 6 | `price_usd ~ np.power(size_m2, 2) * city_center` | 4 | 38,533.86 | 2,893.17 |

What the table says:

- **Either feature alone beats the baseline, but neither alone comes close to both together.** Going from model 2 (73,214) to model 3 (36,225) roughly **halves** the error, so `size_m2` and `city_center` carry largely **different** information.
- **The squared-size models (5 and 6) are worse than their linear counterparts (3 and 4).** Replacing `size_m2` with its square throws away the linear term the data actually wants, so the extra curvature costs rather than pays.
- **Models 3 and 4 are separated by only 760 USD of mean RMSE**, which is well inside either model's standard error (about 2,200). That gap is noise.

## Step 4: plot with error bars and apply the 1-SE rule

```python
sns.catplot(data=results_df, x='model_name', y='rmse', kind='point',
            linestyle='none', errorbar=('ci', 68), height=4)
```

`kind='point'` draws each model's **mean** as a dot with an **error bar**; `linestyle='none'` suppresses the connecting line, which is right for a **categorical** x axis where a line would imply a trend that does not exist. Re-running the cell (Verified) gives a descending staircase from model 0 to model 3, with the biggest single drop between models 2 and 3, then a flat cluster of models 3 through 6 whose bars all overlap each other.

Eyeballing overlaps is the loose version. The arithmetic version is stricter:

```text
best mean          = 35,464.73   (model 4)
its standard error =  2,278.50
threshold          = 37,743.22
within threshold   = model 3 (36,225.02) and model 4 (35,464.73)
simplest of those  = model 3, with 3 coefficients vs 4
```

Models 5 and 6 have overlapping bars but means **above** the threshold, so the rule drops them. The selected model is **`price_usd ~ size_m2 + city_center`** (Verified). The interaction in model 4 buys a slightly lower average error, but not enough to clear the noise floor, so the additive model wins on parsimony.

**Robustness (Verified over 20 different `random_state` values, since seed 9 is an arbitrary choice):** model 4 had the lowest mean in **20 of 20** runs, and the 1-SE rule, scoring simplicity by fewest coefficients, selected model 3 in **19 of 20**. Mean RMSE ranged 36,225 to 37,735 for model 3 and 35,465 to 36,733 for model 4 across those seeds. The conclusion is not an artifact of the seed.

> **Source correction (error-bar type):** the notebook's markdown says "compare models with a 1-standard error interval instead of the usual 2. So that's a 68% confidence interval instead of a 95% interval from a Gaussian distribution." The Gaussian rule of thumb behind that sentence is right, but **`errorbar=('ci', 68)` is not a 1-SE bar**: in seaborn it is a **bootstrap percentile** interval at the 68% level. The literal 1-SE bar is **`errorbar=('se', 1)`**. On these 5-fold results the two are close but not equal (Verified half-widths: model 3 gives 1,866 bootstrap vs 2,152 standard error; model 2 gives 4,076 vs 4,722), and seaborn's bootstrap is **unseeded by default**, so the drawn interval wobbles slightly between runs (Verified: model 4's upper end moved by under 50 USD across 20 repeats). Neither detail changes the selection here.
> *Exam safety:* if a quiz keys to the course's wording, answer its way, that **1 standard error is about a 68% interval and 2 standard errors about 95%**. Just know that `('ci', 68)` and `('se', 1)` are different computations in seaborn.

---

## Quiz-ready facts

- **Cross-validation is the model-comparison tool** because training error can only improve as features are added to a model, so it cannot rank fairly. CV scores each candidate on **data it did not train on**.
- **5-fold CV on 150 rows leaves 30 rows per test fold** (Verified) and yields **5 scores per model**, which is what supplies the spread for an error bar.
- **RMSE is in the target's units** (USD here); lower is better. The **intercept-only** model (`~ 1`) predicts the mean and scores near the target's **standard deviation** (mean RMSE 102,193 vs std 101,571, Verified).
- **`patsy.dmatrices(formula, data)` returns `(y, X)`**: the target as `(n, 1)` and the design matrix as `(n, num_coefs)`, **Intercept column included**. Use `.ravel()` to hand sklearn the conventional 1-D target (Verified: the raw `(n, 1)` array happens to score identically here).
- **`LinearRegression(fit_intercept=False)`** is used *because* patsy already added the Intercept column. (Verified aside: `True` gives identical RMSE here, since a duplicate constant column does not change the design matrix's span.)
- **sklearn scorers are "higher is better,"** so RMSE is requested as **`scoring='neg_root_mean_squared_error'`** and the returned array is **negated** to get positive RMSE.
- **`KFold(n_splits=5, shuffle=True, random_state=9)`**: `shuffle` breaks any row ordering, `random_state` makes the split reproducible. The same seed on the same 150 rows means **all seven models see identical folds**, a paired comparison.
- **`~ 1`** = intercept only. **`a * b`** = `a + b + a:b`. **`np.power(x, 2)`** in a formula **replaces** `x` with its square rather than adding to it.
- **Verified mean RMSE, 7 models:** 102,193 / 83,688 / 73,214 / **36,225** / **35,465** / 39,691 / 38,534 for `~1`, `size_m2`, `city_center`, `size_m2 + city_center`, `size_m2 * city_center`, `power(size_m2,2) + city_center`, `power(size_m2,2) * city_center`.
- **The 1-SE rule:** find the lowest mean CV error, add **one standard error** of that model, and choose the **simplest** model whose mean falls under the threshold. Here **35,464.73 + 2,278.50 = 37,743.22**, so the pick is **model 3, `price_usd ~ size_m2 + city_center`** (Verified), not the lowest-scoring model 4.
- **Convention (course wording):** compare with a **1-standard-error** interval rather than the usual 2, that is roughly **68%** rather than **95%**. Caveat: seaborn's `errorbar=('ci', 68)` is a **bootstrap** interval, while `errorbar=('se', 1)` is the literal one.
- **`kind='point'` with `linestyle='none'`** plots means and error bars with no connecting line, which is correct for a **categorical** x axis.
- **Squaring a feature is not free flexibility.** Models 5 and 6 dropped the linear `size_m2` term and scored **worse** than models 3 and 4.

---

> **See also:** "Comparing Models with Cross-Validation (Cliff Notes)" (the companion lecture note for this same lab, with the spoken framing of the performance-versus-simplicity trade-off), "Cross-Validation with Scikit-learn - Notebook (Cliff Notes)" (where `KFold` and `cross_val_score` are introduced on their own), "Introduction to Cross-Validation (Cliff Notes)" (the conceptual case for held-out scoring over training error), "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" (what the score being minimized actually measures, from squared errors up through MSE and RMSE), and "Comparing Multiple Linear Regression Models - Notebook (Cliff Notes)" (the in-sample counterpart: a ladder of increasingly flexible models on synthetic sine-wave data, ranked by training-set R-squared and RMSE with no held-out split at all).

---

*Source: CMPIF2100 lab notebook (personal study use).*
