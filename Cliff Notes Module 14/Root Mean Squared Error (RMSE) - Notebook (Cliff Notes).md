# Root Mean Squared Error (RMSE): why we square errors, and how to read the number

**Module 14 Cliff Notes** | Source: lecture notebook `Module-14_Root_mean_squared_error_(RMSE).ipynb`

---

## TL;DR

- **Residuals** are `actual - predicted` for every row. A good model has small ones. The problem: **you cannot just average them**, because positives and negatives cancel and the score comes out near zero for good and bad models alike (Verified here: mean residual = 1.98e-09).
- Two fixes. **Mean absolute error (MAE)** averages the absolute values, and is sometimes reported. **More commonly the errors are squared**, which (1) makes every error positive and (2) **penalizes large errors much more** than small ones.
- **MSE** = average of the squared residuals: `MSE = (1/n) * sum((y - yhat)**2)`.
- **The problem with MSE: it is in squared units.** Predicting dollars gives MSE in dollars squared; predicting millimeters of river rise gives square millimeters. Hard to interpret.
- **RMSE = sqrt(MSE)**, which puts the number back **in the units of the outcome**. It answers "On average, how far off are our predictions?"
- Worked on the 150-row apartment dataset with `price_usd ~ size_m2 * city_center` (Verified): **MSE = 1,245,392,839.44 (dollars squared)** and **RMSE = 35,290.12 dollars**. So the model is off by roughly **\$35,000** on a typical condo price.
- **Lower RMSE = better fit.** Both metrics have a manual NumPy form and a statsmodels one-liner (`from statsmodels.tools.eval_measures import mse, rmse`), and the two agree exactly (Verified).

> **The one takeaway:** square the residuals so they stop cancelling and so big misses hurt more (MSE), then take the square root so the score lands back in the outcome's own units (RMSE); RMSE reads directly as "typically off by about this much," and lower is better.

---

## Setup: the model being scored

RMSE is a **score for an already-fitted model**, so the notebook first fits one, reusing the course's running apartment example: predict price from size and from whether the unit is in the city center, with an interaction between them.

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

data_url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(data_url)
model = smf.ols('price_usd ~ size_m2 * city_center', data=data).fit()
model.summary()
```

Verified: **150 rows, 3 columns** (`size_m2` float64, `city_center` bool, `price_usd` int64), 80 city-center rows and 70 not. The `*` is the statsmodels shortcut for both main effects plus their interaction, so the fit estimates **4 coefficients** (intercept plus 3 model degrees of freedom), leaving **146 residual degrees of freedom**.

Verified fit summary numbers: **R-squared = 0.878, Adj. R-squared = 0.876, F = 351.8, Prob(F) = 1.38e-66, n = 150**, coefficients **Intercept 3.159e+04, city_center[T.True] 9.287e+04, size_m2 2996.99, size_m2:city_center[T.True] 870.57**.

Nothing about RMSE depends on this particular model. Any fitted model with predictions and actuals can be scored the same way.

## Why we square the errors

A residual is one row's miss:

```text
residual = y - yhat        (actual minus predicted)
```

Verified sign convention: `model.resid` is exactly `data['price_usd'] - model.fittedvalues` (max difference 0.0), so a **positive** residual means the model **under**-predicted.

The naive summary is "average the residuals," and it fails immediately (not a notebook cell, run here to make the point):

```python
model.resid.mean()   # 1.9823589051763216e-09  (Verified: zero, up to float error)
```

That is not luck. **Least squares with an intercept forces the residuals to sum to zero**, so their mean is always zero no matter how bad the model is. The notebook's softer wording, "can misleadingly give a value close to 0," covers the general case of any model's errors. An invented four-row illustration of the same cancellation:

```text
errors:  -100, +100, -100, +100
mean of raw errors  =    0     <- looks perfect, is not
mean absolute error =  100     <- honest
RMSE                =  100     <- honest
```

(All three Verified.) Two repairs are available, and the notebook names both:

- **Mean absolute error (MAE):** average of `|residual|`, which the notebook says "is indeed sometimes reported."
- **Squared errors:** the common choice, with two stated advantages:
  1. **All errors become positive.**
  2. **Large errors are penalized more.** Quoting the notebook: "to be a little off on a prediction is alright, but being really off carries a much bigger penalty."

That second point is the real reason to prefer squaring, and it is easy to miss because it is invisible in the average. An invented pair with **identical MAE** but very different squared scores:

```text
errors 10, 10, 10, 10   ->  MAE 10   MSE 100   RMSE 10
errors  0,  0,  0, 40   ->  MAE 10   MSE 400   RMSE 20
```

(Verified.) Same total error, but the model that is catastrophically wrong once is scored **twice as badly** by RMSE. In the real apartment fit the same skew shows up: the **10 worst rows out of 150 supply 31.3% of the squared error but only 17.8% of the absolute error** (Verified).

## Mean Squared Error (MSE)

The notebook's formula, with `y` the actual observed outcome and `yhat` the model prediction for each datapoint:

```text
MSE = (1/n) * sum( (y - yhat)**2 )
```

**Way 1, manually with NumPy on `model.resid`:**

```python
mean_sq_error = np.power(model.resid, 2).mean()
print(mean_sq_error)
```

```text
1245392839.4357104
```

**Way 2, the statsmodels built-in.** It takes **two** pandas Series or NumPy arrays, the true values and the model predictions. The in-sample predictions live in **`model.fittedvalues`**:

```python
from statsmodels.tools.eval_measures import mse

mse(data['price_usd'], model.fittedvalues)
```

```text
1245392839.4357104
```

**Same value both ways** (Verified). Internally the built-in is literally `np.mean((x1 - x2) ** 2)`, so the two routes cannot disagree, and the argument order does not matter (the difference is squared).

> **Source discrepancy, and it is version drift.** The notebook's markdown says "You can ignore the `np.float64` part, which just specifies that it's a NumPy floating point number," but the **saved output above has no such wrapper**. Both are right, at different NumPy versions: **NumPy 2.x displays a scalar as `np.float64(1245392839.4357104)` where NumPy 1.x showed the bare number** (Verified: on the course's stated NumPy 2.3.5 this cell returns the wrapped form, so the saved cell was executed under an older NumPy). Only a cell's **return value** is affected. A `print()` shows the bare number either way (Verified), which is why the manual Way 1 cell above prints unwrapped. **Exam note: the wrapper carries no information; if a quiz shows either form, it is the same number.**

## Root Mean Squared Error (RMSE)

The MSE above is a **huge** number, and that is the point: **MSE is in squared units**. Predicting price in dollars gives an MSE in **dollars squared**; predicting millimeters of river rise gives **square millimeters**. Nobody has intuition for that. The fix is one square root:

```text
RMSE = sqrt(MSE)
```

**Way 1, manually:**

```python
root_mean_sq_error = np.sqrt(np.power(model.resid, 2).mean())
print(root_mean_sq_error)
```

```text
35290.12382290136
```

**Way 2, the built-in (same import module, same two arguments):**

```python
from statsmodels.tools.eval_measures import rmse
rmse(data['price_usd'], model.fittedvalues)
```

```text
35290.12382290136
```

Identical again (Verified), because `rmse()` is defined as `np.sqrt(mse(...))`. (Same version caveat as above: on NumPy 2.x this return value displays as `np.float64(35290.12382290136)`, which is the same number.)

**Reading it:** RMSE is in the **same units as the outcome**, so "on average" this model is off by about **\$35,000** on these condo prices. Against a mean price of **\$301,926.67** that is a typical miss of about **11.7%** (Verified). **A lower RMSE means a better fit.**

> **The honest caveat, which the notebook makes itself:** RMSE is not the plain arithmetic average of the misses. That is the MAE, which here is **\$28,448.99** (Verified). RMSE is **always greater than or equal to MAE** (equal only when every error has the same magnitude), and the gap widens with the spread of the errors: here the ratio is **1.24** (Verified). Treat "on average, how far off are we" as a good intuition for RMSE, not a definition.

## Three traps worth knowing

**1. `model.mse_resid` is NOT this MSE.** The fitted-model attribute divides the sum of squared residuals by the **residual degrees of freedom** (n minus the number of estimated coefficients, here 150 - 4 = 146), not by n:

```text
eval_measures.mse  = SSR / n    = 186,808,925,915.36 / 150 = 1,245,392,839.44   (Verified)
model.mse_resid    = SSR / 146  =                            1,279,513,191.20   (Verified)
sqrt(model.mse_resid) = 35,770.28   vs   RMSE = 35,290.12
```

The ratio is exactly 150/146. `model.mse_resid` is the **unbiased estimate of the error variance**, and **its square root (35,770.28) is the residual standard error** behind the standard errors in `summary()`. For this course's MSE and RMSE, **use the n-denominator version** (`eval_measures.mse` / `eval_measures.rmse`, or the NumPy `.mean()` of squared residuals). If a quiz asks for "the MSE" of a statsmodels model, it means the notebook's calculation, not `model.mse_resid`.

**2. This is a *training* RMSE.** Both calculations score the model on the **same rows it was fitted to** (`model.fittedvalues` are in-sample predictions). That number is optimistic, and adding features to an OLS model **can never increase it**, whether or not those features generalize. To estimate error on unseen data you need a held-out set or cross-validation, which is the rest of this module. The notebook does not raise this; it is worth carrying forward anyway.

**3. RMSE is scale-dependent, so it does not travel.** Because it carries the outcome's units, RMSE compares models **of the same target on the same data** and nothing else. A \$35,290 RMSE is neither good nor bad in the abstract. Two extra fits on the same dataset make the comparison concrete (added here, not in the notebook; all three scored the same way):

```text
price_usd ~ size_m2                    R2 0.338   RMSE 82,357.38
price_usd ~ size_m2 + city_center      R2 0.872   RMSE 36,204.31
price_usd ~ size_m2 * city_center      R2 0.878   RMSE 35,290.12
```

(Verified.) Notice RMSE and R-squared rank the models identically. That is not a coincidence: on the same data, `RMSE = sd(y) * sqrt(1 - R2)` using the population (n-denominator) standard deviation. Verified here: `101,232.28 * sqrt(1 - 0.8784742)` gives about **35,290.1**, reproducing the RMSE (the full-precision computation matches it to 15 significant digits). Careful with rounding, though: feeding the summary's displayed `0.878` into that formula gives 35,358.92, not 35,290.12. **R-squared is the unitless version of the same information; RMSE is the version you can quote in dollars.**

## Reproducibility note

Nothing here is random, so every number is reproducible: no `random_state` is needed because there is no sampling, splitting, or simulation. Re-running the notebook's cells reproduced the saved outputs to within **one unit in the last place** (saved `1245392839.4357104`, re-run `1245392839.4357107`; saved `35290.12382290136`, re-run `35290.12382290137`; relative difference 2e-16). That is ordinary floating-point noise from a different NumPy build, not a disagreement, and it is consistent with the version drift the `np.float64` note above pins down. Anything you would round or report is unaffected. The values were also cross-checked against scikit-learn's `mean_squared_error` and `root_mean_squared_error` (Verified: identical).

---

## Quiz-ready facts

- **Residual = actual minus predicted** (`y - yhat`); `model.resid` follows that sign, so positive residual = under-prediction.
- **You cannot average raw residuals**: positive and negative errors cancel, giving a misleading near-zero (Verified: mean residual 1.98e-09).
- **MAE** = mean absolute error, the average of the absolute residuals. Sometimes reported. **Squaring is more common.**
- **Two advantages of squaring:** all errors become positive, and **large errors are penalized more** than small ones (a small miss is tolerable, a big miss is expensive).
- **MSE** = `(1/n) * sum((y - yhat)**2)`, the average of the squared residuals.
- **The problem with MSE is its units:** dollars in gives dollars squared out. **RMSE = sqrt(MSE)** fixes that and is in the **same units as the outcome**.
- **Lower RMSE = better fit.** RMSE answers "On average, how far off are our predictions?", though the literal arithmetic average of the misses is the MAE.
- **Manual NumPy form:** `np.power(model.resid, 2).mean()` and `np.sqrt(np.power(model.resid, 2).mean())`.
- **Built-ins:** `from statsmodels.tools.eval_measures import mse, rmse`, then `mse(y_true, y_pred)` / `rmse(y_true, y_pred)`. They take **two Series or arrays**, and in-sample predictions come from **`model.fittedvalues`**.
- **Verified worked example** (`price_usd ~ size_m2 * city_center`, 150 rows, R-squared 0.878): **MSE = 1,245,392,839.44 dollars squared, RMSE = 35,290.12 dollars**, so the model is "off about \$35,000" on these condo prices. MAE for contrast: **\$28,448.99**.
- **RMSE >= MAE always**; here the ratio is 1.24 (Verified).
- **Do not confuse `eval_measures.mse` (divides by n = 150) with `model.mse_resid` (divides by the residual df, 150 - 4 = 146)**: 1,245,392,839.44 vs 1,279,513,191.20 (Verified). The course's MSE and RMSE use the n version; `sqrt(model.mse_resid) = 35,770.28` is the residual standard error, not the RMSE.
- **RMSE from `.fittedvalues` is training error**, measured on the fitting data, so it is optimistic; held-out data or cross-validation is needed for generalization error.
- **On the same data, `RMSE = sd(y) * sqrt(1 - R2)`** (population sd), so RMSE and R-squared rank models identically (Verified: `101,232.28 * sqrt(1 - 0.8784742)` gives about 35,290.1; the summary's rounded 0.878 is not precise enough for this).
- **The saved notebook shows `1245392839.4357104` bare, while its own text mentions an `np.float64` wrapper.** That is NumPy 1.x versus 2.x scalar display, not two different numbers.

---

> **See also:** "Linear Regression Residuals and R-Squared - Notebook (Cliff Notes)" and "Linear Regression Residuals and R-Squared (Cliff Notes)" for where `model.resid` comes from and for the unitless R-squared this note ties back to; "Comparing Multiple Linear Regression Models - Notebook (Cliff Notes)" for using RMSE as the yardstick when several fits compete; "Introduction to Cross-Validation (Cliff Notes)" and "Comparing Models with Cross-Validation (Cliff Notes)" for replacing this optimistic training RMSE with a held-out estimate.

---

*Source: CMPIF2100 lecture notebook (personal study use).*
