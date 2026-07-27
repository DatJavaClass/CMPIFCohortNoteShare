# Linear Regression Residuals and R-Squared (Lab Notebook)

**Module 14 Cliff Notes** | Source: lab notebook `linear_regression_residuals_rsquared_start.ipynb`

---

## TL;DR

- A model is never a perfect predictor. Measuring how close its predictions land to the real data is called **goodness of fit** or **performance**.
- A **residual** is one row's error: **observed minus predicted**. Positive means the model **under**-predicted that row, negative means it over-predicted.
- Statsmodels gives you both halves directly: **`model.fittedvalues`** (the prediction for every training row) and **`model.resid`** (the residuals). Verified: `model.resid` is **exactly** `observed - fittedvalues`, element for element.
- Running example: **food truck profit predicted from city population**, 97 rows, one input. Fitted line (Verified): **profit = -3.8958 + 1.1930 * population**.
- A **predicted vs observed plot** puts the real values on the x-axis and the predictions on the y-axis. Perfect predictions would sit on the **45-degree line y = x**, which the notebook draws in red.
- **R-squared** condenses all those residuals into one number. Verified: **`model.rsquared` = 0.7020315537841397** (about **0.702**), matching the notebook's saved output exactly.
- The notebook defines R-squared as the correlation between observed and predicted, squared. That is **true here** but is a special case, and the general R-squared **can go negative**. Details and an exam-safety note below.

> **The one takeaway:** a residual is a single row's `observed - predicted` error (`model.resid`), and R-squared rolls all of them into one goodness-of-fit score where 1 is perfect and 0 is no better than predicting the mean; on this food truck model R-squared is 0.702 (Verified), all measured on the training data itself.

---

## Setup: the food truck dataset

The data is a headerless two-column text file pulled from a URL, so the column names have to be supplied by hand:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

profits = pd.read_csv('https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt',
                      header=None,
                      names=['population', 'profit'])
profits.info()
```

`header=None` stops pandas from eating the first data row as a header, and `names=` supplies the labels. Verified: **97 rows, 2 columns, both `float64`, no missing values**.

Verified `head()`:

```text
   population   profit
0      6.1101  17.5920
1      5.5277   9.1302
2      8.5186  13.6620
3      7.0032  11.8540
4      5.8598   6.8233
```

`population` is city population and `profit` is the food truck's profit in that city (the notebook does not state the units, and nothing here depends on them). Goal: predict `profit` from `population`.

## Fitting the model

```python
model = smf.ols(formula='profit ~ population', data=profits).fit()
```

Plain single-input linear regression, no categorical inputs and no interactions. The notebook never prints the coefficients or p-values, but they are worth having (added, Verified by running the fit):

```text
Intercept    -3.895781
population    1.193034
```

So each extra unit of population is associated with about **1.19 more profit**, and both terms are significant (Verified: intercept p = 4.61e-07, slope p = 1.02e-26).

## Fitted values: what the model predicted for its own training rows

```python
model.fittedvalues
```

`.fittedvalues` holds the model's prediction for **every row it was trained on**. Verified output (abridged, and matching the notebook's saved output):

```text
0      3.393774
1      2.698951
2      6.267196
...
95    12.083712
96     2.590624
Length: 97, dtype: float64
```

It comes back as a **pandas Series carrying the original index** (Verified), so it lines up with the DataFrame for free. Row 0 checks out by hand: `-3.895781 + 1.193034 * 6.1101 = 3.3937760` (Verified), matching the printed **3.393774** to four decimals (3.3938). The last digits differ only because the coefficients above are themselves rounded, so do not read that as an error.

```python
profits['predictions'] = model.fittedvalues
```

## Seeing the errors: predictions overlaid on the data

```python
fig, ax = plt.subplots(figsize=(4,3))
sns.scatterplot(data=profits, x='population', y='profit', ax=ax)
sns.scatterplot(data=profits, x='population', y='predictions', ax=ax)
plt.show()
```

Two Seaborn **Axes-level** calls sharing one `ax` overlay the observed points and the predicted points. Axes-level functions take an `ax=` argument; figure-level ones (`relplot`, `lmplot`) do not, because they build their own figure and so cannot be layered this way (Verified from their signatures). That is exactly why the notebook reaches for `scatterplot` here, and it says so again later when it drops `relplot` to add the reference line.

The predicted points fall on a **perfectly straight line** because they are a linear function of `population`. The **vertical gap** between each blue observed point and the orange predicted point directly above or below it **is** that row's residual (Verified: the two layers draw in `#1f77b4` then `#ff7f0e`, the default blue and orange).

## Residuals: observed minus predicted

Computed by hand, then read straight off the model:

```python
profits.profit - profits.predictions   # manual
model.resid                            # same thing, from statsmodels
```

Verified: the two are **identical element for element** (maximum absolute difference 0.0), and both reproduce the notebook's saved output:

```text
0     14.198226
1      6.431249
2      7.394804
3      7.394728
4      3.728142
...
92     4.094738
93    -0.446840
94    -5.853984
95    -3.028612
96    -1.973574
```

The sign convention is **observed minus predicted**, so a **positive residual means the model guessed too low** for that row. Two rows worth remembering (Verified):

- **Row 0** is the largest residual in the dataset: population 6.1101, actual profit **17.592**, predicted **3.394**, residual **+14.198**.
- **Row 94** is the worst over-prediction: population 8.2934, actual profit **0.14454**, predicted **5.999**, residual **-5.854**.

> **Why you cannot just average the residuals.** Verified: they sum to **3.95e-14**, i.e. zero to floating-point precision. That is a **mathematical guarantee of OLS whenever the model has an intercept**, not evidence of a good fit. The split here is **39 positive and 58 negative residuals** (Verified), and they still cancel because the positive ones are larger. Any useful error metric therefore has to remove the signs first, by squaring (R-squared, RMSE) or by taking absolute values (MAE). For reference on this model, RMSE = **2.9923** and MAE = **2.1942** (Verified).

## The predicted vs observed plot

Instead of an input on the x-axis, put the **actual data** on the x-axis and the **predictions** on the y-axis:

```python
sns.relplot(data=profits, x='profit', y='predictions', height=3)
plt.show()
```

Perfect predictions would form a **45-degree line**, so the notebook adds one in red, drawn from the data against itself (`x='profit', y='profit'` is exactly the line y = x):

```python
fig, ax = plt.subplots(figsize=(4,4))
sns.scatterplot(data=profits, x='profit', y='predictions', ax=ax)
sns.lineplot(data=profits, x='profit', y='profit', color='red', ax=ax)
plt.show()
```

> **What the red line quietly fixes.** In the bare `relplot` version the two axes are on **different scales** (Verified: y spans about 1.08 to 23.62, the predictions' range, while x spans about -4.02 to 25.49, the observed range), so the cloud's tilt is misleading before you even start reading it. Adding the y = x line pulls both axes to the **same limits** (Verified: both -4.02 to 25.49), which is what makes the comparison legible. Caveat: a line renders at a true 45 degrees only if the axes box is square too, so matching limits inside `figsize=(4,4)` gets very close while `ax.set_aspect('equal')` would guarantee it.

> **The cloud is flatter than the line on purpose.** For an OLS fit with an intercept, the predictions are always **less spread out** than the observations, because the model only reproduces the part of the variation it can explain. Verified exactly here: `sd(predictions) / sd(profit) = 0.837873`, which is `sqrt(R-squared)` and also the absolute correlation. This flattening is expected behavior, not a defect in the fit.

## R-squared

```python
model.rsquared
```

Verified: **0.7020315537841397**, identical to the notebook's saved output. Scale, in the notebook's words: **1 = perfect predictions, 0 = no correlation between predictions and observations**. The standard added reading is that **about 70% of the variance in profit is accounted for by the model**, which follows from the sum-of-squares form below.

Three routes to the same 0.702 on this model (Verified: they agree to 15 significant figures, spread 4.4e-16, which is just floating-point noise):

```python
import numpy as np
from sklearn.metrics import r2_score

np.corrcoef(profits.profit, profits.predictions)[0, 1] ** 2       # 0.7020315537841393
1 - (model.resid ** 2).sum() / ((profits.profit - profits.profit.mean()) ** 2).sum()
                                                                  # 0.7020315537841397
r2_score(profits.profit, profits.predictions)                     # 0.7020315537841397
```

Here the residual sum of squares is **868.5324** and the total sum of squares is **2914.8471** (Verified). Because there is only one input, `corr(population, profit) ** 2` lands on the same number too (Verified).

> **Naming trap:** statsmodels stores the **residual** sum of squares in **`model.ssr`** (not `sse`), and the explained sum of squares in `model.ess`. Reading `ssr` as "sum of squares regression" is a classic mix-up; here `model.ssr` = 868.5324 and `model.ess` = 2046.3146 (Verified).

The course's "R-squared of 0 means no relationship" also checks out: an intercept-only model (`profit ~ 1`) predicts the mean for every row and returns **R-squared = 0** (Verified: -2.2e-16, floating-point zero).

## Where "squared correlation" stops being R-squared

The notebook's markdown says R-squared "is simply the correlation between the observed and predicted values of the output, **squared so that it doesn't contain negative values**." That gives the right number here, but two parts are loose:

**1. Squaring is not a trick to prevent negatives.** The general definition is `R-squared = 1 - SSE/SST`, and that quantity **can be negative** when a model does worse than just predicting the mean. Verified demonstration: the deliberately upside-down predictor `20 - 1.193034 * population` has **the same squared correlation as the good model (0.702032 either way)** but an actual R-squared of **-2.757976**. Squared correlation cannot tell a good model from a backwards one; R-squared proper can.

**2. The equality only holds for an OLS fit with an intercept, scored on its own training data.** Drop the intercept (`profit ~ population - 1`) and the three quantities separate (Verified): squared correlation **0.702032**, statsmodels `.rsquared` **0.817332**, and `r2_score` **0.610072**. Three different answers to "what is R-squared" for one model. The statsmodels number is the odd one out because without an intercept it silently switches the denominator to the **uncentered** total sum of squares (Verified: `1 - ssr/uncentered_tss = 1 - 1136.5792/6222.1104 = 0.817332`).

> **Exam safety:** inside this course's setup (statsmodels OLS **with** an intercept, evaluated on the training data), the notebook's sentence is correct and produces the right value, so **if a quiz keys to the course's wording, answer its way**: R-squared is the correlation between observed and predicted values, squared, running from 0 (no relationship) to 1 (perfect). Keep the `1 - SSE/SST` definition in your back pocket for the cross-validation notebooks later in this module, where scoring on held-out folds is exactly where negative R-squared shows up.

## Everything here is training performance

The notebook says it in its opening paragraph and it is easy to skim past: the question being asked is how well predictions match "the observed, real-world data **that was used for training the model**." Every number above, R-squared included, is computed on **the same rows the model was fitted on**, so it measures fit, not the ability to predict anything new. Training R-squared can also be pushed up by simply adding terms: adding a column of pure random noise to this model raised it from **0.702032 to 0.702033** while adjusted R-squared **fell** from 0.698895 to 0.695694 (Verified). Fixing this properly is the job of the rest of Module 14.

---

## Quiz-ready facts

- **Residual = observed minus predicted**, one per row. Positive residual means the model **under**-predicted.
- **`model.fittedvalues`** = predictions for the training rows; **`model.resid`** = the residuals. Verified: `model.resid` equals `observed - model.fittedvalues` exactly, and both are pandas Series carrying the data's index.
- Evaluating how well a model matches real data is called **goodness of fit** or **performance**; residuals are the raw material for every such metric.
- Dataset: food truck profit vs city population, **97 rows, 2 columns, both float64, no missing values**, read with `header=None` and `names=['population','profit']` because the source file has no header row.
- Fitted model `profit ~ population` (Verified): **Intercept -3.895781, population 1.193034**; both p-values far below 0.05.
- **OLS residuals sum to zero whenever the model has an intercept** (Verified: 3.95e-14). That is arithmetic, not a quality signal, which is why metrics square or absolute-value the errors first.
- A **predicted vs observed plot** puts observed on x and predicted on y; perfect predictions lie on the **45-degree line y = x**, drawn here with `sns.lineplot(data=profits, x='profit', y='profit', color='red')`.
- **Axes-level Seaborn functions** (`scatterplot`, `lineplot`) accept `ax=` and can be layered on one Axes; figure-level ones (`relplot`) create their own figure and cannot.
- **R-squared: 1 = perfect predictions, 0 = no correlation between predictions and observations.** Verified on this model: **`model.rsquared` = 0.7020315537841397** (about 0.702, roughly 70% of variance explained).
- Course definition: **R-squared = squared correlation between observed and predicted**. Verified equal to `1 - SSE/SST` and to sklearn's `r2_score` **for an OLS model with an intercept scored on its training data** (SSE 868.53, SST 2914.85).
- The general **`1 - SSE/SST` can be negative** (Verified: a backwards predictor scored -2.758 while keeping squared correlation at 0.702). Without an intercept, squared correlation, statsmodels `.rsquared`, and `r2_score` disagree (Verified: 0.702032, 0.817332, 0.610072).
- For an OLS fit with an intercept, predictions are **always less spread out than observations**: Verified `sd(predictions)/sd(observed) = 0.837873 = sqrt(R-squared)`.
- Statsmodels naming trap: the **residual** sum of squares is `model.ssr` (there is no `model.sse`), and `model.ess` is the **explained** sum of squares (Verified: 868.5324 and 2046.3146, summing to the total 2914.8471).
- Everything in this notebook is measured **on the training data**, so it is training performance, not an estimate of performance on new data.

---

> **See also:** "Linear Regression Residuals and R-Squared (Cliff Notes)" is the companion lecture note for this same topic. "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" is the natural next metric, built from the very residuals computed here (RMSE 2.9923 on this model, Verified). "Introduction to Cross-Validation (Cliff Notes)" explains why training-data R-squared flatters a model and what to do instead, "Comparing Multiple Linear Regression Models - Notebook (Cliff Notes)" uses R-squared to rank competing models, and "Cross-Validation with Scikit-learn - Notebook (Cliff Notes)" is where `r2_score` and held-out scoring (including the negative R-squared warned about above) turn up in practice.

---

*Source: CMPIF2100 lab notebook (personal study use).*
