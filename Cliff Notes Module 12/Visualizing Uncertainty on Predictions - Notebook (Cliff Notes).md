# Visualizing Uncertainty on Linear Regression Predictions (Lab Notebook)

**Module 12 Cliff Notes** | Source: lab notebook `visualizing_uncertainty_predictions_start.ipynb`

---

## TL;DR

- The notebook refits the **food truck** model (`profit ~ population`, 97 rows) and then shows how to attach and draw **two kinds of uncertainty** onto its predictions.
- **Confidence interval (CI) on the mean**: where the **true average** output is likely to sit at a given input. Columns **`mean_ci_lower` / `mean_ci_upper`**. Narrow.
- **Prediction interval (PI) on individual observations**: where a **single new data point** is likely to land. Columns **`obs_ci_lower` / `obs_ci_upper`**. Always **wider** than the CI, because it adds the model's leftover scatter on top of the uncertainty in the mean.
- The workhorse call is **`model.get_prediction(grid).summary_frame()`**, which returns a 6 column DataFrame: **`mean`, `mean_se`, `mean_ci_lower`, `mean_ci_upper`, `obs_ci_lower`, `obs_ci_upper`**.
- You plot them as **ribbons** with `ax.fill_between(...)`: the grey CI band hugs the regression line, and a wider band shows the PI. **Verified:** about 96% of the 97 training points fall inside the 95% PI band, but only about 23% fall inside the CI band.

> **The one takeaway:** the **confidence interval** answers "where is the true average?" (narrow); the **prediction interval** answers "where will one new observation be?" (wide). In statsmodels they are the `mean_ci_*` columns versus the `obs_ci_*` columns of `summary_frame()`.

---

## Setup: refit the food truck model

Same dataset and model as the earlier prediction notebook. Profit is predicted from town population.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

profits = pd.read_csv('https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt',
                      header=None,
                      names=['population', 'profit'])
model = smf.ols(formula='profit ~ population', data=profits).fit()
model.summary()
```

**Verified** (fresh run matches the notebook's saved output exactly): 97 rows, 2 float columns, no missing values. The fit gives **Intercept = -3.8958** (std err 0.719) and **slope on population = 1.1930** (std err 0.080), with **R-squared = 0.702** and 95 residual degrees of freedom. The data loads live from GitHub (there is no local CSV in the Module 12 folder); the URL was reachable, so all numbers below are from a fresh execution.

## Two kinds of uncertainty (the whole point)

The notebook opens by naming both intervals up front:

- A **confidence interval** of where we can be confident the **mean output** lies.
- A **prediction interval** where we can be confident **individual output observations** are.

Keep these separate in your head, because statsmodels hands you both at once and it is easy to grab the wrong pair of columns.

| | Question it answers | statsmodels columns | Width |
|---|---|---|---|
| **Confidence interval** | Where is the **true average** profit at this population? | `mean_ci_lower`, `mean_ci_upper` | Narrow |
| **Prediction interval** | Where will **one new** food truck's profit land? | `obs_ci_lower`, `obs_ci_upper` | Wide |

## Build a prediction grid

To draw smooth ribbons you predict on a dense, evenly spaced grid of inputs rather than only the raw data points. The notebook pads the observed population range by 0.1 on each end and takes 101 points.

```python
viz_grid = pd.DataFrame({'population': np.linspace(profits.population.min()-0.1,
                                                   profits.population.max()+0.1,
                                                   num=101)})
```

**Verified:** 101 rows running from **4.9269** (that is `min - 0.1`) to **22.303** (that is `max + 0.1`), evenly spaced. This grid is only the x axis you want to draw across; it carries no `profit` column because that is what the model will estimate.

## Get predictions plus intervals with `summary_frame()`

Two calls chained together: `get_prediction()` makes the predictions, and `summary_frame()` unpacks them into a tidy DataFrame that already contains both intervals. (This is the upgrade over the plain `predict()`, which returns the point estimate only.)

```python
pred_summary = model.get_prediction(viz_grid).summary_frame()
pred_summary
```

**Verified** first row (`population = 4.9269`), matching the notebook's saved output to six decimals:

```text
       mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
0  1.982177  0.400893       1.186304       2.778049     -4.073054      8.037408
```

and the last row (`population = 22.303`):

```text
        mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
100  22.712448  1.168872      20.391943      25.032954     16.276833     29.148064
```

Reading the six columns (as the notebook lists them):

- **`mean`**: the prediction itself, which is an estimate of the **mean** output you would get if you observed profit at this population many times.
- **`mean_se`**: the **standard error of the mean** estimate (your best friend for building the CI).
- **`mean_ci_lower` / `mean_ci_upper`**: the 95% **confidence interval on the mean**.
- **`obs_ci_lower` / `obs_ci_upper`**: the 95% **prediction interval** on an individual observation.

Notice at row 0 the PI runs into **negative** territory (`obs_ci_lower = -4.07`). That is fine here (a truck can run at a loss), and it is a good reminder that the PI is much wider than the CI.

## CI versus PI: the exact formulas (verified by hand)

Both intervals are `mean ± t* × (a standard error)`, using the same t critical value. What differs is which standard error goes inside.

- **t critical value** at 95% with 95 residual degrees of freedom: **1.9853** (this is why a 95% interval is "roughly 2 standard errors," see the note below).
- **Confidence interval** uses `mean_se` directly:
  `CI = mean ± t* × mean_se`
- **Prediction interval** adds the model's **residual variance** `σ²` (statsmodels calls it `model.scale`) under the square root:
  `PI = mean ± t* × sqrt(mean_se² + σ²)`

**Verified at row 0** (`mean = 1.982177`, `mean_se = 0.400893`, `σ² = model.scale = 9.142447`, so `sqrt(σ²) = 3.0236`):

- CI by hand `[1.186304, 2.778049]` equals the `mean_ci_*` columns exactly.
- `obs_se = sqrt(0.400893² + 9.142447) = 3.050108`, so PI by hand `[-4.073054, 8.037408]` equals the `obs_ci_*` columns exactly.
- **CI width = 1.59, PI width = 12.11.** The PI is wider at every single grid point (verified across all 101 rows).

Intuition: the CI only has to account for wobble in **where the line sits**. The PI also has to account for the **spread of individual points around the line** (the `σ²` term), which never shrinks no matter how much data you have, so the PI stays wide.

## Visualize the confidence interval

The notebook draws the regression line, fills the grey CI ribbon between `mean_ci_lower` and `mean_ci_upper`, then overlays the raw training scatter.

```python
fig, ax = plt.subplots(figsize=(4,4))

# Line plot of the linear regression predictions
sns.lineplot(x=viz_grid["population"], y=pred_summary["mean"], ax=ax)

# Ribbon for the confidence interval
ax.fill_between(viz_grid["population"], pred_summary["mean_ci_lower"],
                pred_summary["mean_ci_upper"], facecolor="grey",
                edgecolors="grey", alpha=0.5)

# Scatterplot of the training data
sns.scatterplot(data=profits, x="population", y="profit", ax=ax)

ax.set_ylabel("mean prediction")
plt.show()
```

**Verified:** runs clean and renders. The grey band is thin and hugs the line, and it flares out slightly at the ends of the population range (uncertainty about the mean grows where data is sparse). Most scatter points sit **outside** this narrow band, which is expected: the CI is about the average, not about individual trucks.

## Add the prediction interval (completing the notebook)

The final section is titled **"Visualize predictions with confidence intervals and prediction intervals,"** but in the start notebook its code cells are **left blank as an exercise**. The natural fill-in reuses the same pattern with a second, wider `fill_between` driven by the `obs_ci_*` columns. Draw the wide PI band **first** so the narrow CI band sits on top of it.

```python
fig, ax = plt.subplots(figsize=(4,4))

sns.lineplot(x=viz_grid["population"], y=pred_summary["mean"], ax=ax)

# Wide ribbon: prediction interval (individual observations)
ax.fill_between(viz_grid["population"], pred_summary["obs_ci_lower"],
                pred_summary["obs_ci_upper"], facecolor="lightblue",
                edgecolors="lightblue", alpha=0.5)

# Narrow ribbon: confidence interval (the mean), drawn on top
ax.fill_between(viz_grid["population"], pred_summary["mean_ci_lower"],
                pred_summary["mean_ci_upper"], facecolor="grey",
                edgecolors="grey", alpha=0.5)

sns.scatterplot(data=profits, x="population", y="profit", ax=ax)
ax.set_ylabel("prediction")
plt.show()
```

**Verified:** runs clean. The wide light band swallows almost all of the scatter, the narrow grey band tracks the line inside it.

## Coverage sanity check (why the two bands look so different)

A quick count of how many of the 97 training points land inside each band drives the distinction home:

- **Prediction interval:** **93 of 97 points (95.9%)** fall inside. That is exactly what a 95% PI promises: about 95% of individual observations land within it.
- **Confidence interval:** only **22 of 97 points (22.7%)** fall inside. The CI is never meant to contain individual points; it only brackets the **average** at each input.

If a plot's uncertainty band captures nearly all your scatter, you are looking at a **prediction** interval, not a confidence interval.

## A single worked prediction

Predicting at one specific input makes the two intervals concrete. **Verified** at `population = 15`:

```text
        mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
0  13.999724  0.625926      12.757103      15.242344      7.869755     20.129692
```

Best guess for the **average** profit at a population of 15 is about **14.0**, and we are 95% confident the true average is in **[12.76, 15.24]**. But any **single** truck in such a town could plausibly earn anywhere in **[7.87, 20.13]**.

---

## Source correction / exam-safety note

- The notebook (cell describing the columns) says `mean_ci_lower` "will be approximately **2 × mean_se** lower than **mean**." That is a fair rule of thumb but not exact: the real multiplier is the **t critical value**, here **1.9853** (with 95 residual df), not exactly 2. At row 0, `mean - 2*mean_se = 1.1804` versus the true `mean_ci_lower = 1.1863`. The word "approximately" makes the course's phrasing acceptable. **If a quiz keys to "about 2 standard errors," answer its way**; just know the precise factor is the t value (which sits near 2, and near the normal-distribution 1.96, for reasonably sized samples).
- The notebook's plain-language definitions of the two intervals ("where the mean output lies" versus "where individual output observations are") are correct and match the verified math, so nothing else needs flagging.

---

## Quiz-ready facts

- **`model.get_prediction(grid).summary_frame()`** returns six columns: **`mean`, `mean_se`, `mean_ci_lower`, `mean_ci_upper`, `obs_ci_lower`, `obs_ci_upper`**. Plain `predict()` gives only the point estimate.
- **`mean_ci_*` = confidence interval on the MEAN** (narrow). **`obs_ci_*` = prediction interval on an INDIVIDUAL observation** (wide). This column pairing is the single most testable fact here.
- **CI = mean ± t × mean_se.** **PI = mean ± t × sqrt(mean_se² + σ²)**, where **σ² = `model.scale`** is the residual variance. The extra `σ²` term is why the **PI is always wider** than the CI.
- The PI stays wide even with lots of data (the `σ²` scatter term does not vanish); the CI can shrink toward the line as data grows.
- Draw intervals with **`ax.fill_between(x, lower, upper, alpha=...)`**; plot the wide PI band first, the narrow CI band on top.
- Build a smooth ribbon with a dense input grid via **`np.linspace(min-0.1, max+0.1, num=101)`** (101 evenly spaced points, padded slightly past the data).
- Verified model: **Intercept -3.8958, slope 1.1930, R-squared 0.702**, 97 observations, 95 residual df, t* = 1.9853.
- Coverage rule of thumb: a **95% PI** should contain about **95%** of observations (verified 95.9% here); the CI is not meant to contain individual points (only 22.7% here).

---

> **See also:** the companion lecture note "Visualizing Uncertainty on Linear Regression Predictions (Cliff Notes)" for the concepts without code, and "Making Predictions with Linear Regression - Notebook (Cliff Notes)" for where `get_prediction()` and the food truck model were introduced.

---

*Source: CMPIF2100 lab notebook (personal study use).*
