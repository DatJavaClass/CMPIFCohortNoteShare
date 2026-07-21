# Making Predictions with Linear Regression in Statsmodels (Lab Notebook)

**Module 12 Cliff Notes** | Source: lab notebook `Module-12_Making_Predictions_with_Linear_Regression_Statsmodels.ipynb`

---

## TL;DR

- Once a model is **fitted**, you use it to **predict**, not just to describe. This notebook fits one OLS model on the food truck data and then generates predicted profit for a whole grid of new population values.
- The core workflow: (1) build a **prediction visualization grid** of evenly spaced inputs with `np.linspace`, (2) fit with `smf.ols(...).fit()` on the **original training data**, (3) call `model.predict(grid)` to get one predicted output per grid row, (4) plot the resulting **trend line**, (5) overlay the real training points.
- **`.predict()` returns only the point estimate of the mean** (the trend line). It gives you **no interval and no uncertainty band**. Intervals (mean CI vs prediction interval) come from `model.get_prediction(...)`, which this notebook does **not** use; the sibling "uncertainty" notes cover that.
- The fitted line is **profit = -3.8958 + 1.1930 * population**, with **R-squared = 0.702** on 97 observations. `.predict()` just plugs each grid value into that formula.
- **Fit the model on the data, predict on the grid.** The grid is only for smooth visualization; you never train on it. The prediction DataFrame must contain the **same column name** (`population`) the formula referenced.

> **The one takeaway:** `model.predict(new_df)` maps a tidy grid of inputs to the model's predicted **mean** output, giving you a clean trend line to draw over your data; for the uncertainty around that line, you need `get_prediction`, not `predict`.

---

## Dataset: food truck profits vs city population

The notebook loads the classic food truck dataset (city **population** as input, food truck **profit** as output) straight from a URL:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

profits = pd.read_csv('https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt',
                      header=None,
                      names=['population', 'profit'])
profits.info()
profits.head()
```

- Verified: **97 rows, 2 columns** (`population`, `profit`), both `float64`, no missing values.
- Verified `head()`: population `6.1101, 5.5277, 8.5186, 7.0032, 5.8598` with profit `17.5920, 9.1302, 13.6620, 11.8540, 6.8233`.
- Verified `population.min() = 5.0269`, `population.max() = 22.203`. (These two numbers drive the grid range below.)

There is no local copy of `ex1data1.txt` in the Module 12 folder; the fresh run above pulled the file live from the URL, and every number matched the notebook's saved output.

## Step 1: build a prediction visualization grid

To draw a smooth trend line you feed the model **many evenly spaced inputs**, not just the scattered training values. `np.linspace(start, stop, num)` returns `num` equally spaced points. The range is nudged **0.1 past the data on each side** so the trend is a little easier to see at the edges:

```python
viz_input = np.linspace(profits.population.min() - 0.1, profits.population.max() + 0.1, num=101)
viz_input
```

- Verified: **length 101**, running from `4.9269` (that is `5.0269 - 0.1`) up to `22.303` (that is `22.203 + 0.1`), evenly spaced. First few values: `4.9269, 5.100661, 5.274422, 5.448183, 5.621944`.
- Why **101** and not 100: `num` counts the **points**, and points = intervals + 1, so `100 + 1 = 101` gives a round **100 intervals**. The notebook notes this is a nicety, not a requirement, since the endpoints are set by the data anyway.

Wrap the grid values in a DataFrame so `statsmodels` can match them to the formula by column name:

```python
df_viz = pd.DataFrame({'population': viz_input})
df_viz.head()
```

- Verified `head()`: `4.926900, 5.100661, 5.274422, 5.448183, 5.621944`.
- The column **must be named `population`**, because the model formula (next step) refers to `population`. `.predict()` looks up predictors by name.

## Step 2: fit the model (on the training data, not the grid)

Fit ordinary least squares with the R-style formula interface. Train on **`profits`** (the real data), never on the grid:

```python
model = smf.ols(formula='profit ~ population', data=profits).fit()
model.summary()
```

Verified summary highlights (fresh run matches the notebook's saved output exactly):

```text
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.8958      0.719     -5.415      0.000      -5.324      -2.467
population     1.1930      0.080     14.961      0.000       1.035       1.351

R-squared: 0.702   Adj. R-squared: 0.699   F-statistic: 223.8   Prob (F): 1.02e-26
No. Observations: 97   Df Residuals: 95   Df Model: 1   AIC: 491.9   BIC: 497.1
```

- Verified exact coefficients: **Intercept = -3.895780878**, **population (slope) = 1.193033644**. So the fitted mean line is `profit_hat = -3.8958 + 1.1930 * population`.
- Verified `R-squared = 0.7020`, `Adj. R-squared = 0.6989`, `F = 223.83`, `Prob(F) = 1.02e-26`, `nobs = 97`. Both coefficients are highly significant (p < 0.001).

## Step 3: make predictions with `.predict()`

The fitted object carries a `.predict()` method. Hand it the grid and it returns one predicted output per row:

```python
lm_pred = model.predict(df_viz)
lm_pred
```

- Verified: a pandas Series of **length 101**, indexed `0..100`.
- Verified first 5 predictions: `1.982177, 2.189479, 2.396782, 2.604085, 2.811387`.
- Verified last 5 predictions: `21.883238, 22.090540, 22.297843, 22.505146, 22.712448`.
- Sanity check on what `.predict()` actually does: plugging the first grid value into the fitted line by hand, `-3.895780878 + 1.193033644 * 4.9269 = 1.98217658`, which equals the first prediction. So `.predict()` is literally evaluating `intercept + slope * population` at each grid point.

### Important: `.predict()` gives the mean, not an interval

`model.predict(...)` returns **only the point estimate of the mean response**: a single predicted profit per input, which is exactly the trend line. It carries **no standard error, no confidence interval, and no prediction interval**.

To get uncertainty you use a different method, `model.get_prediction(new_df)`, whose result exposes both:

- the **confidence interval for the mean** (uncertainty about where the average line sits), and
- the wider **prediction interval** (also called the observation interval: uncertainty about a single new observed value, which adds the residual scatter on top).

This notebook stops at the trend line and does **not** call `get_prediction`, so no interval numbers appear here. The mean CI vs prediction interval distinction is developed in the sibling uncertainty notes (see below). For this note, the one fact to hold: **`.predict()` draws the line; `.get_prediction()` draws the bands.**

## Step 4: store the predictions alongside the inputs

Keep inputs and predictions together in one tidy DataFrame so plotting is trivial:

```python
df_viz['pred_trend'] = lm_pred
df_viz.head()
```

- Verified `head()`: population `4.926900, 5.100661, 5.274422, 5.448183, 5.621944` mapping to `pred_trend` `1.982177, 2.189479, 2.396782, 2.604085, 2.811387`.
- Here `population` is the evenly spaced grid and `pred_trend` is the model's predicted profit for each of those inputs.

## Step 5: visualize the predicted trend, then overlay the data

Plot the predictions as a line. `pred_trend` is smooth because the inputs are evenly spaced, so a figure-level `relplot` with `kind='line'` draws the model's predicted relationship:

```python
sns.relplot(data=df_viz, x='population', y='pred_trend', kind='line')
plt.show()
```

To judge the fit, overlay the **actual training points** on the **same Axes**. The notebook uses Axes-level functions (`lineplot` + `scatterplot`) with a shared `ax` so both land on one plot:

```python
fig, ax = plt.subplots()
sns.lineplot(data=df_viz, x='population', y='pred_trend', ax=ax)
sns.scatterplot(data=profits, x='population', y='profit', color='black', ax=ax)
plt.show()
```

- The **blue line** is the predicted trend (`df_viz`); the **black points** are the observed training data (`profits`).
- Pattern to remember: **figure-level** functions (`relplot`, `catplot`) manage their own figure and are awkward to combine, so when you need to layer two things on one Axes, reach for the **Axes-level** functions (`lineplot`, `scatterplot`) and pass them the same `ax`.

## Why this workflow matters

- Linear regression is for **prediction**, not only interpretation of coefficients.
- `.predict()` estimates outputs for **brand-new inputs** the model never saw.
- A **prediction grid** turns the model into a smooth, readable trend line across the whole input range.
- Overlaying training data is a quick eyeball test of **how well the line fits** (and here it also reveals skew: the summary's high skew and kurtosis hint the residuals are not perfectly normal, which is why the intervals in the sibling notes matter).

---

## Quiz-ready facts

- **Workflow order:** build grid with `np.linspace` -> fit `smf.ols('profit ~ population', data=profits).fit()` -> `model.predict(df_viz)` -> store -> plot. **Fit on the data, predict on the grid.**
- `np.linspace(start, stop, num=101)` yields **101 points = 100 intervals**; `num` counts points, not gaps. Verified range here: `4.9269` to `22.303` (data min/max `5.0269`/`22.203`, each nudged out by `0.1`).
- The prediction DataFrame **must contain the predictor column by the same name** used in the formula (`population`); `.predict()` matches predictors by name.
- Fitted model: **profit_hat = -3.8958 + 1.1930 * population**, **R-squared = 0.702**, **97 observations**, both coefficients significant (p < 0.001). Verified.
- `.predict()` returns a **Series of point estimates of the mean** (length 101 here), with **no interval**. Verified first/last predictions: `1.982177 ... 22.712448`.
- **Mean CI vs prediction interval:** `.predict()` gives neither; `model.get_prediction(new_df)` gives the **confidence interval for the mean** and the wider **prediction (observation) interval**. This notebook only draws the line.
- **Plotting rule:** to overlay a line and a scatter on one plot, use **Axes-level** `lineplot` + `scatterplot` sharing one `ax`, not two figure-level `relplot` calls.

---

> **See also:** "Fitting Linear Regression Models - Notebook (Cliff Notes)" for how the `smf.ols(...).fit()` model and its summary are produced, and "Visualizing Uncertainty on Predictions - Notebook (Cliff Notes)" for the `get_prediction` confidence and prediction intervals this notebook stops short of.

---

*Source: CMPIF2100 lab notebook (personal study use).*
