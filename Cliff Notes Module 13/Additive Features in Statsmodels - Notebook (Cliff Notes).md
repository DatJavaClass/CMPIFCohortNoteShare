# Additive Features in Linear Regression with `statsmodels` (Lab Notebook)

**Module 13 Cliff Notes** | Source: lab notebook `Module-13_Additive_Features_in_Statsmodels.ipynb`

---

## TL;DR

- Moves from one predictor to **many**. The simplest way to combine inputs is **addition**: **ŷ = β₀ + β₁x₁ + β₂x₂**, where each input contributes **independently** (β₁x₁ and β₂x₂) and the prediction is the **sum of those contributions** plus the intercept.
- Fitting a multi-input additive model in the formula interface is just **`+` between the inputs**: **`smf.ols('y ~ x1 + x2', data=df).fit()`**. The intercept is still added automatically.
- Uses a **synthetic** dataset (100 rows) built from a known truth **y = 5 + 2·x1 - 3·x2 + noise** (Gaussian noise, sd 2, `np.random.seed(2100)`). `x1` is uniform on [0,10], `x2` uniform on [0,5].
- The fit **recovers the true coefficients** closely: **Intercept 4.7269**, **x1 = 1.9675**, **x2 = -2.9085**, **R-squared = 0.944** (Verified). Positive `x1` slope, negative `x2` slope, matching the data-generating formula.
- Visualizing predictions uses a **grid of input combinations**: a nested loop over **101 values of `x1`** crossed with **9 values of `x2`** = **909 rows** (Verified), fed to `model.predict()`, then drawn as a **line plot with `x1` on the x-axis and `x2` as line color (hue)**. The 9 lines are **parallel** (equal slope), the visual signature of a purely **additive** (no-interaction) model.

> **The one takeaway:** additive features just add each input's contribution: write them with a `+` in the formula (`'y ~ x1 + x2'`), and because the effects are separate, changing one input shifts every prediction by the same amount without changing the other input's slope, so a prediction-grid line plot shows parallel lines.

---

## Setup: imports

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
```

Same formula interface as the single-predictor notebook (**`statsmodels.formula.api as smf`**), plus **`numpy`** this time because the data is generated synthetically rather than read from a file.

## The concept: additive effects

With more than one input, the simplest combination is a straight sum:

```text
ŷ = β₀ + β₁x₁ + β₂x₂
```

- Each input contributes **independently**. The contribution from `x1` is **β₁·x₁**, the contribution from `x2` is **β₂·x₂**.
- The final prediction is the **sum** of those contributions plus the intercept β₀.
- "Additive" means the effects **do not depend on each other**: `x2`'s value never changes how much a one-unit change in `x1` moves the prediction. (When they *do* interact, you need an interaction term, which is a later notebook.)

Two intuition examples from the notebook:

1. **Movie-review rating** predicted from counts of "good"-synonyms and "bad"-synonyms. The "good" coefficient is positive, the "bad" coefficient negative; the predicted rating is the positive contribution plus the (negative) contribution added together.
2. **Office attendance** for hybrid workers predicted from precipitation and day-of-week. If it is both raining and a Friday, each feature is associated with staying home, and the two effects add.

## Generate the data

The dataset is synthetic, built from a known equation so we can check the fit against the truth:

```python
np.random.seed(2100)

x1 = np.random.uniform(0, 10, 100)
x2 = np.random.uniform(0, 5, 100)

y = 5 + 2*x1 - 3*x2 + np.random.normal(0, 2, 100)

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
df.head()
```

Key points:
- **`np.random.seed(2100)`** fixes the random stream so every run gives identical numbers (that is why the outputs below are reproducible).
- **True model:** intercept **5**, `x1` slope **+2**, `x2` slope **-3**, with independent Gaussian **noise** of mean 0 and standard deviation 2 added to each row.
- `x1` spans **[0, 10]** (uniform), `x2` spans **[0, 5]** (uniform), **100 rows**.

Verified `head()`:

```text
         x1        x2          y
0  8.472328  2.778576  14.684253
1  0.231498  1.843218  -3.100701
2  0.868333  4.279402  -4.483096
3  3.358812  2.780849   6.631815
4  4.888804  1.115036  10.118405
```

## Look at each input against `y`

Before fitting, plot each predictor separately against the outcome:

```python
sns.lmplot(data=df, x='x1', y='y')
sns.lmplot(data=df, x='x2', y='y')
plt.show()
```

`sns.lmplot` draws a scatterplot **with a fitted regression line** overlaid. Verified from the data and the coefficients:
- **`x1` vs `y`:** the cloud and its fitted line slope **up** (as `x1` rises, `y` rises), matching the positive true slope +2.
- **`x2` vs `y`:** the cloud and its fitted line slope **down** (as `x2` rises, `y` falls), matching the negative true slope -3.

As the notebook puts it, both are linear but **the inverse of each other**.

(Study note: each `lmplot` line is a *single-variable* regression, so it shows the **marginal** relationship, whereas the model's coefficients below are **partial** effects, each input's effect holding the other constant. Here `x1` and `x2` were generated independently, so the marginal slopes and the partial coefficients land close together; that agreement is not guaranteed when predictors are correlated.)

## Fit the additive model

To combine the two inputs additively in the formula interface, **add them with `+`**:

```python
model = smf.ols('y ~ x1 + x2', data=df).fit()
model.summary()
```

The formula `'y ~ x1 + x2'` encodes **ŷ = β₀ + β₁x₁ + β₂x₂**. As always, **the intercept is included automatically**, so you never write β₀.

Verified full summary output (fresh run reproduces the notebook exactly):

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.944
Model:                            OLS   Adj. R-squared:                  0.943
Method:                 Least Squares   F-statistic:                     819.6
Date:                Wed, 22 Jul 2026   Prob (F-statistic):           1.73e-61
Time:                        12:52:18   Log-Likelihood:                -205.95
No. Observations:                 100   AIC:                             417.9
Df Residuals:                      97   BIC:                             425.7
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      4.7269      0.500      9.455      0.000       3.735       5.719
x1             1.9675      0.063     31.378      0.000       1.843       2.092
x2            -2.9085      0.131    -22.154      0.000      -3.169      -2.648
==============================================================================
Omnibus:                        2.375   Durbin-Watson:                   1.917
Prob(Omnibus):                  0.305   Jarque-Bera (JB):                1.803
Skew:                          -0.205   Prob(JB):                        0.406
Kurtosis:                       3.515   Cond. No.                         16.6
==============================================================================
```

## Read the results

- **The `coef` column is the fitted model.** The fitted equation is:

  ```text
  y = 4.7269 + 1.9675 x1 - 2.9085 x2
  ```

  Verified full-precision coefficients: **Intercept = 4.726926**, **x1 = 1.967544**, **x2 = -2.908475**.
- **Coefficients recover the truth.** The data was built with intercept 5, slope +2, slope -3; the fit lands at 4.73, +1.97, -2.91. Close but not exact, because the added noise perturbs each estimate. This is the point of the synthetic setup: it shows regression backing out the true structure.
- **Interpreting each slope (partial effect):** a **one-unit increase in `x1`**, holding `x2` fixed, is associated with about a **+1.97** change in predicted `y`; a **one-unit increase in `x2`**, holding `x1` fixed, is associated with about a **-2.91** change. The intercept **4.73** is the prediction when both inputs are 0.
- **All three terms are highly significant.** Every `P>|t|` prints `0.000`. Verified exact p-values: Intercept **2.02e-15**, x1 **1.36e-52**, x2 **9.93e-40** (a printed `0.000` means "< 0.0005," never literally zero). Verified 95% CIs, none containing 0: Intercept **[3.735, 5.719]**, x1 **[1.843, 2.092]**, x2 **[-3.169, -2.648]**.
- **`R-squared = 0.944`:** the two-input model explains about **94%** of the variance in `y`. Verified full precision **0.9441302** (**Adj. R-squared = 0.9429782**).
- **`Df Model = 2`, `Df Residuals = 97`:** 2 predictors now (not 1), so residual df = 100 - 1 intercept - 2 slopes = 97.
- **`F-statistic = 819.6`, `Prob (F-statistic) = 1.73e-61`:** the overall model is overwhelmingly significant. Verified F full precision **819.589**.

## Visualize predictions: build a grid of input combinations

To see how predictions move across the input space, build a **grid** of every combination of evenly spaced `x1` and `x2` values, using a nested list comprehension:

```python
viz_grid = pd.DataFrame(
    [
        (x1, x2)
        for x1 in np.linspace(df.x1.min(), df.x1.max(), 101)
        for x2 in np.linspace(df.x2.min(), df.x2.max(), 9)
    ],
    columns=['x1', 'x2']
)

viz_grid.info()
viz_grid.head()
```

- **`np.linspace(start, stop, N)`** makes `N` evenly spaced values between the data's min and max.
- **101 values of `x1`** (the axis we will plot along) crossed with **9 values of `x2`** (the lines we will color) = **101 × 9 = 909 rows** (Verified). The lab suggests roughly **5-10 values** for the second variable so the plot does not get too crowded with lines.

Verified `info()`: **909 entries, 2 float64 columns** (`x1`, `x2`), no missing values. Verified `head()` (note `x1` is held fixed while `x2` cycles through its 9 values first, because `x2` is the inner loop):

```text
         x1        x2
0  0.033213  0.007941
1  0.033213  0.630909
2  0.033213  1.253877
3  0.033213  1.876845
4  0.033213  2.499812
```

## Predict on the grid

```python
viz_grid['pred'] = model.predict(viz_grid)   ## grid in, predictions out
```

**`model.predict(new_df)`** takes a DataFrame whose columns match the model's inputs and returns the predicted outcome for each row. Assigning it back adds a **`pred`** column. Verified first rows of the resulting predictions:

```text
         x1        x2      pred
0  0.033213  0.007941  4.769177
1  0.033213  0.630909  2.957291
2  0.033213  1.253877  1.145405
3  0.033213  1.876845 -0.666482
4  0.033213  2.499812 -2.478368
```

(At fixed `x1 ≈ 0.033`, each step up in `x2` of about 0.623 drops the prediction by about 1.81, which is exactly slope × step = -2.9085 × 0.623. That constant drop is the additive effect in action.)

## Plot the predictions

```python
sns.relplot(
    data=viz_grid,
    x='x1',
    y='pred',
    hue='x2',
    kind='line',
)
plt.show()
```

- **`x='x1'`** puts `x1` on the x-axis, **`y='pred'`** the model output on the y-axis, **`hue='x2'`** colors a separate line for each distinct `x2` value, **`kind='line'`** draws lines (not points).
- Result: **9 lines**, one per unique `x2` value. Verified the 9 values with `viz_grid.x2.unique()`:

  ```text
  array([0.00794148, 0.63090922, 1.25387695, 1.87684468, 2.49981242,
         3.12278015, 3.74574788, 4.36871561, 4.99168335])
  ```

  They span roughly **0 to 5**; seaborn labels the legend at tidy integers **1-4** as color samples.

**Reading the plot:**
- **As `x1` increases, every line rises** (the positive `x1` coefficient, +1.97).
- **As `x2` increases (darker hue), the whole line shifts down** (the negative `x2` coefficient). A higher `x2` lowers *all* predictions on that line by the same amount.
- **The lines are parallel: all 9 have the same `x1` slope.** `x2` changes only the height (intercept) of each line, not its slope. That parallelism is the visual fingerprint of a purely **additive** model: the two effects are **separate and additive**, with no interaction between them.

---

## Quiz-ready facts

- **Additive model equation:** **ŷ = β₀ + β₁x₁ + β₂x₂**; each input contributes independently (β₁x₁, β₂x₂) and the prediction is their **sum** plus the intercept.
- **Formula syntax for additive inputs:** put a **`+`** between them, `smf.ols('y ~ x1 + x2', data=df).fit()`. The intercept is still added automatically (you never write β₀).
- **`np.random.seed(2100)`** makes the synthetic data reproducible. The true generating model is **y = 5 + 2·x1 - 3·x2 + Normal(0, 2)**, with `x1` uniform on [0,10], `x2` uniform on [0,5], 100 rows.
- **Fitted coefficients (Verified):** Intercept **4.7269**, x1 **1.9675**, x2 **-2.9085**; close to the true 5, +2, -3, with the gap due to noise.
- **R-squared = 0.944** (Adj. 0.943): the two inputs explain about 94% of `y`'s variance. **F = 819.6**, Prob(F) = 1.73e-61.
- All three terms are highly significant (every `P>|t|` prints `0.000`; true p-values 2.02e-15, 1.36e-52, 9.93e-40); 95% CIs exclude 0.
- **`Df Model = 2`** (two predictors), **`Df Residuals = 97`** (= 100 - 1 intercept - 2 slopes).
- A slope is a **partial effect:** the change in predicted `y` per one-unit change in that input **holding the other input fixed**. An `lmplot` line, by contrast, shows the **marginal** single-variable relationship.
- **Prediction grid:** a nested comprehension crossing **`np.linspace(min, max, 101)` for `x1`** with **`np.linspace(min, max, 9)` for `x2`** gives **101 × 9 = 909** input combinations; keep the second variable to roughly **5-10** values so the plot stays readable.
- **`model.predict(new_df)`** returns predictions for any DataFrame whose columns match the inputs; assign it to a new column.
- **`sns.relplot(..., x='x1', y='pred', hue='x2', kind='line')`** draws one line per `x2` value. In an additive model these lines are **parallel** (same slope, shifted up or down by `x2`); parallel lines = **no interaction**.

---

> **See also:** "Interaction Features for Linear Regression - Notebook (Cliff Notes)" and "Prediction Visualizations for Interaction Features - Notebook (Cliff Notes)" for the contrast case, where effects are *not* separable and the grid lines stop being parallel; "Nonlinear Feature Transformations in Statsmodels - Notebook (Cliff Notes)" for bending the straight-line assumption; "Standardizing Variables for Linear Regression - Notebook (Cliff Notes)" for putting differently-scaled coefficients on a comparable footing; and "Categorical Inputs to Linear Regression - Notebook (Cliff Notes)" with the companion lecture note "Categorical Inputs to Linear Regression (Cliff Notes)" for adding non-numeric predictors to the same additive framework.

---

*Source: CMPIF2100 lab notebook (personal study use).*
