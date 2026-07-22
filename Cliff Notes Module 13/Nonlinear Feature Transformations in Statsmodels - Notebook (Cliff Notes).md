# Nonlinear Feature Transformations in `statsmodels` (Lab Notebook)

**Module 13 Cliff Notes** | Source: lab notebook `Module-13_Nonlinear_Feature_Transformations_in_Statsmodels.ipynb`

---

## TL;DR

- Linear regression does **NOT** require a straight-line relationship between input and output. "Linear" means **linear in the coefficients (β)**, not in the input `x`. A model like **ŷ = β₀ + β₁·sin(x)** is still linear regression because β₁ multiplies `sin(x)` and does not sit *inside* the sine.
- A **feature** is any input to the model. It can be the raw input or a **nonlinear transformation** of it. You transform the raw `x` with a nonlinear function first, then fit an ordinary least squares line to the transformed feature. This is a **feature transformation**.
- `statsmodels`' formula interface lets you write the transform **directly inside the formula string**: **`smf.ols('y ~ np.sin(x)', data=data).fit()`** for a sine wave, **`smf.ols('y ~ np.power(x, 2)', data=data).fit()`** for `x²`. NumPy functions work inline.
- The notebook builds two toy datasets from known formulas and recovers them: a **sine wave** (true slope 1, intercept 0) fits with **np.sin(x) ≈ 1.0** and high R-squared (**saved run 0.932**; my re-runs land ~0.91 to 0.93), and a **parabola** `2 + 0.8·x²` fits with **intercept ≈ 2, x² coef ≈ 0.8** (**saved run R-squared 0.954**; my re-runs ~0.95 to 0.96). Both Verified in behavior (data is random and unseeded, so exact digits vary per run).
- **Coefficient interpretation changes** under a transform: the coefficient is no longer "change in output per 1-unit rise in raw `x`," it is "change in output per **1-unit rise in the transformed feature**" (per 1-unit rise in `sin(x)`, or in `x²`).
- Predictions are drawn as a smooth curve using a **prediction visualization grid**: a fine `np.linspace` of x-values (with a small buffer past the min/max), fed through **`model.predict()`**, then plotted as a line over the raw scatter.

> **The one takeaway:** you can fit curved data with plain linear regression by transforming the input first, and `statsmodels` lets you drop the transform straight into the formula string (`'y ~ np.sin(x)'`, `'y ~ np.power(x, 2)'`); the model stays "linear" because it is linear in the coefficients, not in `x`.

---

## The core idea: linear in β, not in x

Earlier modules stressed this, and the notebook opens with it: linear regression only requires the model to be a **straight-line combination of its coefficients**. The inputs can be bent through any function you like first. The example given:

```text
y = β₀ + β₁·sin(x) + ε
```

is a **linear** model because β₁ is multiplied by `sin(x)`; the coefficient is not trapped inside the sine. So even wave-shaped or curved data can be fit with ordinary least squares, as long as you first pass the raw input through the right nonlinear **feature transformation**. A "feature" is just an input to the regression, raw or transformed.

## Setup

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

Same **formula (R-style) interface** as the straight-line labs (`statsmodels.formula.api as smf`), plus **NumPy** (`np`), because the transformations (`np.sin`, `np.power`) are NumPy functions called from inside the formula string.

## Example 1: a sine-wave transformation

### Generate curved data

The notebook fabricates data with a known sine pattern so it can check the fit against the truth:

```python
data = pd.DataFrame({'x': np.linspace(0, 7, 100)})
noise = np.random.normal(0, 0.2, size=len(data))
data['y'] = np.sin(data['x']) + noise

sns.relplot(data=data, x='x', y='y')
plt.show()
```

- **`np.linspace(0, 7, 100)`**: 100 evenly spaced x-values from 0 to 7.
- **`np.random.normal(0, 0.2, size=len(data))`**: Gaussian noise, mean 0, standard deviation 0.2, one value per row.
- **`y = sin(x) + noise`**: the true signal is a clean sine wave, roughened by the noise.

The scatter (Verified from the code and data) is a noisy sine wave: it rises from 0, peaks near **y ≈ 1** around **x ≈ 1.57** (π/2), falls back through zero near **x ≈ 3.14** (π), bottoms out near **y ≈ -1** around **x ≈ 4.71** (3π/2), then climbs again toward x = 7. Clearly cyclical, clearly not a straight line.

> **Note (no random seed):** the notebook never calls `np.random.seed(...)`, so `np.random.normal` draws fresh noise every run. Every number below (coefficients, R-squared, t, p) shifts slightly each time the cell runs. The saved outputs shown here are one such run; my re-runs reproduce the same *shape* of result but not the exact digits.

### Fit with the transform inside the formula

```python
model = smf.ols(formula='y ~ np.sin(x)', data=data).fit()
model.summary()
```

You put **`np.sin(x)` right in the formula string**. `statsmodels` evaluates it against the `data` frame, so the single feature the model actually sees is `sin(x)`. The intercept is still added automatically. Saved notebook output:

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.932
Model:                            OLS   Adj. R-squared:                  0.931
Method:                 Least Squares   F-statistic:                     1344.
Prob (F-statistic):                                                   4.95e-59
No. Observations:                 100
Df Residuals:                      98
Df Model:                           1
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.0169      0.019     -0.900      0.370      -0.054       0.020
np.sin(x)      1.0130      0.028     36.666      0.000       0.958       1.068
==============================================================================
```

Reading it:

- **`np.sin(x)` coefficient ≈ 1.013**, which recovers the true slope of **1** used to generate the data. Its p-value prints `0.000` (well under 0.05), so the sine feature is **highly significant**: it fits well.
- **Intercept ≈ -0.017** with **p = 0.370**. That is **not** significant, which is correct, the data was generated with no intercept (true β₀ = 0). A non-significant intercept here is the expected, honest result, not a problem.
- **R-squared = 0.932**: the sine feature explains about **93%** of the variance in y. The leftover ~7% is the injected noise.
- **Verified (behavior):** re-running the exact code gives, for example, `np.sin(x) ≈ 0.99`, intercept ≈ 0.03, R-squared ≈ 0.915; seeded runs (seeds 0/1/42) give slopes 1.06 / 0.96 / 0.95 and R-squared ≈ 0.93 each. The slope always sits near 1 and R-squared near 0.9-plus, matching the generating truth.

### Interpretation changes under a transform

The notebook flags this explicitly and it is a common quiz trap: with a nonlinear feature you can **no longer** say "a 1-unit rise in the original `x` changes the output by the coefficient." Instead the coefficient is the change in output per **1-unit rise in the transformed feature**, that is, per 1-unit change in **`sin(x)`**, not in `x`.

### Prediction visualization grid

To draw the fitted curve smoothly (rather than only at the 100 training points), build a dense grid of x-values, predict on it, and plot the predictions as a line:

```python
df_viz = pd.DataFrame({'x': np.linspace(data.x.min() - 0.1, data.x.max() + 0.1, num=101)})
df_viz.info()
```

- **`data.x.min() - 0.1` and `data.x.max() + 0.1`**: a small **0.1 buffer** on each side so the fitted curve is visible right to (and just past) the edges of the data.
- **101 points**, one column `x`, `float64`, no missing values (Verified). Only `x` exists so far; the grid holds the *inputs* you want predictions for.

```python
df_viz['pred'] = model.predict(df_viz)
df_viz.info()
```

**`model.predict(df_viz)`** runs the fitted model on the grid. It applies the *same* `np.sin(x)` transform under the hood (the formula is stored with the model), so you pass in raw `x` and get back predictions. `df_viz` now has 2 columns, `x` and `pred` (Verified: 101 non-null each).

```python
sns.scatterplot(data=data, x='x', y='y', color='black')
sns.lineplot(data=df_viz, x='x', y='pred', color='blue')
plt.show()
```

Two layers on one axis: the **black scatter** is the raw noisy data, the **blue line** is the model's prediction across the grid. Verified from the fitted values: the blue line is a **smooth sine curve** (prediction ranges roughly -0.96 to +1.02), threading through the middle of the point cloud. The prediction is a **curve, not a straight line**, which is the whole point.

## Example 2: a polynomial (x²) transformation

Sine is not the only option. The notebook repeats the pattern with a parabola.

### Generate U-shaped data

```python
data = pd.DataFrame({'x': np.linspace(-3, 3, 100)})
noise = np.random.normal(0, 0.5, size=len(data))
data['y'] = 2 + 0.8*(data['x']**2) + noise

sns.relplot(data=data, x='x', y='y')
plt.show()
```

The true model is **y = 2 + 0.8·x²**, that is, intercept β₀ = **2** and x² coefficient β₁ = **0.8**, plus noise with standard deviation 0.5. Verified shape: a **U-shaped parabola** over x in [-3, 3], with a minimum near **y ≈ 2** at **x = 0**, rising symmetrically to about **y ≈ 2 + 0.8·9 = 9.2** at **x = ±3**.

### Fit with `np.power(x, 2)`

```python
poly_model = smf.ols('y ~ np.power(x, 2)', data=data).fit()
poly_model.summary()
```

**`np.power(x, 2)`** is how you write `x²` inside a formula. Saved notebook output:

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.954
Adj. R-squared:                                                          0.954
F-statistic:                                                             2046.
Prob (F-statistic):                                                   1.80e-67
No. Observations:                 100
Df Residuals:                      98
Df Model:                           1
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          1.9087      0.074     25.846      0.000       1.762       2.055
np.power(x, 2)     0.8136      0.018     45.233      0.000       0.778       0.849
==============================================================================
```

- **Intercept ≈ 1.909** recovers the true **2**, and **`np.power(x, 2)` coef ≈ 0.814** recovers the true **0.8**. The model "found" the values the data was generated with.
- **Both p-values print `0.000`** (< 0.05), so both terms are significant.
- **R-squared = 0.954**: the x² feature explains about **95%** of the variance.
- **Verified (behavior):** re-running gives intercept ≈ 2.17, x² coef ≈ 0.78, R-squared ≈ 0.955; seeded runs (0/1/42) give x² coefs 0.87 / 0.79 / 0.81 and intercepts near 2. Always close to the generating (2, 0.8).

### Why `np.power(x, 2)` and not `x**2`

**Exam note (important patsy gotcha):** you might expect `'y ~ x**2'` to give a quadratic. It does **not**. In the formula (patsy) mini-language, **`**` means interaction expansion**, not exponentiation, and a variable interacted with itself is just itself, so `'y ~ x**2'` collapses to plain `'y ~ x'`. Verified: fitting `'y ~ x**2'` on this parabola data gives a useless **R-squared ≈ 0.0001** and just an `x` term, while `'y ~ np.power(x, 2)'` gives R-squared ≈ 0.96. To raise a variable to a power inside a formula you must either call a real function, **`np.power(x, 2)`**, or wrap the arithmetic in the identity operator, **`I(x**2)`** (both give the identical fit, Verified). This is why the notebook uses `np.power`.

### Visualize the parabola fit

```python
df_viz = pd.DataFrame({'x': np.linspace(data.x.min(), data.x.max(), 200)})
df_viz['pred'] = poly_model.predict(df_viz)

sns.scatterplot(data=data, x='x', y='y', color='black')
sns.lineplot(data=df_viz, x='x', y='pred')
plt.show()
```

Same recipe: a dense grid (here **200 points**, no buffer this time), `poly_model.predict()` to get `pred`, then a scatter of raw data plus a line of predictions. Verified: the prediction line is a **smooth upward parabola** sitting through the U-shaped cloud.

## When to reach for a nonlinear transformation

The notebook closes with a cheat-sheet of shape to transform:

- **Wave / cyclical** shape -> **`sin(x)`** or **`cos(x)`**
- **Simple curve (U-shaped)** -> **`x²`** (`np.power(x, 2)`)
- **Complex curve** -> **higher-order polynomials** such as **`x³`**

The workflow is always the same three steps: (1) look at the scatter and guess a shape, (2) fit `smf.ols('y ~ <transform of x>', data=...).fit()`, (3) build a prediction grid and plot the predicted curve over the data.

---

## Quiz-ready facts

- "Linear" regression means **linear in the coefficients (β)**, not in the input `x`. **ŷ = β₀ + β₁·sin(x)** is a linear model; the coefficient multiplies `sin(x)` and is not inside the sine.
- A **feature** is any input to the model, raw or transformed. A **feature transformation** passes raw `x` through a nonlinear function *before* fitting an ordinary least squares line.
- You write the transform **inside the formula string**: **`smf.ols('y ~ np.sin(x)', data=data).fit()`**; **`smf.ols('y ~ np.power(x, 2)', data=data).fit()`**. NumPy functions evaluate inline against the data.
- **`np.power(x, 2)`** encodes `x²` in a formula. **`'y ~ x**2'` does NOT** work: in patsy, `**` is interaction expansion, so `x**2` reduces to just `x` (Verified R-squared ≈ 0.0001 on parabola data). Use **`np.power(x, 2)`** or **`I(x**2)`** instead.
- **Coefficient interpretation under a transform:** it is the change in output per **1-unit rise in the transformed feature** (per 1-unit rise in `sin(x)`, or in `x²`), NOT per 1-unit rise in raw `x`.
- Sine example (data = `sin(x) + N(0, 0.2)`, x over [0, 7], 100 points): fitted **np.sin(x) ≈ 1.013**, intercept ≈ -0.017 (**p = 0.370, not significant, correctly, true intercept is 0**), **R-squared = 0.932** in the saved run. The intercept being insignificant is the right answer, not an error.
- Polynomial example (data = `2 + 0.8·x² + N(0, 0.5)`, x over [-3, 3]): fitted **intercept ≈ 1.909 (true 2)**, **x² coef ≈ 0.814 (true 0.8)**, both significant, **R-squared = 0.954** in the saved run. The fit recovers the generating coefficients.
- The data is generated with **`np.random.normal`** and **no `np.random.seed`**, so exact coefficients / R-squared / p-values change every run (Verified: re-runs give sine R-squared ~0.91-0.93, poly R-squared ~0.95-0.96). The *pattern* is stable; the digits are not.
- **Prediction visualization grid**: a fine `np.linspace` of x-values (sine example adds a **0.1 buffer** each side, 101 points; poly example uses 200 points, no buffer), then **`model.predict(grid)`**, then plot the predictions as a line over the raw scatter. `predict` re-applies the stored transform, so you feed it raw `x`.
- The fitted predictions form a **curve** (sine wave / parabola), not a straight line, which is the visible payoff of the transformation.
- Shape-to-transform guide: **wave/cyclical -> sin or cos**; **U-shaped curve -> x²**; **complex curve -> higher-order polynomials (x³, ...)**.

---

> **See also:** "Prediction Visualizations for Interaction Features - Notebook (Cliff Notes)" for the same `np.linspace` grid plus `model.predict()` plotting technique used here; "Additive Features in Statsmodels - Notebook (Cliff Notes)" and "Interaction Features for Linear Regression - Notebook (Cliff Notes)" for other ways to build features inside the `smf` formula string; and "Standardizing Variables for Linear Regression - Notebook (Cliff Notes)" for another pre-fit transformation of the inputs.

---

*Source: CMPIF2100 lab notebook (personal study use).*
