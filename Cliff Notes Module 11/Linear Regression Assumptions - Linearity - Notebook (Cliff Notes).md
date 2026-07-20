# Linear Regression Assumptions: Linearity (Lab Notebook)

**Module 11 Cliff Notes** | Source: lab notebook `Module-11_Linear_Regression_Linearity.ipynb`

---

## TL;DR

- The notebook recaps linear regression (**yₙ = β₀ + β₁xₙ + εₙ**, modeling the **mean** of Y given X), lists **all four assumptions**, then digs into the **linearity** one in code.
- The four assumptions: **(1) Linearity of the mean**, **(2) Normality of errors**, **(3) Independence**, **(4) Constant variance (homoscedasticity)**.
- The key idea (and common misconception fixed): **"linear" means the mean output is linear in the coefficients (β), not in the input data.** You may transform the input any way you like, and as long as the **coefficient stays outside the transformation**, it is still a linear model.
- Three code demos on generated data: (a) genuinely **linear** data (slope 1, intercept 1) fit by the line `y = 1 + x`; (b) a **sine-wave** pattern made linear by transforming `x → sin(x)`; (c) a **parabola** made linear by transforming `x → x²` (polynomial regression).
- A model is **truly nonlinear** only when a coefficient is **inside a function or multiplied by another coefficient**, for example `y = β₀ + e^(β₁x)`. Move the coefficient outside (`y = β₀ + β₁e^x`) and linear regression works again via feature transformation.

> **The one takeaway:** you do **not** need a straight-line relationship in the raw data to use linear regression. Transform curved inputs (sin, log, exp, powers) first, and linear regression fits the transformed feature. It is far more flexible than "a straight best-fit line."

---

## Recap: what linear regression models

For every datapoint **n**, the model assumes:

```text
yₙ = β₀ + β₁xₙ + εₙ
```

with **β₀ = intercept**, **β₁ = slope**, **εₙ = error term**. Linear regression models the **mean (average) of Y given X**, so `yₙ` is really the estimate of the **mean** value of Y at that input (given the β coefficients and input x).

## The four assumptions of linear regression

Models are simplifying lenses, so it helps to know exactly what they assume:

1. **Linearity of the mean.** The mean output (written **μₙ** as an alternative to yₙ) is a **linear combination of predictors**: `μₙ = β₀ + β₁xₙ`. Individual points scatter due to noise, but the **average** relationship follows a straight line. Even if the data looks curved, you can often **transform the inputs** (x², log(x), sin(x)) and still use a linear model.
2. **Normality of errors.** `εₙ ~ Normal(0, σ)`. For any given input, the observed outputs are distributed like a **normal (Gaussian)** curve; most values lie near the peak (the mean), which is the regression line, so most datapoints fall along that line.
3. **Independence.** Datapoints are **independent**; the prediction errors are **not correlated** with each other, and one observation gives no information about another. This is **violated in time-series** data, where current values depend on past values (creating correlated errors). The notebook flags it but does not go deep.
4. **Constant variance (homoscedasticity).** The **variance of Y is constant across all inputs**, so the spread of points around the line is a **uniform band** (uncertainty does not grow or shrink with x). Also flagged but not explored in depth.

## What does "linear" actually mean?

Linear regression belongs to the family of **"linear" models**. The intuitive answer, "a straight best-fit line between input and output," is **not** the real definition:

> **"Linear" means the mean output is linearly related to the regression coefficients (β), not to the input data itself.** You can transform the input however you want, and as long as the coefficient is **outside** that transformation, it is still a linear model.

For example,

```text
y = β₀ + β₁ sin(x)
```

is **still linear**, because β is outside the transformation of x into sin(x). This is what makes feature transformation possible.

## Demo 1: genuinely linear data

Generate data with a real linear relationship (slope 1, intercept 1) plus normal noise, which fits linear regression's assumptions:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
data = pd.DataFrame({'x': np.linspace(-2, 2, 50)})
noise = np.random.normal(0, 0.5, size=len(data))
data['y'] = 1 + data['x'] + noise          # true model: intercept 1, slope 1
```

- `data.info()` reports a **50-row, 2-column** DataFrame (`x`, `y`, both `float64`, no missing values).
- Verified `head()` (seed 42): the first `y` values are `-0.751643, -0.987499, -0.512890, 0.006413, -0.790546` at `x = -2.0, -1.918367, -1.836735, -1.755102, -1.673469`.

Add the exact model prediction (a straight line) and plot it in red over the scatter:

```python
data['estimated_y'] = 1 + data['x']        # matches the data-generating line, so it fits well
fig, ax = plt.subplots()
sns.scatterplot(data=data, x='x', y='y', ax=ax)
sns.lineplot(data=data, x='x', y='estimated_y', ax=ax, color='red')
plt.show()
```

Because `estimated_y` uses the **same** intercept and slope the data was generated from, the red line runs right through the cloud, a clean linear fit. (Verified `estimated_y` head: `-1.0, -0.918367, -0.836735, -0.755102, -0.673469`.)

## Demo 2: feature transformation (sine wave → linear)

Now generate data that is clearly **not** linear in x, a sine wave with noise:

```python
data = pd.DataFrame({'x': np.linspace(0, 7, 100)})     # 100 points, x from 0 to 7
noise = np.random.normal(0, 0.2, size=len(data))
data['y'] = np.sin(data['x']) + noise
sns.relplot(data=data, x='x', y='y')                    # a wavy, non-linear scatter
plt.show()
```

Fitting a straight line directly to `x` vs `y` would fail. The fix is to **transform the input first**, then model a linear relationship with the transformed feature:

```python
data['x_transformed'] = np.sin(data['x'])               # sin(x) ranges about -1 to 1
sns.relplot(data=data, x='x_transformed', y='y')        # now clearly linear
plt.show()
```

Against `sin(x)`, the relationship is a clean straight line. And `y = β₀ + β₁ sin(x)` is **still linear in β**. Knowing the **general shapes** of a few transforms (the cyclical sine wave here, polynomials next) tells you which transformation to reach for.

## Demo 3: polynomial regression (parabola → linear)

A **polynomial** transformation raises the input to a power. Generate a parabola (intercept 0, slope 3 on x²) with noise:

```python
np.random.seed(9)
data = pd.DataFrame({'x': np.linspace(-2, 2, 50)})
noise = np.random.normal(0, 0.5, size=len(data))
data['y'] = 3 * np.power(data['x'], 2) + noise          # roughly y = 3x^2
sns.relplot(data=data, x='x', y='y')                     # a parabola, not linear in x
plt.show()
```

- Verified: `x` runs `-2.0` to `2.0`, `y` runs roughly `-0.64` to `12.41`.

Transform `x → x²` and the relationship straightens out:

```python
data['x_transformed'] = np.power(data['x'], 2)           # x^2 ranges 0 to 4 here
sns.relplot(data=data, x='x_transformed', y='y')         # now linear
plt.show()
```

Assigning `f = x²`, the model is just `y = β₀ + β₁f`, an ordinary linear regression on the new feature. **Any transformation** can be dropped into `f`, and it is still regular linear regression.

## When it is truly nonlinear

A model is genuinely **nonlinear** only when a **coefficient is inside a function or multiplied by another coefficient**. For example:

```text
y = β₀ + e^(β₁ x)      # β₁ is INSIDE the exponential  -> NOT linear in coefficients
```

You **cannot** use linear regression here. But if the coefficient sits **outside** the function, you can, with a feature transformation `x → exp(x)`:

```text
y = β₀ + β₁ e^x        # β₁ is OUTSIDE  -> linear (transform x into exp(x) first)
```

## Why this matters

You do not need a straight line in the raw data to use linear regression. If the data looks curved:

- Try transformations like **sin, log, exp**.
- Add **polynomial** terms (x², x³, ...).

Linear regression is **more flexible than it looks**: the "linear" constraint is on the **coefficients**, not on the shape of the input.

---

## Quiz-ready facts

- Model recap: **yₙ = β₀ + β₁xₙ + εₙ**, and linear regression models the **mean of Y given X**.
- The **four assumptions**: **linearity of the mean**, **normality of errors** (`ε ~ Normal(0, σ)`), **independence** (violated by time series), **constant variance / homoscedasticity**.
- **"Linear" = linear in the coefficients (β), not in the input data.** Transform the input freely; if the coefficient is **outside** the transformation, it is still linear.
- `y = β₀ + β₁ sin(x)` and `y = β₀ + β₁x²` (with `f = x²`, `y = β₀ + β₁f`) are **linear** models via **feature transformation**.
- **Truly nonlinear** = a coefficient **inside a function or multiplied by another coefficient**, e.g. `y = β₀ + e^(β₁x)`; moving the coefficient outside (`y = β₀ + β₁e^x`) makes it linear again.
- Demo seeds/shapes: linear data `np.random.seed(42)`, `x = linspace(-2, 2, 50)`, `y = 1 + x + Normal(0, 0.5)`; sine data `x = linspace(0, 7, 100)`, `y = sin(x) + Normal(0, 0.2)`; polynomial `np.random.seed(9)`, `y = 3x² + Normal(0, 0.5)`.
- Takeaway: curved data does **not** rule out linear regression; transform (sin, log, exp, powers) and fit the transformed feature.

---

> **See also:** the lecture note "Linear Regression Assumptions - Normality of Errors (Cliff Notes)" (assumption 2 in depth), and "Linear Regression Formula (Cliff Notes)" for the base equation these assumptions sit on.

---

*Source: CMPIF2100 lab notebook (personal study use).*
