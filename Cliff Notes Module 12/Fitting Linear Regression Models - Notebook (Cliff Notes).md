# Fitting Linear Regression Models with `statsmodels` (Lab Notebook)

**Module 12 Cliff Notes** | Source: lab notebook `fitting_linear_regression_models_start.ipynb`

---

## TL;DR

- The notebook fits a **single-predictor linear regression** with `statsmodels`' **formula interface**, `import statsmodels.formula.api as smf`, on a food-truck dataset (**profit vs city population**, 97 rows from Andrew Ng's ML course).
- One line does the whole fit: **`model = smf.ols(formula='profit ~ population', data=profits).fit()`**. The formula `'y ~ x'` means **ŷ = β₀ + β₁x**, and the **intercept β₀ is added automatically**, so you never write it in the formula.
- **`.ols(...)` builds the model, `.fit()` trains it** (solves for the coefficients). Chaining both on one line is the normal pattern.
- **`model.summary()`** prints the full results table. The fitted line is **profit = -3.8958 + 1.1930 x population**, with **R-squared = 0.702** (Verified).
- Both coefficients are **highly significant** (population t = 14.96, p ~ 1e-26), so population is a strong linear predictor of profit; the negative intercept just means the line crosses below zero for tiny populations.

> **The one takeaway:** `smf.ols('output ~ input', data=df).fit()` fits an ordinary least squares line in a single line of code, the intercept is implied, and `.summary()` hands you the coefficients, their significance, and R-squared all at once.

---

## Setup: the formula interface

The notebook imports pandas, seaborn, matplotlib, and the **formula API** of statsmodels:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

The key import is **`statsmodels.formula.api as smf`**. This "formula" (R-style) interface lets you describe a model as a short string like `'profit ~ population'` instead of hand-building design matrices. It is the friendliest way to fit regressions in statsmodels.

## The scenario and the data

Premise: you work for a food-truck company deciding which city to expand into. You have **profit** and **population** for cities with existing franchises. The data comes from Andrew Ng's Machine Learning course and is loaded straight from a GitHub URL (no local file):

```python
profits = pd.read_csv('https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt',
                      header=None,
                      names=['population', 'profit'])
profits.info()
profits.head()
```

Notes on the read: the raw file has **no header row**, so `header=None` tells pandas the first line is data, and `names=['population', 'profit']` supplies the column labels.

Verified: the frame is **97 rows, 2 columns**, both `float64`, with **no missing values** (97 non-null each). Verified `head()`:

```text
   population   profit
0      6.1101  17.5920
1      5.5277   9.1302
2      8.5186  13.6620
3      7.0032  11.8540
4      5.8598   6.8233
```

(Context, not stated in the lab: in Ng's original dataset population is in tens of thousands of people and profit in units of $10,000, which is why the numbers are small. The regression works the same regardless of units.)

## Visualize before fitting

Always eyeball the relationship first:

```python
sns.relplot(data=profits, x='population', y='profit', height=3)
plt.show()
```

This is a scatterplot of profit against population. Verified: the cloud slopes **up and to the right** (bigger population tends to mean bigger profit), roughly linear, which is exactly the shape a straight-line model wants to see before you fit one.

## Fit the model with `smf.ols()`

The fitting function is **`smf.ols()`** ("ols" = ordinary least squares). It takes two important arguments:

1. **`formula`**: a string describing the model.
2. **`data`**: the training DataFrame.

The formula is R-inspired. For output `y` and one input `x` (both column names in the data):

```text
'y ~ x'
```

which represents:

```text
ŷ = β₀ + β₁x
```

**You do not write the intercept β₀ in the formula: statsmodels adds it for you.** After building the model with `smf.ols(...)`, call **`.fit()`** to train it (solve for the β coefficients). The notebook chains both on one line:

```python
model = smf.ols(formula='profit ~ population', data=profits).fit()
```

That single statement builds and trains the model, storing the fitted result in `model`.

## Read the results with `model.summary()`

```python
model.summary()
```

Verified full output (fresh run reproduces the notebook's saved output exactly):

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:                 profit   R-squared:                       0.702
Model:                            OLS   Adj. R-squared:                  0.699
Method:                 Least Squares   F-statistic:                     223.8
Date:                Tue, 21 Jul 2026   Prob (F-statistic):           1.02e-26
Time:                        21:12:26   Log-Likelihood:                -243.95
No. Observations:                  97   AIC:                             491.9
Df Residuals:                      95   BIC:                             497.1
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.8958      0.719     -5.415      0.000      -5.324      -2.467
population     1.1930      0.080     14.961      0.000       1.035       1.351
==============================================================================
Omnibus:                       39.986   Durbin-Watson:                   0.994
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              108.142
Skew:                           1.455   Prob(JB):                     3.29e-24
Kurtosis:                       7.276   Cond. No.                         21.4
==============================================================================
```

What the important pieces mean:

- **The `coef` column is the fitted model.** `Intercept = -3.8958` is β₀, `population = 1.1930` is β₁, so the fitted line is:

  ```text
  profit = -3.8958 + 1.1930 x population
  ```

  Verified full-precision coefficients: **Intercept = -3.895781**, **population slope = 1.193034**.
- **Interpreting the slope:** each **one-unit** increase in population is associated with about a **1.19-unit** increase in predicted profit. The **negative intercept** (-3.90) is where the line crosses population = 0; it just reflects the line's geometry (a tiny-population city is predicted to run at a loss), not a data error.
- **`std err`, `t`, `P>|t|`:** the standard error, t-statistic, and p-value for each coefficient. Both p-values print as `0.000`, meaning "smaller than 0.0005," so both terms are strongly statistically significant. Verified exact values: the population coefficient's p-value is **1.02e-26** and the intercept's is **4.61e-07** (the summary rounds both to `0.000`, so `0.000` never means literally zero).
- **`[0.025  0.975]`:** the 95% confidence interval for each coefficient. Verified: population is **[1.035, 1.351]**, intercept is **[-5.324, -2.467]**. Neither interval contains 0, another sign both terms matter.
- **`R-squared = 0.702`:** the model explains about **70%** of the variance in profit. Verified full precision **0.7020316** (**Adj. R-squared = 0.6989**).
- **`F-statistic = 223.8`, `Prob (F-statistic) = 1.02e-26`:** the overall-model significance test; the tiny p-value says the model as a whole is highly significant.
- **`No. Observations = 97`, `Df Residuals = 95`, `Df Model = 1`:** 97 rows, 1 predictor, so residual degrees of freedom = 97 - 1 intercept - 1 slope = 95.

The bottom block (Omnibus, Durbin-Watson, Jarque-Bera, Skew, Kurtosis) reports **diagnostics on the residuals** (tests of normality and autocorrelation). The lab does not analyze them here; later Module 12 notes do.

## Pulling values out programmatically

Beyond the printed table, the fitted `model` exposes the same numbers as attributes you can compute with (handy for predictions and reports). Verified against the fresh run:

- `model.params` -> `Intercept -3.895781`, `population 1.193034`
- `model.rsquared` -> `0.7020316`
- `model.rsquared_adj` -> `0.6988950`
- `model.fvalue` -> `223.826`, `model.f_pvalue` -> `1.02e-26`
- `model.pvalues['population']` -> `1.02e-26`
- `model.conf_int()` -> population `[1.034722, 1.351345]`, intercept `[-5.324135, -2.467427]`

(The notebook itself stops at `.summary()`; these attributes are the same results in machine-readable form and are the standard next step.)

---

## Quiz-ready facts

- Import for the formula interface: **`import statsmodels.formula.api as smf`**.
- Fit function: **`smf.ols(formula='y ~ x', data=df).fit()`**. `ols` = ordinary least squares; `.ols()` builds the model, **`.fit()` trains it**.
- The formula `'y ~ x'` encodes **ŷ = β₀ + β₁x**, and the **intercept is included automatically** (you never write β₀ in the formula string).
- `pd.read_csv(..., header=None, names=[...])` is used because the raw file **has no header row**; the dataset is **97 rows, 2 float64 columns**, no missing values.
- Fitted model (Verified): **profit = -3.8958 + 1.1930 x population**; Intercept -3.895781, slope 1.193034.
- **R-squared = 0.702** (Adj. 0.699): the model explains about 70% of profit's variance.
- Both coefficients are highly significant: population **t = 14.96**, and **p ~ 1.02e-26**; 95% CI for the slope is **[1.035, 1.351]** (excludes 0).
- **F-statistic = 223.8, Prob(F) = 1.02e-26**: the whole model is significant.
- A summary p-value of **`0.000` means "< 0.0005," not exactly zero** (the population term's true p-value is 1.02e-26).
- `model.summary()` prints coefficients, standard errors, t and p values, confidence intervals, R-squared, and residual diagnostics all in one table.

---

> **See also:** the companion lecture note "Fitting Linear Regression Models with Statsmodels (Cliff Notes)" for the concepts behind this code, and "P-values and Confidence Intervals - Notebook (Cliff Notes)" for a deeper read of the `P>|t|` and `[0.025 0.975]` columns.

---

*Source: CMPIF2100 lab notebook (personal study use).*
