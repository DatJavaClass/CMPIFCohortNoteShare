# Fitting Linear Regression Models with Statsmodels: building your first OLS model from a formula

**Module 12 Cliff Notes** | Source: lecture transcript "Fitting Linear Regression Models with Statsmodels"

---

## TL;DR

- **Statsmodels** fits linear regression in Python. Import the **formula interface** with `import statsmodels.formula.api as smf` (API means "application programming interface," a way to drive another package through code).
- You describe the model with a **formula string** shaped `"output ~ input"` (read the tilde `~` as **"is a function of"**). The **output/response** goes on the **left**, the **inputs/predictors** on the **right**. This syntax is borrowed from R.
- Build and fit in two moves: **`model = smf.ols(formula="profit ~ population", data=profits)`** then **`result = model.fit()`**. The `smf.ols` call just sets the model up (it "does nothing" on its own); **`.fit()`** estimates the coefficients (finds the best-fit line by **OLS**, ordinary least squares).
- The names in the formula must be **actual column names** in your data. **You do not specify an intercept**: statsmodels adds it automatically.
- Read the fitted model two ways: **`result.summary()`** prints the full "OLS Regression Results" table (R-squared, coefficients, p-values, confidence intervals), and **`result.params`** returns just the coefficients as a pandas Series.
- In the food-truck example the population slope came out to **≈ 1.19**: each one-unit increase in population is associated with about a **1.19-unit increase in profit** (a positive relationship, as the scatter plot suggested).

> **One-line takeaway:** `smf.ols("y ~ x", data).fit()` estimates a linear regression; `.summary()` shows everything and `.params` hands you the coefficients.

---

## Why statsmodels, and the formula API

Linear regression lets you do two things: **describe/measure the relationship** between input and output variables, and **predict output values for inputs you have not seen yet**. Statsmodels does both.

Alongside the usual stack (pandas, numpy, seaborn, matplotlib) you import statsmodels itself. This lecture uses the **formula API** specifically:

```python
import statsmodels.formula.api as smf
```

An **API** (application programming interface) is just a way of calling another program or package through code. Statsmodels offers more than one interface; the **formula interface** is the one used here because the formula string is compact and readable.

**Precise-code note:** in the formula API the callable is spelled **lowercase**, `smf.ols(...)`. The lecture refers to it verbally as "the OLS function," and "OLS" does stand for **ordinary least squares** (the method that estimates the coefficients), but if you type `smf.OLS` you will get an error. Uppercase `OLS` belongs to a different (non-formula) interface. When you use `smf`, write `smf.ols`.

## The running example: a food-truck company

Pretend you work for a food-truck franchise deciding **which type of city to expand into**. You have data on **profit** earned by trucks in various cities and the **population** of each city. That gives one input (population) and one output (profit). The exact units are unknown (population could be in tens of thousands, profit in some unspecified scale), which does not change the mechanics.

**Always do exploratory data analysis before modeling.** A quick scatter plot with population on the x-axis and profit on the y-axis (for example seaborn's figure-level `sns.relplot(...)`) shows a **positive relationship**: as population rises, profit tends to rise, so bigger cities look more profitable. Note also that **some profit values are below zero**, which is realistic (a truck is not guaranteed to profit).

You could summarize that relationship with a **correlation**, but here you measure it instead with the **slope of the regression line**, the estimated coefficient on population.

## Fitting the model: two arguments, then `.fit()`

The OLS routine takes **two arguments**:

1. a **formula** (a string describing the model), and
2. a **training dataset** (a DataFrame; parameters are always estimated from data).

```python
model  = smf.ols(formula="profit ~ population", data=profits)
result = model.fit()
```

- **`smf.ols(...)`** creates the model object but does not estimate anything yet; on its own it essentially just initializes.
- **`.fit()`** trains the model: it estimates the parameters, finding the **best-fit line** on your data by ordinary least squares.

Statsmodels looks up `profit` and `population` as **columns in `profits`**, so the names in the formula must match real columns. (This is a bit different from scikit-learn's fit style, but the idea is the same.)

## The formula string, decoded

The formula is a plain **string** you build up to represent the model. With an output `Y` and an input `X`, the shape is:

```text
"Y ~ X"
```

The tilde `~` reads naturally as **"Y is a function of X."** The rule:

- **Left of `~`:** the **output/response** variable (here `profit`).
- **Right of `~`:** the **input/predictor** variable(s) (here `population`). Multiple inputs are added later; this example uses one.

This string stands for the linear regression model you have already seen mathematically:

```text
ŷ = β₀ + β₁·x
```

where **ŷ** is the estimated mean output at a given x, **β₀** is the intercept, and **β₁** is the slope (the coefficient on the input). **You never write the intercept in the formula**: statsmodels **assumes and adds β₀ for you**, which is convenient. The coefficient you most care about is β₁ on `population`, because it quantifies **how strongly, and in which direction, the input is associated with the output**.

## Reading the fitted model

**Full report, `result.summary()`:**

```python
result.summary()
```

This prints the **"OLS Regression Results"** table ("OLS regression" just means linear regression). It packs in a lot:

- **Goodness-of-fit** measures such as **R-squared** (how closely the model fits the observed training data).
- The **coefficients** themselves, plus their **standard errors**, **t-values**, **p-values** (the `P>|t|` column), and **95% confidence intervals** (the `[0.025  0.975]` columns). P-values and confidence intervals are covered in their own lecture.

In this example the summary shows a **negative intercept** and a **positive `population` coefficient**, consistent with the upward-sloping scatter plot.

**Just the coefficients, `result.params`:**

```python
result.params
```

When you want the numbers directly instead of the whole table, `.params` returns a **pandas Series** indexed by term name (`Intercept` and `population`).

Verified fit on the food-truck data:

```text
Intercept    -3.8958
population    1.1930
```

(R-squared ≈ 0.70 on this dataset.)

## Interpreting the coefficient

The population coefficient of **≈ 1.19** is your measured relationship: **for each one-unit increase in population (in whatever the population units are), profit increases by about 1.19 units**, on average. It is **positive**, confirming the "bigger cities, more profit" pattern the scatter plot hinted at. That single number is the payoff, a concrete measure of the association between the input (population) and the output (profit), and the thing you would lean on when deciding where to expand.

---

## Quiz-ready facts

- **Import:** `import statsmodels.formula.api as smf`. **API** = application programming interface (calling a package through code).
- **OLS** = **ordinary least squares**, the method that estimates the coefficients. Formula-API callable is lowercase **`smf.ols`** (uppercase `smf.OLS` is not in the formula API).
- **Two arguments to `smf.ols`:** a **formula** (string) and a **data** set (DataFrame).
- **Formula shape:** `"output ~ input"`; **output on the left**, **inputs on the right**; `~` reads as "is a function of." Syntax is **R-inspired**. Formula names must be **real column names** in the data.
- **`smf.ols(...)` only initializes; `.fit()` trains** (estimates parameters, finds the best-fit line).
- **The intercept is added automatically**; you do not put it in the formula.
- **`result.summary()`** prints the full "OLS Regression Results" table (R-squared, coef, std err, t, `P>|t|`, 95% CI). **`result.params`** returns just the coefficients as a **pandas Series**.
- **Food-truck fit:** intercept ≈ **-3.90** (negative), population slope ≈ **1.19** (positive). Slope 1.19 means **+1 unit population ⇒ +1.19 units profit**.
- **EDA first:** scatter population (x) vs profit (y) shows a **positive relationship**; some profits are **below zero** (profit is not guaranteed).

---

> **See also:** "Fitting Linear Regression Models - Notebook (Cliff Notes)" (the companion lab that runs this code end to end), and "P-values and Confidence Intervals on Coefficient Estimates in Linear Regression (Cliff Notes)" (reading the rest of the summary table).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
