# Module 12 - Cliff Jumper Notes

*One continuous lesson stitched from the seven Module 12 cliff notes (Fitting
Linear Regression Models with Statsmodels, lecture and lab; P-values and
Confidence Intervals on Coefficient Estimates, lecture and lab; Making
Predictions with Linear Regression, lab; and Visualizing Uncertainty on Linear
Regression Predictions, lecture and lab). The through-line: **Module 11 built
linear regression on paper; Module 12 runs it for real in Python, and then asks
the question that separates modeling from guessing: how much should I trust
what it says?** That question gets asked twice, once about the **coefficients**
and once about the **predictions**, and the module answers both in five moves.
First **fit** the model with the statsmodels formula API. Then **read** the fit
(`summary()` and `.params`). Then **trust-check the coefficients** (standard
errors, confidence intervals, p-values). Then **predict** on new inputs
(`.predict()` over a grid). Finally **trust-check the predictions** (confidence
interval on the mean vs the prediction interval on individuals).*

> All code targets **statsmodels** (formula interface, `smf`), **pandas**,
> **numpy**, **seaborn**, and **matplotlib** with the standard aliases. One
> dataset runs through the whole module: the classic **food-truck** data (97
> cities, columns `population` and `profit`), loaded live from a GitHub URL, so
> no local CSV is needed. Every number below was reproduced by execution
> (statsmodels 0.14.5, pandas 2.3.3, numpy 2.3.5), and the described plot
> readings are the observations the source notes report.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
```

---

## 1. The map: one dataset, five moves

Module 11 ended with the linear regression formula, its coefficients, and its
assumptions, all worked by hand with made-up coefficients. Module 12 hands the
estimation to software and then spends most of its time on **uncertainty**:

| The move | The idea | The tool | Section |
|---|---|---|---|
| **Fit** | describe the model as a formula, estimate it from data | `smf.ols("profit ~ population", data).fit()` | 2 |
| **Read** | the fitted line and how well it fits | `.summary()`, `.params` | 3 |
| **Trust the coefficients** | how sure are we of the slope? | `.bse`, `.conf_int()`, `.pvalues` | 4 |
| **Predict** | outputs for inputs never seen | `.predict(grid)` | 5 |
| **Trust the predictions** | how sure are we of each prediction? | `.get_prediction(grid).summary_frame()` | 6 |

The running example: you work for a food-truck franchise deciding which kind of
city to expand into. You have the **profit** of existing trucks and the
**population** of their cities, one input and one output. Sections 2 and 3 build
and read the model; section 4 is uncertainty about the *parameters*; sections 5
and 6 are prediction and uncertainty about the *outputs*.

```python
profits = pd.read_csv(
    'https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt',
    header=None,
    names=['population', 'profit'])
```

Verified: **97 rows, 2 float64 columns, no missing values**. The raw file has no
header row, hence `header=None` plus explicit `names`. Population runs
**5.0269 to 22.203** (mean about 8.16); units are unspecified in the course (in
the dataset's original home they are tens of thousands of people and $10,000s of
profit, which is why the numbers are small).

**EDA before modeling, always.** A quick scatter
(`sns.relplot(data=profits, x='population', y='profit')`) shows a **positive**,
roughly linear relationship: bigger cities, more profit. Some profits are
**negative** (a truck is not guaranteed to profit). Module 10 would have
measured this with a correlation; Module 12 measures it with a **slope**.

---

## 2. Fit: the statsmodels formula API

Statsmodels is the course's package for estimating linear regressions. The
import of note is the **formula API**:

```python
import statsmodels.formula.api as smf
```

(API = application programming interface, a way of driving another package
through code. Statsmodels has several interfaces; this course uses the formula
one.)

**The formula string.** You describe the model as a compact string, a syntax
borrowed from R:

```text
"profit ~ population"
```

Read the tilde as **"is a function of"**: the **output goes on the left**, the
**input(s) on the right**, and both must be **actual column names** in your
DataFrame. The string stands for the Module 11 equation `ŷ = β₀ + β₁x`, with
one convenience: **you never write the intercept; statsmodels adds β₀
automatically**.

**Fitting takes two arguments and one call:**

```python
model = smf.ols(formula='profit ~ population', data=profits).fit()
```

- **`smf.ols(formula, data)`** builds the model object (ols = **ordinary least
  squares**, the method that estimates the coefficients by minimizing the sum of
  squared errors). On its own it only initializes.
- **`.fit()`** trains it: estimates β₀ and β₁, finding the best-fit line on your
  data. Chaining both on one line is the normal pattern.
- Spelling matters: in the formula API the callable is **lowercase `smf.ols`**
  (the lecture says "the OLS function" out loud, but `smf.OLS` errors; uppercase
  `OLS` lives in the non-formula interface).

---

## 3. Read: `summary()` and `params`

**The whole report:**

```python
model.summary()
```

prints the "OLS Regression Results" table. Verified highlights:

```text
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -3.8958      0.719     -5.415      0.000      -5.324      -2.467
population     1.1930      0.080     14.961      0.000       1.035       1.351

R-squared: 0.702    No. Observations: 97    Df Residuals: 95
```

- **The `coef` column is the model.** The fitted line is
  **`profit = -3.8958 + 1.1930 x population`**.
- **R-squared = 0.702**: the model explains about 70% of the variance in profit.
- The other coefficient columns (`std err`, `P>|t|`, the `[0.025 0.975]`
  interval) are exactly what section 4 unpacks.
- A displayed p-value of **`0.000` means "smaller than 0.0005," never literally
  zero** (the true values here are 4.6e-07 and 1.0e-26).

**Just the numbers:**

```python
model.params
```

returns the coefficients as a pandas Series. Verified:

```text
Intercept    -3.895781
population    1.193034
```

Reading them the Module 11 way: the **slope 1.19** says each one-unit increase
in population is associated with about **1.19 more profit** (positive, matching
the scatter); the **intercept -3.90** is the predicted profit at population 0
(a negative extrapolation artifact, not a data error).

---

## 4. Trust the coefficients: SE, CI, p-value

The fitted slope is a **point estimate from one sample**. This year's 97 trucks
are one draw from an unreachable population of possible food-truck outcomes;
next year's data would give a slightly different best-fit line. Fitting over and
over across imagined repeated samples would produce a **spread of slopes**, and
the model can estimate that spread. Three tools, one story:

**Standard error, `model.bse`.** Verified:

```text
Intercept     0.719483
population    0.079744
```

Small SE = tightly pinned down. The slope's SE (0.08) is tiny relative to the
estimate (1.19), so there is little wiggle room on it; the intercept's is
looser.

**95% confidence interval, `model.conf_int()`.** Verified (column `0` = lower
bound, column `1` = upper):

```text
                   0         1
Intercept  -5.324135 -2.467427
population  1.034722  1.351345
```

The course's rule of thumb: **CI ≈ estimate ± 2 x SE** (central limit theorem,
Gaussian sampling distribution). Worked on the intercept: -3.90 ± 2(0.72) gives
about [-5.33, -2.46], matching the exact [-5.324, -2.467]. Precision note,
verified: the true multiplier is the **t critical value** with n - 2 = 95
degrees of freedom, **1.9853**, close to 2 and to the normal 1.96, so "about
two" is honest shorthand.

How to read a CI, stated with the same care as Module 4: the 95% refers to the
**procedure**, across many repeated samples about 95% of such intervals would
bracket the true coefficient. Avoid "there is a 95% chance the true value is in
this interval."

**P-values, `model.pvalues`.** Verified:

```text
Intercept     4.607887e-07
population    1.023210e-26
```

(The `e` is scientific notation: 1.0e-26 = 10⁻²⁶.) In practice p-values are
what you reach for most. The test is against the **null hypothesis that the
true coefficient is 0**, and 0 means **no relationship**: a zero slope returns
the same output no matter the input. The working rule:

- **p < 0.05: statistically significant.** You can be confident the coefficient
  is not 0, and in particular you can trust its **sign** (a positive estimate
  reflects a genuinely positive association) even if not its exact value.
- **p > 0.05: you cannot trust the coefficient or even its sign.** As the
  lecturer's PhD advisor put it, if it is not significant, you have to ignore
  it; you cannot say anything about that relationship.

A convenient shortcut when the `e` numbers get hard to read:
`model.pvalues < 0.05` returns booleans (here `True` for both).

> **Correction the course needs (read before a quiz).** The lecture and lab both
> define the p-value as *"the chance that the true coefficient is actually
> zero."* That is the classic misinterpretation. The p-value is the probability,
> **computed assuming the true coefficient is 0**, of getting an estimate **at
> least as far from 0** as the one observed. It is not the probability the null
> is true, and p < 0.05 does not mean "95% sure it is nonzero" (0.05 is a
> significance threshold). **Exam safety:** if a quiz keys to the course's
> wording, answer the course's way; just know the precise definition.

**Seeing it: the error-bar plot.** CI and p-value tell the same story, visible
by plotting each coefficient as a dot with its CI as a vertical segment against
a dashed reference line at 0 (build a small DataFrame from
`model.conf_int().rename(columns={0: 'ci_lower', 1: 'ci_upper'})` plus a `coef`
column, then layer `sns.scatterplot` calls for bounds and estimate, `ax.vlines`
between the bounds, and `ax.axhline(y=0, linestyle='--')`). Verified reading:
the intercept's bar sits entirely below zero, the slope's entirely above, so
**neither CI touches 0 and both coefficients are significant**. The mechanical
rule (the lecture verbally flipped it once; this is the right pairing):

- **CI excludes 0 ⇔ p < 0.05** (significant, sign trustworthy).
- **CI contains 0 ⇔ p > 0.05** (not significant, no confidence even in sign).

---

## 5. Predict: a grid in, a trend line out

Fitted models predict. The workflow that produces a smooth trend line:

**Build a visualization grid** of evenly spaced fake inputs spanning (and
slightly overshooting) the observed range:

```python
viz_grid = pd.DataFrame({
    'population': np.linspace(profits.population.min() - 0.1,
                              profits.population.max() + 0.1,
                              num=101)})
```

Verified: 101 points from **4.9269 to 22.303**. Familiar Module 11 details: the
grid overshoots each end by 0.1 to give the line breathing room, and `num=101`
(a round number plus one) makes 100 clean intervals, since linspace counts
points, not gaps. The column **must be named `population`**, because `.predict()`
matches predictors to formula names.

**Predict on the grid, fit on the data, never the reverse:**

```python
lm_pred = model.predict(viz_grid)
```

Verified: a Series of 101 point predictions, running **1.9822** (at population
4.9269) up to **22.7124** (at 22.303). Hand-check that `.predict()` is nothing
mystical: `-3.895781 + 1.193034 x 4.9269 = 1.98218`, exactly the first value.
It plugs each input into the fitted line.

**Plot the trend over the data.** Store the predictions
(`viz_grid['pred_trend'] = lm_pred`), then layer an **axes-level** line and
scatter on one `ax` (the same figure-level vs axes-level rule the visualization
modules taught: to combine layers, use axes-level functions sharing an `ax`):

```python
fig, ax = plt.subplots()
sns.lineplot(data=viz_grid, x='population', y='pred_trend', ax=ax)
sns.scatterplot(data=profits, x='population', y='profit', color='black', ax=ax)
plt.show()
```

The straight line runs through the cloud of training points, which scatter
around it. That scatter is the error term, and it is the whole subject of
section 6.

> **What `.predict()` does not give you: any measure of uncertainty.** It
> returns the point estimate of the mean, nothing else. For intervals you need
> `get_prediction`, next.

---

## 6. Trust the predictions: confidence interval vs prediction interval

Module 11's fine print said linear regression predicts the **mean** outcome at
each input, while individual observations scatter around that mean (errors
`~ Normal(0, σ)`). So "how sure are we?" has **two different answers**:

| | Question it answers | `summary_frame()` columns | Width |
|---|---|---|---|
| **Confidence interval (CI)** | where is the **true average** profit at this population? | `mean_ci_lower`, `mean_ci_upper` | narrow |
| **Prediction interval (PI)** | where will **one new** truck's profit land? | `obs_ci_lower`, `obs_ci_upper` | wide |

**The workhorse call.** Instead of `.predict()`:

```python
pred_summary = model.get_prediction(viz_grid).summary_frame()
```

returns one row per grid point with six columns:
**`mean`, `mean_se`, `mean_ci_lower`, `mean_ci_upper`, `obs_ci_lower`,
`obs_ci_upper`**. Verified first row (population 4.9269):

```text
    mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
1.982177  0.400893       1.186304       2.778049     -4.073054      8.037408
```

`mean` is the same point prediction `.predict()` gave; `mean_se` is its
standard error; the `mean_ci_*` pair is the CI on the mean; the `obs_ci_*` pair
is the prediction interval. Naming trap worth flagging: statsmodels calls the PI
bounds "obs_ci," and the lecture itself calls them "these weird columns."
**`obs_ci_*` is the prediction interval, not a second confidence interval.**
(The default level is 95%; `summary_frame(alpha=0.10)` would give 90% bands.)

**The two formulas (verified by hand).** Both intervals are
`mean ± t x (a standard error)` with the same t = 1.9853; what differs is the
standard error inside:

- **CI = mean ± t x mean_se.** By hand at row 0: [1.186304, 2.778049], matching
  `mean_ci_*` exactly.
- **PI = mean ± t x sqrt(mean_se² + σ²)**, where **σ² is the residual variance**
  (`model.scale`, verified 9.1424, so σ ≈ 3.02). By hand at row 0:
  [-4.073054, 8.037408], matching `obs_ci_*` exactly.

That extra σ² term is the intuition: the CI only carries uncertainty about
**where the line sits**; the PI must also absorb the **scatter of individual
points around the line**, which no amount of data shrinks. Verified: the PI is
wider than the CI at **all 101 grid points** (row-0 widths: CI 1.59 vs PI
12.11), and a PI can dip **negative** (a truck can run at a loss).

**The temperature analogy (lecture).** Measure the temperature on the same
calendar day for 100 years: the **CI** bounds where the **average** for that day
sits; the **PI** bounds where **any single year's** measurement lands.

**Where the CI pinches.** The gray CI ribbon is not uniform: verified, it is
**narrowest at population ≈ 8.2, essentially the input mean (8.16), and widest
at the range's edges** (width 1.22 at the pinch vs 1.59 and 4.64 at the ends).
The lecture explains this as "more data points there"; that intuition works here
because the cities cluster toward the lower populations, near the input mean,
but the precise statement is that the mean prediction's standard error **grows
with distance from the mean of x**.
Exam safety: if a quiz keys to the course's wording, answer "the CI shrinks
where there are more observations."

**Drawing the ribbons.** Line, then `fill_between` bands, then the training
scatter, all on one axes-level `ax` (the lecture drew the CI in gray, then added
the PI in orange):

```python
fig, ax = plt.subplots(figsize=(4, 4))
sns.lineplot(x=viz_grid['population'], y=pred_summary['mean'], ax=ax)

# wide ribbon: the prediction interval
ax.fill_between(viz_grid['population'], pred_summary['obs_ci_lower'],
                pred_summary['obs_ci_upper'], facecolor='orange', alpha=0.5)

# narrow ribbon on top: the confidence interval on the mean
ax.fill_between(viz_grid['population'], pred_summary['mean_ci_lower'],
                pred_summary['mean_ci_upper'], facecolor='grey', alpha=0.5)

sns.scatterplot(data=profits, x='population', y='profit', ax=ax)
plt.show()
```

Verified reading: the narrow gray band hugs the line; the wide band swallows
almost all of the scatter, with a few outliers legitimately outside.

**The coverage check that makes it stick.** Counting the 97 training points
against each band (verified): **93 of 97 (95.9%) fall inside the 95% PI**,
exactly what a 95% prediction interval promises, while only **22 of 97 (22.7%)
fall inside the CI**, because the CI never claimed to contain points, only the
average. If an uncertainty band captures nearly all your scatter, you are
looking at a prediction interval.

**One worked prediction (verified), population = 15:** best guess for the
average profit is **14.00**, with 95% CI **[12.76, 15.24]** for the average and
PI **[7.87, 20.13]** for any single truck. Same fitted line, two very different
"plus or minus" statements.

---

## One-paragraph recap

Module 12 takes Module 11's linear regression into statsmodels and interrogates
its trustworthiness twice. **Fit:** `smf.ols(formula='profit ~ population',
data=profits).fit()`, where the R-style formula string puts the output left of
the tilde, inputs right, names must match DataFrame columns, and the intercept
is added automatically; ols = ordinary least squares, lowercase in the formula
API. **Read:** `.summary()` prints the "OLS Regression Results" table and
`.params` the coefficients; on the 97-city food-truck data the fitted line is
`profit = -3.8958 + 1.1930 x population` with R-squared 0.702, and a printed
p-value of 0.000 means "less than 0.0005," never zero. **Trust the
coefficients:** each estimate carries a standard error (`.bse`: slope 0.0797), a
95% CI (`.conf_int()`: slope [1.035, 1.351], approximately estimate ± 2 x SE,
exactly a t multiplier of 1.9853), and a p-value (`.pvalues`: slope 1.0e-26);
p < 0.05 means significant with a trustworthy sign, the CI-vs-0 error-bar plot
shows it, and CI excludes 0 exactly when p < 0.05. The p-value is the
probability of data this extreme **assuming** the coefficient is 0, not the
probability that it is 0 (answer the course's way if a quiz keys to its looser
wording). **Predict:** build a 101-point `np.linspace` grid padded 0.1 past the
data, keep the formula's column name, and `.predict(grid)` returns point
estimates of the mean (1.98 up to 22.71), which plot as a straight trend line
through the scatter via axes-level lineplot + scatterplot. **Trust the
predictions:** `.get_prediction(grid).summary_frame()` adds the intervals:
`mean_ci_*` is the confidence interval on the **mean** (narrow, mean ± t x
mean_se, pinched near the input mean), `obs_ci_*` is the **prediction interval**
on an **individual** observation (wide, mean ± t x sqrt(mean_se² + σ²) with
σ² = `model.scale` ≈ 9.14, never shrinking away); drawn as `fill_between`
ribbons, the PI swallowed 95.9% of the training points (a 95% interval keeping
its promise) while the CI contained only 22.7%, because averages are not
observations.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
