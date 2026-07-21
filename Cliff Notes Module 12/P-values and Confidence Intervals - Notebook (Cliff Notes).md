# P-values and Confidence Intervals on Coefficient Estimates (Lab Notebook)

**Module 12 Cliff Notes** | Source: lab notebook `p-values_linear_regression_start.ipynb`

---

## TL;DR

- The notebook refits the **food-truck model** (`profit ~ population` via `statsmodels`) and then quantifies the model's **uncertainty** about its coefficients using three tools: the **standard error** (`.bse`), the **95% confidence interval** (`.conf_int()`), and the **p-value** (`.pvalues`).
- Each coefficient estimate (slope and intercept) is itself an **estimate of a mean** and therefore carries a **standard error**. The **95% CI is roughly the estimate ± 2 × standard error** (the exact multiplier is the t-critical value, **1.985** here, close to 2).
- A **p-value < 0.05** on a coefficient means we can be confident the true coefficient is **not 0**, so the input/output association is **statistically significant** and its **sign** (positive or negative) is trustworthy.
- **CI and p-value tell the same story:** if the 95% CI **excludes 0**, the p-value is **< 0.05**; if it **includes 0**, the p-value is **> 0.05**. The final demo plots the coefficients with CI error bars and a dashed line at 0 to read this off visually.
- **Two wording slips in the lab are corrected below** (what a p-value actually is, and a "p-value less than 0" typo). The correct frequentist statements are used in this note.

> **The one takeaway:** a coefficient estimate is a point guess with a spread around it; the **standard error, confidence interval, and p-value** all describe that spread, and they agree on whether the relationship is real (does the CI clear 0?).

---

## Setup: refit the food-truck model

The notebook loads the food-truck dataset and refits the simple linear regression from the previous module. Knowing the coefficients tells you the **relationship** between population and profit; this notebook is about how **sure** we are of those numbers.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

profits = pd.read_csv(
    'https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt',
    header=None,
    names=['population', 'profit'])
profits.info()
profits.head()
```

The data is pulled from a **URL** (not a local file), so no CSV lives in the Module 12 folder. Verified by fresh run: **97 rows, 2 columns** (`population`, `profit`), both `float64`, no missing values. First row is `population 6.1101, profit 17.5920`.

```python
model = smf.ols(formula='profit ~ population', data=profits).fit()
model.params
```

Verified fitted coefficients:

```text
Intercept    -3.895781
population    1.193034
```

So the fitted line is **profit ≈ -3.8958 + 1.1930 × population**. The **intercept** is the estimated output when population is 0 (here a negative, non-physical profit, which is fine as an extrapolation artifact), and the **slope** says each unit of population adds about **1.19** to profit.

---

## Confidence intervals on coefficient estimates

The key mental model: a coefficient estimate is an **estimate of a mean** (the mean output across a range of inputs), and just like any sample-mean estimate, it has a **standard error**.

```python
model.bse
```

Verified standard errors:

```text
Intercept     0.719483
population    0.079744
```

The **95% confidence interval** is available directly with `.conf_int()`. The lab's rule of thumb: the interval is **about the estimate ± 2 × standard error** (column `0` is the lower bound, column `1` is the upper bound).

```python
model.conf_int()
```

Verified bounds:

```text
                   0         1
Intercept  -5.324135 -2.467427
population  1.034722  1.351345
```

**Precision note (verified, not a discrepancy):** the "± 2 × SE" rule is an **approximation**. `statsmodels` actually multiplies the SE by the **t-critical value** for 95% with `n - 2 = 95` degrees of freedom, which is **1.985251**, not exactly 2. For the slope: `1.193034 ± 1.985251 × 0.079744 = [1.034722, 1.351345]`, matching `.conf_int()` exactly. Using a flat 2 would give the slightly-too-wide `[1.033546, 1.352522]`. Close enough for a mental estimate, but the real multiplier is the t-value (which approaches **1.96** for large samples).

> **What a confidence interval actually means (frequentist care):** a 95% CI is produced by a procedure that, **across many repeated samples**, brackets the true coefficient **95% of the time**. It is **not** a "95% probability that this particular interval contains the true value." The true coefficient is a fixed (unknown) constant; the **interval** is the random thing that changes sample to sample. The notebook does not commit the "95% chance the true value is inside" fallacy in its text, so no correction is needed there, but state it the careful way on an exam unless the question keys to looser wording.

---

## P-values on coefficient estimates

Depending on how much data you have, the 95% CI might **include 0**. A coefficient of **0 means no relationship** between that input and the output (the "null hypothesis"). The p-value measures how compatible the data is with that null.

```python
model.pvalues
```

Verified p-values:

```text
Intercept     4.607887e-07
population    1.023210e-26
```

Both are far below 0.05 (the slope's is astronomically small, `1.0e-26`), so **both coefficients are highly statistically significant**. Because the slope is positive with p < 0.05, we can be confident the **true relationship is positive** (more population, more profit).

> **Correction 1 (course wording is loose):** the lab says, *"P-values represent the chance of this 'null hypothesis' that the true coefficient is 0."* That is the classic **p-value fallacy**. A p-value is **not** the probability that the null hypothesis is true. Correct statement: the p-value is the probability, **computed while assuming the true coefficient is exactly 0**, of observing a coefficient estimate **at least as far from 0** as the one we got. Low p-value means such data would be very unlikely under a true-zero coefficient, so we reject the null. **Exam safety:** use the correct definition; but if a quiz keys to the course's wording ("chance of the null hypothesis"), answer its way.

The rest of the lab's interpretation is fine: **p < 0.05 → confident the coefficient is not 0 → the association is significant and its sign is trustworthy.**

---

## Visualize confidence intervals on coefficients

Plotting the coefficient estimates with their 95% CIs as error bars lets you **see** whether each interval clears **0**. If an interval **contains 0**, the p-value is **> 0.05** (sign not trustworthy); if it **excludes 0**, the p-value is **< 0.05** (sign trustworthy).

First build a tidy table of the estimate plus its bounds:

```python
coef_info = model.conf_int().rename(columns={0: 'ci_lower', 1: 'ci_upper'})
coef_info['coef'] = model.params
coef_info
```

Verified:

```text
            ci_lower  ci_upper      coef
Intercept  -5.324135 -2.467427 -3.895781
population  1.034722  1.351345  1.193034
```

Then plot the estimate as a dot, the bounds as tick marks, the CI as a vertical segment, and a dashed reference line at **y = 0**:

```python
fig, ax = plt.subplots(figsize=(3, 3))
ax.margins(x=0.2)  # whitespace around the x-axis

# coefficient point and the two CI bounds
sns.scatterplot(data=coef_info, x=coef_info.index, y='ci_lower', color='black', marker='_', ax=ax)
sns.scatterplot(data=coef_info, x=coef_info.index, y='coef',     color='black', marker='o', ax=ax)
sns.scatterplot(data=coef_info, x=coef_info.index, y='ci_upper', color='black', marker='_', ax=ax)

# confidence intervals as vertical line segments
ax.vlines(x=coef_info.index, ymin=coef_info['ci_lower'], ymax=coef_info['ci_upper'], color='black', linewidth=1)

# horizontal reference line at 0
ax.axhline(y=0, linestyle='--', linewidth=1, color='gray')

ax.set_xlabel('input variable')
ax.set_ylabel('coefficient value')
plt.show()
```

Reading the plot: the **intercept** interval sits entirely **below 0** (`[-5.32, -2.47]`) and the **population** interval sits entirely **above 0** (`[1.03, 1.35]`). Neither error bar touches the dashed line, so **both coefficients are significant**, consistent with the tiny p-values above.

> **Correction 2 (typo in the lab):** the lab's markdown says, *"If the confidence interval on the coefficient doesn't contain 0, the p-value is less than 0 and we can be confident about the sign of the coefficient."* A **p-value can never be less than 0** (probabilities live in `[0, 1]`). This is a typo for **"less than 0.05."** The intended rule is correct: **CI excludes 0 ⇔ p-value < 0.05.**

---

## Quiz-ready facts

- **Model:** `smf.ols('profit ~ population', data=profits).fit()`; fitted line **profit ≈ -3.8958 + 1.1930 × population**. Data is 97 rows, loaded from a URL.
- **Standard error** via `model.bse`: intercept **0.7195**, slope **0.0797**. It is the standard deviation of the coefficient's sampling distribution (the estimate is an estimate of a mean).
- **95% CI** via `model.conf_int()`: intercept **[-5.3241, -2.4674]**, slope **[1.0347, 1.3513]**. Column `0` = lower, column `1` = upper.
- **CI ≈ estimate ± 2 × SE** is an approximation; the true multiplier is the **t-critical value** (here **1.985** with df = n - 2 = 95, approaching **1.96** for large n).
- **P-values** via `model.pvalues`: intercept **4.61e-07**, slope **1.02e-26**. Both ≪ 0.05, so both are significant.
- **A p-value is** the probability of a coefficient estimate at least this far from 0 **assuming the true coefficient is 0**; it is **not** the probability that the null hypothesis is true.
- **A 95% CI is** an interval from a procedure that captures the true value 95% of the time over repeated samples; it is **not** a 95% probability the fixed true value sits in this one interval.
- **CI-vs-p-value rule:** CI **excludes 0** ⇔ **p < 0.05** (sign trustworthy); CI **includes 0** ⇔ **p > 0.05** (no confidence in sign).
- **Cross-check (beyond the notebook, verified):** all of these appear in `model.summary()`, where `std err` = `.bse`, `[0.025  0.975]` = `.conf_int()`, and `P>|t|` = `.pvalues`; R-squared here is **0.702**.

---

> **See also:** the lecture note "P-values and Confidence Intervals on Coefficient Estimates in Linear Regression (Cliff Notes)" (the companion lecture on the same topic), and "Fitting Linear Regression Models - Notebook (Cliff Notes)" for building the underlying `smf.ols` model these intervals sit on.

---

*Source: CMPIF2100 lab notebook (personal study use).*
