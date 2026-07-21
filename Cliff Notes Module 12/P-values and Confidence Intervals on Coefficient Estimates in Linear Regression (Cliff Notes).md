# P-values and Confidence Intervals: how sure is the model about each coefficient?

**Module 12 Cliff Notes** | Source: lecture transcript "P-values and Confidence Intervals on Coefficient Estimates in Linear Regression"

---

## TL;DR

- A fitted regression gives you **point estimates** for each coefficient (via least squares), but a point estimate is just a guess from **one sample**. Different samples (next month, last year) would give slightly different slopes, so every coefficient carries **uncertainty** you must report.
- **Standard error** (`model.bse`) measures that uncertainty per coefficient: a **small** SE means the estimate is **tightly pinned down**, a **large** SE means it is **wobbly**. In the food truck model the slope SE (0.0797) is tiny and the intercept SE (0.7195) is larger.
- The **95% confidence interval** (`model.conf_int()`) is roughly **coefficient ± 2 × SE** (statsmodels actually uses the t multiplier, about 1.99 here, so "2" is a good round number). It gives a **plausible range** for the coefficient.
- **P-values** (`model.pvalues`) test the **null hypothesis that the true coefficient is 0**. A coefficient of 0 means **no relationship** (the input does not move the output). **p < 0.05** ⇒ **statistically significant**: you can trust the **sign** of the coefficient even if not its exact value.
- **Confidence interval and p-value agree:** if the 95% CI **excludes 0**, then **p < 0.05** (significant); if the CI **contains 0**, then **p > 0.05** (not significant, cannot trust the sign). Plotting the CIs as error bars against a reference line at 0 shows this at a glance.

> **One-line takeaway:** the point estimate is the model's best guess; the **standard error, confidence interval, and p-value** together tell you **how much to trust it**, and the rule of thumb is **p < 0.05 or ignore the relationship**.

---

## Setup: recreate the food truck model

Same regression as before, predicting food truck **profit** from city **population** size. The data has about 100 rows (exactly **97**), one column for `population` and one for `profit`.

```python
import pandas as pd
import statsmodels.formula.api as smf

profits = pd.read_csv(
    'https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt',
    header=None,
    names=['population', 'profit'])

model = smf.ols(formula='profit ~ population', data=profits).fit()
model.params
```

```text
Intercept    -3.895781
population    1.193034
dtype: float64
```

The formula `profit ~ population` reads as "**profit as a function of population**": profit is the **outcome**, population is the (single) **input**. As population increases, profit increases too (**positive slope**, +1.193 profit per unit of population). That is a fine point estimate, but it is only an estimate, and we want to know **how confident** we can be in it.

## Why coefficients are uncertain

Our data is **one sample** drawn from a larger, unreachable true population of possible food truck outcomes. This year's 97 trucks are one draw; next year, last month, or a different city would give a **different sample** and therefore a **slightly different best fit line** with a **different slope**. Fitting the same model over and over across many samples would produce a **spread** of slope values.

So the coefficient we get is effectively a **mean** across that imagined spread, and like any estimated mean it comes with a **standard error** and a **confidence interval** describing how far the true value might realistically sit from our estimate.

## Standard error (`model.bse`)

The **standard error** of each coefficient comes straight off the fitted model:

```python
model.bse
```

```text
Intercept     0.719483
population    0.079744
dtype: float64
```

Read it as **how much wiggle room** there is on each coefficient:

- The **intercept** SE (about **0.72**) is fairly large.
- The **population slope** SE (about **0.08**) is small, so we have **little wiggle room** on the slope and can be quite confident in it.

A **small** standard error ⇒ a **narrow** confidence interval ⇒ a **more trustworthy** estimate.

## Confidence intervals (`model.conf_int()`)

The 95% confidence interval is approximately **coefficient ± 2 × SE**. The exact bounds:

```python
model.conf_int()
```

```text
                   0         1
Intercept  -5.324135 -2.467427
population  1.034722  1.351345
```

Column **`0`** is the **lower bound**, column **`1`** is the **upper bound**. Worked example on the intercept: SE is about 0.72, so 2 × SE is about 1.44; the intercept estimate is about -3.90, so the lower bound is roughly -3.90 - 1.44 = **-5.33** (actual -5.324) and the upper bound is roughly -3.90 + 1.44 = **-2.46** (actual -2.467). The slope has a **narrower** interval, `[1.035, 1.351]`, because its SE is smaller.

**Two technical footnotes (the lecture rounds; here is the precise version):**

- The multiplier is not exactly 2. statsmodels uses the **t-distribution** with `n - 2 = 95` degrees of freedom, whose 97.5th percentile is about **1.99** (very close to 2, and close to the normal-distribution 1.96). "About two times the standard error" is a good rule of thumb, and the lecture's appeal to the central limit theorem and a normal shape is the right intuition.
- **How to read a confidence interval (frequentist care).** The correct reading is a statement about the **procedure**, not this one interval: if we repeated the sampling-and-interval process many times, about **95% of the intervals produced would contain the true coefficient**. Avoid saying "there is a 95% chance the true value is inside this specific interval." (This matches how Module 4 framed confidence intervals.)

## P-values (`model.pvalues`)

In practice you will usually reach for **p-values** to judge a coefficient. Extract them from the fitted model:

```python
model.pvalues
```

```text
Intercept     4.607887e-07
population    1.023210e-26
dtype: float64
```

The `e` is scientific notation: `4.6e-07` is 4.6 × 10⁻⁷, and `1.0e-26` is 1.0 × 10⁻²⁶ (a decimal point followed by 25 zeros then a 1). Both are **extremely small**. Because the raw `e` numbers are easy to misread, a common shortcut is to just check the threshold:

```python
model.pvalues < 0.05
```

```text
Intercept     True
population    True
dtype: bool
```

**What a small p-value buys you.** The test is against the **null hypothesis that the true coefficient is 0**. A coefficient of 0 means a **flat relationship**: no matter what input you feed in, the predicted output does not change, so you can say nothing about the input-output link. When **p < 0.05**, you can be confident the coefficient is **not 0** and the relationship is **statistically significant**, and in particular you can trust its **sign**: a positive estimate reflects a genuinely positive association, a negative estimate a genuinely negative one, even if the exact number is uncertain. When **p > 0.05**, you cannot trust the coefficient or even its sign (it could be positive, negative, or truly zero). As the lecturer's PhD advisor put it: "Is it significant, is it a p-value less than [0.05]? If not, you have to ignore it, you can't say anything about that relationship."

> **Correction (read this before a quiz).** The lecture and the companion notebook both define a p-value loosely as *"the chance that the true coefficient is actually zero."* That is the classic misinterpretation. A p-value is **not** the probability that the null hypothesis is true. It is the probability, **computed assuming the true coefficient really is 0**, of getting a coefficient estimate **at least as far from 0** as the one we observed. Low p-value ⇒ our data would be very surprising if the coefficient were truly 0 ⇒ we reject "coefficient = 0." Likewise, **p < 0.05** does not mean "95% sure it is nonzero"; it means the result clears a **5% significance threshold**. **Exam safety:** if a quiz keys to the course's wording ("p-value = the chance the coefficient is 0"), answer it the course's way, but know the precise definition above is the correct one.

## Visualizing the confidence intervals (does the bar cross 0?)

The p-value and the confidence interval tell the **same story**, and you can see it by plotting each coefficient as a point with its CI drawn as an **error bar**, plus a **dashed reference line at 0**.

First build a tidy frame (rename the unhelpful `0`/`1` columns and attach the estimate):

```python
coef_info = model.conf_int().rename(columns={0: 'ci_lower', 1: 'ci_upper'})
coef_info['coef'] = model.params
coef_info
```

```text
            ci_lower  ci_upper      coef
Intercept  -5.324135 -2.467427 -3.895781
population  1.034722  1.351345  1.193034
```

Then plot three scatter layers on the **same axes** (lower bound with a `_` marker, the estimate with an `o` marker, the upper bound with a `_`), connect each pair of bounds with a vertical line via `ax.vlines(...)`, and add `ax.axhline(y=0, linestyle='--')` as the zero reference:

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(3, 3))
ax.margins(x=0.2)
sns.scatterplot(data=coef_info, x=coef_info.index, y='ci_lower', color='black', marker='_', ax=ax)
sns.scatterplot(data=coef_info, x=coef_info.index, y='coef',     color='black', marker='o', ax=ax)
sns.scatterplot(data=coef_info, x=coef_info.index, y='ci_upper', color='black', marker='_', ax=ax)
ax.vlines(x=coef_info.index, ymin=coef_info['ci_lower'], ymax=coef_info['ci_upper'], color='black', linewidth=1)
ax.axhline(y=0, linestyle='--', linewidth=1, color='gray')
ax.set_xlabel('input variable')
ax.set_ylabel('coefficient value')
plt.show()
```

The big dots are the coefficient estimates and the bars are the CI bounds (they look like error bars). **Neither bar crosses the zero line**, so both coefficients are **significant**. The read is mechanical:

- **CI excludes 0** (bar does not touch the dashed line) ⇒ **p < 0.05** ⇒ significant, sign is trustworthy.
- **CI contains 0** (bar passes through the dashed line) ⇒ **p > 0.05** ⇒ not significant, cannot be confident about the coefficient.

(The lecture verbally slipped here and said an interval crossing zero would give "a p-value less than 0.05"; the correct pairing is the one above, and the companion notebook states it correctly: crossing zero means **p > 0.05**.)

---

## Quiz-ready facts

- **`model.params`** = point estimates; **`model.bse`** = standard errors; **`model.conf_int()`** = 95% CI (col `0` lower, col `1` upper); **`model.pvalues`** = p-values.
- **Small SE ⇒ narrow CI ⇒ more trustworthy** coefficient. Food truck slope SE ≈ **0.080** (tight), intercept SE ≈ **0.719** (looser).
- **95% CI ≈ coefficient ± 2 × SE.** statsmodels uses the **t multiplier** (df = n - 2 = 95 ⇒ about **1.99**), close to the normal 1.96.
- Food truck results: slope **1.193**, CI **[1.035, 1.351]**, p ≈ **1.0e-26**; intercept **-3.896**, CI **[-5.324, -2.467]**, p ≈ **4.6e-7**. **Both significant.**
- **P-value tests H0: true coefficient = 0.** Precisely, it is the probability of an estimate at least this extreme **assuming the coefficient is 0** (not "the probability the coefficient is 0", and not "95% sure it is nonzero").
- **p < 0.05 ⇒ significant**: coefficient is not 0, its **sign is trustworthy**. **p > 0.05 ⇒ ignore it** (cannot trust value or sign).
- **CI excludes 0 ⇔ p < 0.05** (significant). **CI contains 0 ⇔ p > 0.05** (not significant). The error-bar plot vs a dashed line at 0 shows this.
- **A coefficient of 0 means no relationship**: the input does not change the predicted output.

---

> **See also:** "P-values and Confidence Intervals - Notebook (Cliff Notes)" (its companion lab, which runs this exact code end to end), and "Fitting Linear Regression Models with Statsmodels (Cliff Notes)" (where `smf.ols(...).fit()` and `model.params` come from).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
