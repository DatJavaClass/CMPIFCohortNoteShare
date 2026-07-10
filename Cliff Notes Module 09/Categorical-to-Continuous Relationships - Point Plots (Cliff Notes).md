# Point Plots: Categorical-to-Continuous Relationships

**Module 9 Cliff Notes** | Source: lecture transcript "Categorical-to-Continuous Relationships Point Plots"

---

## TL;DR

- A **point plot** shows the **mean** of a continuous variable for each group of a categorical variable, with an **error bar** (default: 95% confidence interval) around each mean.
- One glance answers two questions:
  1. Do the group means differ?
  2. Is that difference **statistically significant**? (Non-overlapping error bars = yes.)

---

## What a point plot is for

- Use it when you want to compare **means (averages)** of a continuous variable across the categories of a categorical variable.
- Each plotted point = the mean of the continuous variable for that group.
- The error bar around each point tells you how confident you can be in that mean, so you can judge significance visually.

> **Source note:** the lecturer's opening line says point plots show the "median and spread." That is a slip or a lead-in (he is contrasting with box-style plots). A Seaborn point plot plots the **mean** by default, which is what the rest of the lecture and the quiz use. Answer **mean**.

---

## Basic code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# gap = the gapminder DataFrame
# (life expectancy, population, GDP by country / continent, across years)

sns.catplot(
    data=gap,
    x="year",          # integer year, treated as CATEGORICAL
    y="lifeExp",       # continuous variable
    kind="point",
    linestyle="none",  # do NOT connect the means with a line
)
```

- Built with **`catplot`** and **`kind="point"`**.
- **`year` is an integer but is treated as categorical** because there are only a limited number of unique years (fewer than ~10 to 20 values), so treating them as discrete categories makes sense.
- **`y`** is the continuous variable (life expectancy).
- **`linestyle="none"`** stops Seaborn from connecting the points with a line. Seaborn draws that connecting line by default, but it is usually not useful here.
- Result: one point per year = mean life expectancy for that year (trends upward over time in the gapminder data).

---

## Styling the error bars

```python
sns.catplot(
    data=gap, x="year", y="lifeExp", kind="point",
    linestyle="none",
    capsize=0.1,                # add little end caps to the error bars
    err_kws={"linewidth": 1},   # make the error bars thinner (value illustrative; transcript gave no number)
)
```

- **`capsize=0.1`** puts end caps on the error bars.
- Keyword args about the error bars (passed via `err_kws`) let you restyle them, e.g. make them skinnier.
- You do not need to memorize every styling parameter. Look them up in the **Seaborn documentation**, or use generative AI to find them, but **watch out**: AI may push you to matplotlib. It is nicer to keep everything within Seaborn.

---

## Three variables at once (add `hue`)

You can compare **one continuous + two categorical** variables:

```python
sns.catplot(
    data=gap, x="year", y="lifeExp",
    hue="continent",   # color = the SECOND categorical variable
    kind="point",
    linestyle="none", capsize=0.1,
)
```

- **`x`** = one categorical variable (year).
- **`hue`** = a second categorical variable (continent) mapped to **color**.
- Instead of one overall mean per year, means are now separated by continent. Seaborn auto-generates a legend.

---

## The error bars = confidence intervals (the statistics)

- The default error bar is a **95% confidence interval (CI)** around the mean.
- It is built from the **standard error of the mean (SEM)**.
- The upper and lower bounds of the 95% CI are about **2 x SEM** (two standard errors on each side of the mean).
- **Why 2 x SEM?**
  - The SEM is the **standard deviation of the sampling distribution of the mean**: imagine repeatedly re-sampling and taking the mean each time.
  - By the **Central Limit Theorem**, that sampling distribution is approximately **Gaussian (normal)**, and the SEM is **one standard deviation** of that Gaussian.
  - So 2 x SEM covers roughly 95% of that distribution.
- **CI width depends mostly on sample size (n)**: the more data points you average, the smaller the error bar.

> **Precision note:** the lecture rounds the 95% multiplier to **2**; the exact value is **1.96** standard errors. For quiz answers, use the lecture's framing (about 2 SE for 95%, 1 SE for 68%).

> **Mechanism note (Seaborn internals):** The lecture explains the CI as parametric mean +/- 2 x SEM, and that is the quiz answer. Under the hood, Seaborn's default `errorbar=("ci", 95)` computes a **bootstrap** percentile interval (resample about 1000 times, take the 2.5 / 97.5 percentiles), which for large, roughly symmetric samples lands very close to +/- 2 x SEM. For a literal standard-error bar, use `errorbar=("se", 2)` (about 95%) or `errorbar=("se", 1)` (about 68%). Use the lecture's SEM framing on the quiz.

---

## Interpreting significance from the plot

- **Error bars do NOT overlap** -> the means are **significantly different** (at the 95% level). You can be 95% confident the true means differ.
- **Error bars overlap** -> the means *look* different, but you are **not 95% confident** they actually differ.
- Reading a CI: "I am 95% sure the true mean falls within this little range marked by the error bar."

**Lecture examples (gapminder, life expectancy by continent):**
- **Oceania** sits at the top with error bars so small you can barely see them. It is higher than **Europe** across most points; in **1952** and **2007** the bars clearly do not overlap (significant).
- In **1967**, the **Americas** vs **Asia**: the Americas' mean looks higher, but their error bars **overlap**, so you are not 95% confident the difference is real.

---

## Adjusting the confidence interval

Change the CI level with the **`errorbar`** argument:

```python
sns.catplot(
    data=gap, x="year", y="lifeExp", kind="point",
    linestyle="none", capsize=0.1,
    errorbar=("ci", 68),   # 68% CI instead of the default 95%
)
```

- **68% CI = one standard deviation / one standard error** on each side (1 sigma).
- **95% CI = two standard deviations** on each side (2 sigma).
- A 68% CI produces **smaller** error bars, but "68% sure" is weaker than "95% sure," so **95% is the usual default**.

---

## Quiz-ready facts

- Point plots visualize **means** across categories, plus a CI per mean.
- **`kind="point"`** in **`catplot`**; **`linestyle="none"`** removes the connecting line.
- **`hue`** adds a second categorical variable as color.
- Error bars come from the **standard error of the mean**; 95% CI is about **2 SEM**, 68% CI is about **1 SEM**.
- **Q: What does a 68% confidence interval correspond to?** A: **One standard deviation / one standard error** on either side (1 sigma).
- **Non-overlapping error bars = statistically significant difference** (the quiz answer). Caveat: the reverse is a rule of thumb only. Overlapping bars usually mean "not confident of a difference," but two 95% CIs can overlap slightly and still differ significantly.
- CI width is driven mostly by **sample size**.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
