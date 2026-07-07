# Point Plots (Notebook Walkthrough)

**Module 9 Cliff Notes** | Source: notebook `Module_9_point_plots_start.ipynb` (Gapminder demo)

---

## TL;DR

- A **point plot** shows the **mean** of a continuous variable for each value of a categorical variable, with an **error bar** (default: 95% confidence interval) around each mean.
- Built in Seaborn with **`sns.catplot(..., kind="point")`**; use **`linestyle="none"`** so the means are not connected by a line.
- This notebook plots **`lifeExp`** (life expectancy) by **`year`**, then splits by **`continent`** using **`hue`**, all on the **Gapminder** dataset.
- The error bars come from the **standard error of the mean**; the 95% CI bounds are about **2 x SEM** above and below the mean.

---

## Setup and dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Gapminder (tab-separated file loaded straight from a URL)
gap_url = 'https://raw.githubusercontent.com/chendaniely/pandas_for_everyone/master/data/gapminder.tsv'
gap = pd.read_csv(gap_url, sep='\t')
gap.info()
```

- **`sep='\t'`** because the file is a `.tsv` (tab-separated), not a comma CSV.
- `gap.info()` reports **1704 rows, 6 columns**:
  - `country` (object), `continent` (object), `year` (int64), `lifeExp` (float64), `pop` (int64), `gdpPercap` (float64).
- The continuous variable plotted here is `lifeExp` (the `y` value); `year` and `continent` act as the categorical groupers.

---

## What a point plot is for

- Answers: **"Do the averages of a continuous variable change across categories?"**
- Each plotted point = the **mean** of the continuous variable for that group.
- The error bar around each point shows how confident you can be in that mean.

---

## Basic point plot

```python
sns.catplot(data=gap, x='year', y='lifeExp', kind='point', linestyle='none')
```

- Built with **`catplot`** + **`kind="point"`** (a figure-level function).
- **`year` is an integer but is treated as categorical**, because `catplot` treats the x variable as categorical by default, so each distinct year becomes a discrete point on the x-axis.
- **`y='lifeExp'`** is the continuous variable being averaged.
- **`linestyle='none'`** stops Seaborn from drawing the connecting line it adds by default.
- Result: one point per year = mean life expectancy for that year.

> **Doc note (from the notebook):** `sns.catplot(kind='point')` runs the axes-level function **`sns.pointplot`** under the hood. Styling options live in the `sns.pointplot` documentation.

---

## Styling the error bars and figure

```python
sns.catplot(data=gap, x='year', y='lifeExp', kind='point', linestyle='none',
            capsize=0.1, err_kws={'linewidth': 1}, height=3, aspect=2)
```

- **`capsize=0.1`** adds end caps to the error bars.
- **`err_kws={'linewidth': 1}`** restyles the error bars (here, makes the lines thinner).
- **`height=3`** and **`aspect=2`** set the figure size (aspect = width / height, so a wide short panel).

---

## Three variables at once (add `hue`)

Compare **one continuous + two categorical** variables: one categorical on the x-axis, the other set by color.

```python
sns.catplot(data=gap, x='year', y='lifeExp', hue='continent', kind='point', linestyle='none',
            capsize=0.1, err_kws={'linewidth': 1}, height=3, aspect=2)
plt.show()
```

- **`x='year'`** = first categorical variable.
- **`hue='continent'`** = second categorical variable, mapped to **color**.
- Now you get a separate mean-per-year line of points **for each continent**, and Seaborn auto-builds the legend.

---

## The error bars = confidence intervals (the statistics)

From the notebook's "Error bars" markdown:

- The default error bar is a **95% confidence interval**.
- It is **calculated from the standard error of the mean (SEM)**.
- Specifically, the upper and lower bounds are **2 times the SEM** from the estimated mean.
- The **range of the confidence interval can be adjusted** (see below).

> **Precision note:** the notebook rounds the 95% multiplier to **2 x SEM**; the exact value is about **1.96 x SEM**. Use the notebook's "2 x SEM" framing for quiz answers.

---

## Adjusting the confidence interval

Change the CI level with the **`errorbar`** argument:

```python
sns.catplot(data=gap, x='year', y='lifeExp', hue='continent', kind='point', linestyle='none',
            errorbar=('ci', 68), capsize=0.1, err_kws={'linewidth': 1}, height=3, aspect=2)
plt.show()
```

- **`errorbar=('ci', 68)`** requests a **68% CI** instead of the default 95%.
- A 68% CI produces **smaller** error bars: "68% sure" is weaker than "95% sure," so 95% stays the usual default.

> **Added note (inference, not stated in the notebook):** a 68% CI is roughly **1 x SEM** on each side, versus about **2 x SEM** for the 95% default.

---

## Interpreting significance from the plot (cross-reference)

> **Added note:** the notebook shows the code and defines the error bars, but does not spell out the overlap rule. Quick version below; the sibling note **"Categorical-to-Continuous Relationships - Point Plots (Cliff Notes)"** covers it in depth with worked examples.

- **Error bars do NOT overlap** -> the means are **significantly different** at that confidence level.
- **Error bars overlap** -> the means look different, but you are **not confident** they truly differ.
- CI width is driven mostly by **sample size (n)**: more data averaged = smaller error bar.

---

## Quiz-ready facts

- Point plots visualize **means** of a continuous variable across categories, plus a CI per mean.
- **`kind="point"`** inside **`sns.catplot`**; the axes-level function is **`sns.pointplot`**.
- **`linestyle="none"`** removes the default connecting line.
- **`hue`** adds a second categorical variable as color (here, `continent`).
- **`capsize`** adds error-bar end caps; **`err_kws`** restyles the error bars; **`height`** / **`aspect`** size the figure.
- Default error bar = **95% CI**, calculated from the **standard error of the mean**; bounds are about **2 x SEM** from the mean.
- **`errorbar=('ci', 68)`** switches to a 68% CI (smaller bars, roughly 1 SEM each side).
- Dataset: **Gapminder**, 1704 rows, columns `country, continent, year, lifeExp, pop, gdpPercap`, loaded with `sep='\t'`.
- **Non-overlapping error bars = statistically significant difference** (standard interpretation, not stated in the notebook).

---

*Source: University of Pittsburgh course notebook `Module_9_point_plots_start.ipynb` (personal study use).*
