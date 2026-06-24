# Categorical-to-Continuous Relationships Boxplots

## What this covers
- A **relationship plot**: how a **continuous** variable changes across the values of a **categorical** variable (categorical-to-continuous).
- **Boxplots** show basic properties of a distribution at a glance: the **median** (middle of the data) and the **spread/variation**.
- Draw one box per category side by side -> quickly compare whether the median and spread shift across categories, then dig into the interesting ones.
- Big advantage: fast read of center + spread. Trade-off: it hides finer detail (violin plots show more), and it misleads on skewed data.

## Setup
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset('penguins')
penguins.info()
```
- **Penguins** dataset again. Each row = one penguin.
- Categorical: `species`, `island`, `sex`. Continuous: `bill_length_mm`, `flipper_length_mm`, `body_mass_g`, etc.

## Figure-level: catplot with kind='box'
```python
sns.catplot(data=penguins, x='species', y='flipper_length_mm', kind='box', height=3)
plt.show()
```
- `catplot` is the figure-level categorical function; `kind='box'` -> boxplots.
- Put the **categorical** variable on `x=`, the **continuous** variable on `y=`.
- Vertical is the usual default; you *can* go horizontal (swap x/y), but vertical is standard these days.
- `height` is optional (sizing only). `plt.show()` suppresses the returned-object text.
- Reads: across species you can see the median flipper length (the middle line) and spread shift right away.

## How to read a boxplot
- **Median**: the line inside the box. 50% of data points are above it, 50% below. (Not necessarily the mean.)
- **The box = middle 50% of the data.** Sort the data into four equal-count **quartiles**; the box spans **Q1 to Q3**. That width is the **interquartile range (IQR)**.
- **Whiskers**: lines out to the most extreme value still within **1.5 × IQR** of the box (so they reach the true min/max only when there are no outliers).
- **Outliers**: any point beyond 1.5 × IQR shows up as an individual dot. Treat these as outliers (far from the median). Example: the Adelie species shows a couple of points outside the whiskers.
- Any distribution works here -> the data does **not** have to be Gaussian (the textbook diagram just maps the box onto a normal curve as a visual aid).

### Show the mean too
```python
sns.catplot(data=penguins, x='species', y='flipper_length_mm', kind='box',
            height=3, showmeans=True)
plt.show()
```
- `showmeans=True` plots the **mean** (a small marker, e.g. a green triangle) alongside the median.
- The mean usually differs slightly from the median. Example: for Adelie/Chinstrap it sits near the median; for Gentoo it reads a bit above -> a hint of asymmetry.

## When boxplots fail: skewed distributions
- Boxplots are most effective for **roughly symmetric** distributions (most values near the middle of the range).
- **Skewed** distributions (lots of values bunched at one end) produce ugly, hard-to-read boxes drowning in outliers.
```python
diamonds = sns.load_dataset('diamonds')          # ~50,000 rows
# See the skew first with faceted histograms:
sns.displot(data=diamonds, x='price', col='color', kind='hist', col_wrap=3)
plt.show()
# Boxplots of the same skewed data look nasty:
sns.catplot(data=diamonds, x='color', y='price', kind='box', height=4)
plt.show()
```
- `price` is heavily right-skewed (most diamonds cheap, a long thin tail of expensive ones). The boxplots bury that shape under a wall of outliers and don't convey how skewed it is.
- Side note: `diamonds` has a true pandas **`category`** dtype (an alternative to `object` for categoricals, with some handy built-in functions). `object` is still fine and common.
- Fix: for skewed continuous distributions across categories, reach for **faceted histograms** or **conditional KDE plots** instead.

## Quick reference
- Boxplot (categorical x continuous): `sns.catplot(data=, x=<categorical>, y=<continuous>, kind='box', height=)`.
- Box = Q1->Q3 (middle 50%); line = **median**; **IQR** = box width; whiskers reach 1.5 × IQR; points beyond = **outliers**.
- Median != mean in general; add `showmeans=True` to plot the mean.
- Works for any distribution, but only *useful* for roughly symmetric ones.
- Skewed data -> use faceted histograms (`displot(..., col=, col_wrap=)`) or KDE plots instead.

---

*Authored and directed by **Victor Sverdlin (DatJavaClass)**: conceived, structured, formatted, fact-checked, and edited by Victor Sverdlin, with assistance by Claude.*
