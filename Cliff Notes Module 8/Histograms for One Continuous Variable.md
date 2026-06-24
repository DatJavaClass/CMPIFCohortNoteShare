# Histograms for One Continuous Variable

## What this covers
- The **continuous** counterpart to bar charts. Can't count every unique value — numeric/continuous variables have too many (often every row is unique to a decimal).
- Instead use a **histogram**: break the value range into **intervals (bins)**, count how many values fall in each. Quickly shows the **distribution** — where values concentrate, where they're sparse.
- Still a marginal plot (one variable at a time).

## Setup
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
- **Gapminder** dataset again. Categorical: `country`, `continent`. Continuous: `lifeExp`, `pop`, `gdpPercap`.
- Reminder: `year` has only ~12 unique values, so it *could* be treated as categorical.

## Figure-level: displot
```python
sns.displot(data=gap, x='lifeExp', kind='hist', height=4)
plt.show()
```
- `displot` = **dist**ribution plot (figure-level).
- `kind='hist'` -> histogram.
- `plt.show()` suppresses the returned-object text.
- Reads: life expectancy concentrates ~70–75, with a secondary cluster ~40–50.

### Bins
```python
sns.displot(data=gap, x='lifeExp', kind='hist', bins=10)
```
- **bins** = number of intervals. Seaborn's default is decent.
- **Use a smallish number of bins** — you want the general shape at a glance.
  - Larger datasets can support more bins.
  - **Too many** (e.g., 200) -> tiny choppy spikes, dramatic ups/downs, no added info. Same shape but noisier. Aim for smoother.

## Axes-level: histplot
- `displot(kind='hist')` calls `histplot` underneath.
```python
fig, ax = plt.subplots(figsize=(4, 3))   # width, height in inches
sns.histplot(data=gap, x='gdpPercap', ax=ax)
```
- Lets you control `figsize` via your own fig/ax.
- `gdpPercap` example: heavy concentration at low values, counts drop off sharply from there (right-skewed).

## KDE (Kernel Density Estimate)
- Alternative view of concentration — a **smooth** estimate instead of bars (math behind it not covered).
```python
sns.displot(data=gap, x='lifeExp', kind='kde')
```
- No bars, not choppy.
- **Y-axis is density, not counts** — higher density = more rows concentrated there. Peak ~70 for life expectancy, low at the extremes.

### Combine histogram + KDE
```python
sns.displot(data=gap, x='lifeExp', kind='hist', kde=True)
```
- `kde=True` overlays the smooth trend line on the bars. Best of both — specific intervals plus the smoothed trend.

## Looping over all continuous variables
Grab numeric column names with `select_dtypes`, then iterate:
```python
num_cols = gap.select_dtypes('number').columns

for col in num_cols:
    sns.displot(data=gap, x=col, kind='hist', kde=True, height=3)
```
- `select_dtypes('number')` selects columns by data type (numeric here); `.columns` gets their names.
- For real assignments, consider dropping `year` first (low unique count -> treat as categorical).
- Watch for **skew**: heavily skewed variables (e.g., `pop`, `gdpPercap` — high then dropping over time) produce funny-looking graphs. Fewer bins can help readability.

## Quick reference
- Figure-level continuous: `sns.displot(data=, x=, kind='hist', bins=, height=, aspect=)`
- Axes-level continuous: `sns.histplot(data=, x=, ax=)`
- Smooth density: `kind='kde'` (y-axis = density, not counts)
- Overlay: `kind='hist', kde=True`
- Fewer bins -> coarse/general; too many -> noisy. Favor fewer for distribution shape.
- `df.select_dtypes('number').columns` -> iterate over all numeric variables.

---

*Authored and directed by **Victor Sverdlin (DatJavaClass)**: conceived, structured, formatted, fact-checked, and edited by Victor Sverdlin, with assistance by Claude.*
