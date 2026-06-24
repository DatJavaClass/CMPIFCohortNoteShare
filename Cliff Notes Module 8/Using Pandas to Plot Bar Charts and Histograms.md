# Using Pandas to Plot Bar Charts and Histograms

## What this covers
- Pandas has **built-in plotting** that calls **Matplotlib** under the hood. Good for **quick, simple** plots without importing/setting up Seaborn or Matplotlib plotting calls directly.
- Two recipes here: **bar charts** for categorical variables, **histograms** for continuous variables.

## Setup
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Bar plots
Call `.plot.bar()` on a **Series** (a single column, or the output of `.value_counts()`). Equivalent: `.plot(kind='bar')`.
```python
df["team"].value_counts().plot.bar()
plt.show()
```
- `.value_counts()` -> counts per category; `.plot.bar()` -> one bar per category.
- `plt.show()` suppresses the returned-object text.

### Customizing bar plots
Pass arguments straight into `.plot.bar()`:
```python
df["team"].value_counts().plot.bar(
    title="Team Frequency",
    xlabel="Team", ylabel="Count",
    rot=45,              # rotate x tick labels
    figsize=(3.5, 3.5)  # width, height in inches
)
plt.show()
```
- See the [pandas .plot docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html) for the full argument list.

### More control: plot onto your own fig/ax
For finer control (e.g. plotting multiple datasets on one plot), make Matplotlib `Figure`/`Axes` objects and pass `ax=`. Then tweak the `ax` directly.
```python
fig, ax = plt.subplots(figsize=(3.5, 3.5))
df["team"].value_counts().plot.bar(ax=ax)   # plot onto the ax
ax.set_title("Team Frequency")
ax.set_xlabel("Team")
ax.set_ylabel("Count")
ax.tick_params(axis="x", labelrotation=45)
plt.show()
```
- `ax.set_title / set_xlabel / set_ylabel` replace the inline `title=/xlabel=/ylabel=` args.
- `ax.tick_params(axis="x", labelrotation=45)` is the fig/ax way to do `rot=45`.

## Histograms
For the distribution of a **continuous** variable, call `.plot.hist()` (or `.hist()`) on a Series.
```python
gapminder = pd.read_csv(
    'https://raw.githubusercontent.com/chendaniely/pandas_for_everyone/master/data/gapminder.tsv',
    sep='\t'
)
gapminder.lifeExp.plot.hist()
plt.show()
```
- Gapminder columns: categorical `country`, `continent`; continuous `lifeExp`, `pop`, `gdpPercap` (plus `year`).

### Histograms of every continuous column at once
Call `.hist()` on the **whole DataFrame** -> a grid of histograms for all numeric columns.
```python
gapminder.hist()
plt.show()
```

## Quick reference
- Bar (categorical): `series.plot.bar()` or `series.plot(kind='bar')`; pair with `.value_counts()`.
- Bar args: `title=`, `xlabel=`, `ylabel=`, `rot=`, `figsize=(w, h)`.
- Fig/ax control: `fig, ax = plt.subplots(figsize=...)` then `.plot.bar(ax=ax)` + `ax.set_*` / `ax.tick_params(...)`.
- Histogram (continuous): `series.plot.hist()` or `series.hist()`.
- All numeric columns: `df.hist()`.
- Pandas plotting = quick & simple; reach for Matplotlib/Seaborn when you need more.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
