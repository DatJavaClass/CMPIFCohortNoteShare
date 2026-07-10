# Bar Charts for One Categorical Variable

## What this covers
- Plotting the **count of each unique value** for a categorical variable.
- These are **marginal plots / marginal distributions** (one variable, ignore the rest).
- Categorical reminder: non-numeric, usually string dtype. Either/or values — dog/cat, yes/no, male/female.
- Goal: a **bar chart** that shows row counts per value at a glance.

## Seaborn basics
```python
import matplotlib.pyplot as plt
import seaborn as sns               # customary alias
```
- Seaborn is built on top of matplotlib but more user-friendly, with strong pandas DataFrame integration.
- Two function levels:
  - **Figure-level functions**: goal-oriented, good for quick observations, default choice. Big advantage: easy **multiple subplots (facets)** within one matplotlib figure.
  - **Axes-level functions**: lower-level, create one specific plot, more control for tweaking.

## Dataset: penguins
```python
penguins = sns.load_dataset('penguins')
```
- `species`, `island`, `sex` are `object` -> categorical (strings).
- `bill_length`, `bill_depth`, etc. are numeric. Each row = one penguin.

## Figure-level: catplot
```python
sns.catplot(data=penguins, x='sex', kind='count', height=3)
```
- `catplot` is the figure-level function for categorical variables.
- Pass the DataFrame directly via `data=`.
- `x=` sets the variable to plot.
- `kind='count'` -> **count plot**: counts rows per value automatically (no manual `value_counts()` like in pandas).
- Example reads: ~168–170 each for male/female; species skews heavily Adelie, with fewer Gentoo and Chinstrap.

### Sizing: height and aspect
- `height` = height of each facet **in inches** (rough — DPI affects actual size; if too big, shrink, if too small, grow).
- `aspect` sets **width = aspect × height**. Increase to make bars wider.
```python
sns.catplot(data=penguins, x='island', kind='count', height=3, aspect=2)
```

### Suppressing the returned-object text
- catplot prints a description of the returned object. To get just the plot:
```python
sns.catplot(data=penguins, x='sex', kind='count', height=2)
plt.show()
```

## Axes-level: countplot
- catplot with `kind='count'` calls `countplot` behind the scenes.
- Recall the matplotlib hierarchy: **figure** = frame, **axes** = subplots within it. Axes-level functions plot into a single axes.
```python
sns.countplot(data=penguins, x='sex')
```
- Harder to size directly. To control it, make your own matplotlib figure/axes and pass `figsize`:
```python
fig, ax = plt.subplots(figsize=(2, 3))   # 2 in wide, 3 in tall
sns.countplot(data=penguins, x='sex', ax=ax)
```

## Horizontal bars
- Swap `x` for `y` to flip the orientation — put the categorical variable on the y-axis.
```python
fig, ax = plt.subplots(figsize=(3, 2))   # wider than tall reads better here
sns.countplot(data=penguins, y='sex', ax=ax)
```
- Horizontal bars are easier to read in some cases (e.g., long category labels).

## Quick reference
- Figure-level categorical: `sns.catplot(data=, x=, kind='count', height=, aspect=)`
- Axes-level categorical: `sns.countplot(data=, x=, ax=)`
- Count plot = automatic row counts per category (Seaborn's built-in `value_counts` equivalent).
- `x=` vertical bars, `y=` horizontal bars.
- `height`/`aspect` for figure-level sizing; `figsize` (via your own fig/ax) for axes-level sizing.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
