# Pair Plots (Notebook): Visualizing Multiple Continuous Variables

**Module 10 Cliff Notes** | Source: notebook `Module_10_Pairplots.ipynb`

---

## TL;DR

- A **pair plot** (**`sns.pairplot`**) draws a grid that visualizes relationships among **multiple continuous variables** in one figure.
- **Off-diagonal** cells = **scatter plots** for each pair of variables; **diagonal** cells = each variable's **marginal distribution**.
- Add **`hue`** to color by a category (this also switches the diagonal to **KDE** curves). Use **`diag_kws={"common_norm": False}`** to compare distribution **shapes** without sample-size distorting the diagonal KDE heights.
- Pair plots get crowded fast: best with about **3 to 6** variables, **~20 at most**. Restrict them with the **`vars`** argument.

---

## Setup and dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
penguins.info()
penguins.head()
```

- **penguins**: 344 rows, 7 columns.
  - Categorical (`object`): `species`, `island`, `sex`.
  - Continuous (`float64`): `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- Missing values are present: the four measurement columns are 342 non-null, `sex` is 333 non-null. (Missing points are simply not drawn in each cell.)

---

## Basic pair plot

```python
sns.pairplot(data=penguins)
plt.show()
```

- **Off-diagonal** cells: a **scatter plot** for every pair of the numeric columns (so you see all pairwise relationships at once).
- **Diagonal** cells: each variable's **marginal distribution**, drawn as a **histogram** by default (no `hue`).
- One glance surfaces patterns worth a closer look: linear, non-linear, or clustering. From there you can build focused scatter plots on the pairs of interest.
- Only the **numeric** columns are plotted; the categorical columns (`species`, `island`, `sex`) are skipped unless used for grouping.

---

## Conditioning on a category with `hue`

```python
sns.pairplot(data=penguins, hue="species")
plt.show()
```

- **`hue="species"`** colors every point by species, exposing group structure in the scatter cells.
- Side effect: with `hue` set, the **diagonal switches from histograms to KDE** curves, one curve per group (a conditional KDE).

---

## Removing the sample-size effect on diagonal KDEs

```python
sns.pairplot(data=penguins, hue="species", diag_kws={"common_norm": False})
plt.show()
```

- By default, the diagonal KDE curves keep the effect of **sample size**: groups with fewer rows draw **smaller-looking** density curves, because all groups share one normalization.
- **`diag_kws={"common_norm": False}`** normalizes **each group's diagonal distribution separately**, so:
  - smaller groups no longer look artificially low, and
  - it becomes easier to compare the **shape** of the distributions across groups (rather than their sizes).
- **`diag_kws`** is the general mechanism: it passes keyword arguments through to the diagonal plotting function (here, the KDE). `common_norm` is a KDE argument.

> Use `common_norm=False` when you care about **distribution shape** per group; keep the default when you want the relative **group sizes** reflected.

---

## Don't use too many variables

- Pair plots become **very crowded** if you include too many variables (the grid is variables x variables).
- Practical guidance from the notebook:
  - Best with a **small to moderate** number of numeric variables.
  - **~20 variables at most**, but usually far fewer.
  - For most exploratory work, **3 to 6** variables is the sweet spot.
- Restrict to the variables you care about with the **`vars`** argument:

```python
sns.pairplot(
    data=penguins,
    vars=["bill_length_mm", "bill_depth_mm", "body_mass_g"],
    hue="species",
    diag_kws={"common_norm": False},
)
plt.show()
```

- **`vars=[...]`** limits the grid to just those columns (here a tidy 3x3), while still coloring by `species` and per-group-normalizing the diagonal.

---

## Quiz-ready facts

- **`sns.pairplot()`** visualizes relationships among **multiple continuous variables** in one grid.
- **Off-diagonal** = pairwise **scatter plots**; **diagonal** = each variable's **marginal distribution**.
- Diagonal is a **histogram** by default; adding **`hue`** switches the diagonal to **KDE** and colors the points by group.
- **`diag_kws={"common_norm": False}`** normalizes each group's diagonal KDE separately, so sample size does not shrink smaller groups' curves; use it to compare distribution **shape**.
- Only **numeric** columns are plotted; categoricals are used via **`hue`**, not plotted as axes.
- Pair plots are best with **3 to 6** variables, **~20 at most**; restrict them with **`vars=[...]`**.
- penguins dataset: **344 rows, 7 columns** (4 continuous, 3 categorical), with some missing values.

> **See also:** the Module 9 note "Continuous-to-Continuous Relationships - Scatterplots (Cliff Notes)" for reading a single scatter plot and the within-group trend reversal (Simpson's paradox), and the Module 10 note "Continuous-to-Continuous Relationships - Correlation Plots (Cliff Notes)" for quantifying these relationships with correlation coefficients and heat maps.

---

*Source: CMPIF2100 notebook (personal study use).*
