# Faceted Histograms and Conditional KDE: Categorical-to-Continuous Relationships

**Module 9 Cliff Notes** | Source: Jupyter notebook "Module_9_Faceted_Histograms_Conditional_KDE"

---

## TL;DR

- Both tools answer one question: **does the distribution of a continuous variable change across the categories of a categorical variable?**
- **Faceted histogram:** one separate histogram per category, drawn side by side (no overlap).
- **Conditional KDE:** smooth density curves, one per category, overlaid on the same axes so you can compare shapes.
- Everything is built with one figure-level function: **`sns.displot(...)`**, switching behavior via `col` / `row` / `hue` and `kind="hist"` vs `kind="kde"`.
- The one gotcha to memorize: **`common_norm`** controls whether group sample size affects KDE curve height.

---

## What these plots are for

- **Goal-first:** before picking a plot, ask what you want to learn. Here the goal is comparing a continuous distribution across categories.
- Notebook example: distribution of **`flipper_length_mm`** (continuous) across **`species`** (categorical) in Seaborn's **`penguins`** dataset.
- **Faceted histogram:** use when you want each category's distribution on its own axis, side by side, without stacking everything together.
- **Conditional KDE:** use when you want smooth curves overlaid so differences in shape and location pop out at a glance.

---

## Setup and data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
penguins.info()   # 344 rows, 7 columns
```

- **`penguins`** ships with Seaborn: 344 rows, 7 columns (`species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`).
- `flipper_length_mm` has 342 non-null values (2 missing).
- **Drop missing values in the columns you will plot** before charting:

```python
penguins_clean = penguins.dropna(subset=["flipper_length_mm", "species"])
penguins_clean.shape   # (342, 7)
```

- 344 rows in, 342 rows out, so **2 rows were dropped**.
- `dropna(subset=[...])` only drops rows missing a value in the listed columns, not every column.

---

## Faceted histograms (one plot per category)

**Facets = subplots within one figure.** Instead of one histogram for the whole `flipper_length_mm` distribution, you get one histogram per `species` value.

Facet across **columns** with the `col` argument:

```python
sns.displot(data=penguins_clean, x="flipper_length_mm", col="species", kind="hist")
plt.show()
```

Facet across **rows** with the `row` argument (and shrink each panel with `height`):

```python
sns.displot(data=penguins_clean, x="flipper_length_mm", row="species", kind="hist", height=2.5)
plt.show()
```

- **`col="species"`** puts each species in its own column; **`row="species"`** stacks them vertically.
- **`kind="hist"`** makes each facet a histogram (y-axis = count).
- **`height=2.5`** sets the height (in inches) of each facet, handy so stacked rows do not get too tall.
- One facet is drawn per unique category value. The penguins set has three species (Adelie, Chinstrap, Gentoo), so you get three panels.

---

## Conditional KDE plots (overlaid smooth curves)

- A **KDE (kernel density estimate)** is a smooth estimate of where values are concentrated. It does the same job as a histogram but smooths out the individual bars into a continuous curve.
- **Conditional** = one curve per category, so you can compare distributions across the categorical variable.
- Overlay the curves by mapping the category to **`hue`** and setting **`kind="kde"`**:

```python
sns.displot(data=penguins_clean, x="flipper_length_mm", hue="species", kind="kde")
plt.show()
```

- **`hue="species"`** draws a separate colored curve for each species and auto-generates a legend.
- Note the y-axis now reads **"Density"**, not count. Density reflects where rows concentrate along the x-axis, not a raw count of rows.

---

## Key argument: `common_norm` (the sample-size effect)

By default Seaborn factors in how many rows each group has. This is the **sample-size effect**.

- **`common_norm=True`** (the default for conditional KDE):
  - All curves are scaled together so the **total area across all groups combined equals 1**.
  - Groups with fewer rows get **smaller (shorter) curves**.
  - In the notebook, the Chinstrap peak sits **lower** than Adelie, because there are fewer Chinstrap penguins.
  - Use when the question involves overall frequency, e.g. "Which species is more common, and where are its values concentrated?"

- **`common_norm=False`** (remove the sample-size effect):

```python
sns.displot(data=penguins_clean, x="flipper_length_mm", hue="species", kind="kde", common_norm=False)
plt.show()
```

  - **Each group's curve is normalized independently so every curve has area 1.**
  - Curve height no longer reflects group size, so shapes are easier to compare directly.
  - In the notebook, the Chinstrap peak becomes **much higher** once the sample-size effect is removed.
  - Use when the question is "Within each species, where are values concentrated?" regardless of how common the species is.

> **Source note:** the specific peak comparisons above (Chinstrap sitting lower than, then higher than, Adelie) come from the notebook's written commentary, not from re-reading the rendered plot images. Treat them as the notebook's stated observations.

---

## How to read them

- **Faceted histogram:** scan panel to panel. Because each category is on its own axis, you compare where each distribution centers and how spread out it is without curves overlapping.
- **Conditional KDE (default, `common_norm=True`):** taller / larger curves mean more of the overall data sits in that group. Height encodes both concentration and group frequency.
- **Conditional KDE (`common_norm=False`):** ignore relative heights as a frequency signal; every curve integrates to 1, so focus on **shape and location** (where each species' flipper lengths cluster).
- KDE y-axis is **density**, not counts. Do not read a KDE peak as a raw row count.

---

## Quiz-ready facts

- **Goal both tools serve:** does a continuous variable's distribution change across categories?
- **`sns.displot(...)`** is the figure-level function for both; switch with **`kind="hist"`** or **`kind="kde"`**.
- **`col=`** = facets across columns; **`row=`** = facets across rows; each unique category value gets its own facet.
- **`hue=`** overlays one curve/color per category on shared axes (auto legend).
- **`height=`** sets each facet's height in inches.
- **Faceted histogram** = separate, side-by-side, non-overlapping distributions. **Conditional KDE** = smooth overlaid curves.
- KDE y-axis = **Density**, not count.
- **`common_norm=True`** (default): curves scaled so total area across all groups = 1; smaller groups get smaller curves (sample-size effect kept).
- **`common_norm=False`**: each group's curve has area 1; sample-size effect removed; compare shapes directly.
- Clean your data first: **`dropna(subset=[...])`** drops rows missing values in only the listed columns (dropped 2 rows here, 344 to 342).

---

*Source: University of Pittsburgh course Jupyter notebook (personal study use).*
