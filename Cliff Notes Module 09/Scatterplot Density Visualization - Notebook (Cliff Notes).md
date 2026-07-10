# Visualizing Point Density in Scatterplots (Notebook Walkthrough)

**Module 9 Cliff Notes** | Source: notebook `Module_9_Scatterplots_With_Density_Visualization.ipynb`

---

## TL;DR

- When many points overlap in a scatterplot, plain dots hide **where the data is most concentrated**. This notebook shows two fixes.
- **Transparency (`alpha`)**: lower the opacity so overlapping points stack and darken. Dense areas look darker, sparse areas stay light.
- **2D histogram**: bin the x-y plane into rectangles and color each tile by how many points land in it. Built with **`sns.displot(..., kind="hist", cbar=True)`**.
- Dataset is the Seaborn **penguins** set again, plotting **`flipper_length_mm`** (x) vs **`bill_depth_mm`** (y).
- Rule of thumb: transparency keeps individual points visible (good for moderate overlap); the 2D histogram summarizes into bins (better for heavy overlap).

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
  - Categorical: `species`, `island`, `sex`.
  - Continuous: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- Missing values: the four measurement columns are **342 non-null**, `sex` is **333 non-null**.
- This notebook focuses on `flipper_length_mm` vs `bill_depth_mm`.

---

## Basic scatterplot (the problem)

```python
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm")
plt.show()
```

- A standard scatterplot of two continuous variables (`relplot` defaults to `kind="scatter"`).
- With this many points, overlap in the busy regions makes it hard to tell where the data is densest. That is what the next two techniques fix.

---

## Fix 1: transparency (`alpha`) to show density

```python
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            s=250, alpha=0.33, edgecolor='none')
plt.show()
```

- **`alpha=0.33`** reduces the opacity of each point. Where points overlap, the semi-transparent dots stack on top of each other and the region gets **darker**.
- **`s=250`** controls point size (here, large points).
- **`edgecolor='none'`** removes the point borders, so only the fill color stacks.
- Read it as: areas with **few points** stay **light**, areas with **many overlapping points** look **darker** = a visual read on where the data concentrates.

> **Doc pointer (from the notebook):** more point-style arguments are listed in Seaborn's properties tutorial: https://seaborn.pydata.org/tutorial/properties.html

---

## Fix 2: 2D histogram to show density

```python
sns.displot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            kind="hist", cbar=True)
plt.show()
```

- A **2D histogram** divides the x-y plane into rectangular **bins** and counts how many points fall into each bin.
- Built with the **figure-level `sns.displot`** function (the same one used for regular histograms), with **`kind="hist"`**. Passing **both `x` and `y`** is what makes it 2D.
- **`cbar=True`** adds the **color bar**, which provides the count scale.
- Useful when there are many overlapping points, or you want a direct region-by-region summary of density.

### Reading the color bar

- Each tile = a region of the x-y space.
- **Color intensity = how many observations** fall in that region.
- Darker or more intense color means **more points** in that area.

---

## Transparency vs 2D histogram

| Technique | Keeps individual points? | Best for | Trade-off |
|---|---|---|---|
| Transparency (`alpha`) | Yes, exact points stay visible | Moderate overlap | Very dense regions can still saturate to solid color |
| 2D histogram | No, points are binned | Heavier overlap | You lose the individual observations |

- **Transparency**: keeps the exact points visible, good when you still want to see individual observations.
- **2D histogram**: summarizes data into bins, makes concentration patterns easier to read when overlap is heavy.

---

## Practice exercises (with the notebook's answers)

**1. Transparent scatter of `bill_length_mm` vs `bill_depth_mm` (`alpha=0.3`).**

```python
sns.relplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", alpha=0.3)
```

**2. Increase the point size (`s`) in the transparent scatter.**

```python
sns.relplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", alpha=0.3, s=100)
```

> **Added note (interpretation, not written in the notebook):** larger points overlap sooner, so dense regions fill in and darken more readily, at the cost of blurring exactly where each point sits.

**3. 2D histogram for `bill_length_mm` vs `bill_depth_mm` with `cbar=True`.**

```python
sns.histplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", cbar=True)
```

> **Watch this:** the practice answer uses the **axis-level `sns.histplot`**, not the figure-level `sns.displot` from the lesson above. Both draw a 2D histogram when given both `x` and `y`; the difference is figure-level (`displot`) vs axis-level (`histplot`), the same relplot/scatterplot and displot/histplot pairing seen throughout the module.

**4. Which gives a clearer picture of density?**

> Victor's answer (opinion): the size-increased scatterplot is clearer for **precision**, but the 2D histogram is better suited for **clarity at a glance**, that is, rapid ingestion of the data.

---

## Quiz-ready facts

- Two ways to show density in a scatterplot: **transparency (`alpha`)** and a **2D histogram**.
- `sns.relplot` is figure-level with default **`kind="scatter"`**; the axis-level scatter function is `sns.scatterplot`.
- **`alpha`** lowers opacity so overlapping points darken; **`s`** sets point size; **`edgecolor='none'`** removes point borders.
- 2D histogram (figure-level): **`sns.displot(..., kind="hist", cbar=True)`** with both `x` and `y`.
- 2D histogram (axis-level, used in the practice): **`sns.histplot(..., cbar=True)`** with both `x` and `y`.
- **`cbar=True`** adds the count color bar; **color intensity = point count per bin** (darker = more points).
- penguins = 344 rows, 7 columns; this notebook plots `flipper_length_mm` vs `bill_depth_mm`.
- Transparency keeps individual points visible (moderate overlap); the 2D histogram bins them (heavy overlap).

> **See also:** the sibling note **"Scatterplots - Notebook (Cliff Notes)"** for the basic `relplot` scatter and adding categoricals with `hue` / `col` / `row`, and **"Faceted Histograms and Conditional KDE (Cliff Notes)"** for more `displot` histogram usage.

---

*Source: University of Pittsburgh course notebook `Module_9_Scatterplots_With_Density_Visualization.ipynb` (personal study use).*
