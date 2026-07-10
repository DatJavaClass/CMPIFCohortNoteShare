# Styles and Color Palettes in Seaborn (Notebook)

**Module 10 Cliff Notes** | Source: notebook `Module_10_Styles_Color_Palettes_in_Seaborn.ipynb`

---

## TL;DR

- Two levers control a Seaborn figure's look: the **color palette** (`palette=`) and the **background style** (`sns.set_style()`).
- **`hue`** can take a **continuous** variable, not just a categorical one; Seaborn then shades points along a gradient (like histogram bins).
- Palettes come in three flavors: **qualitative** (categories), **sequential** (continuous), **diverging** (continuous around a center). Match the palette type to the variable type.
- Add **`_r`** to any palette name to **reverse** it (e.g. `mako_r`). Change the whole figure background with **`sns.set_style("dark" | "whitegrid" | "white")`**.

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

- **penguins**: 344 rows, 7 columns (4 continuous measurements, 3 categorical: `species`, `island`, `sex`); some missing values.

---

## Coloring by a continuous variable

Earlier notes colored points by a **categorical** `hue`. `hue` also accepts a **continuous** variable, and Seaborn shades the points along intervals of that variable (conceptually like the bins of a histogram), producing a color gradient with a colorbar.

```python
# Basic scatter plot
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm")
plt.show()

# Color by the continuous variable body_mass_g
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm", hue="body_mass_g")
plt.show()
```

---

## Color palettes

A **color palette** is a pre-chosen set of colors used together to show variation. Pass one by name with the **`palette=`** argument. There are three main types, and the point is to **match the palette type to the variable type**:

| Palette type | Shows | Best for |
|---|---|---|
| **qualitative** | separate colors for separate categories | categorical variables |
| **sequential** | shades within one or two colors | continuous variables |
| **diverging** | shades within two colors from a center | continuous variables that diverge from a midpoint (like 0) |

- Some palettes are built to be **accessible** for color vision deficiencies, e.g. **`colorblind`** and **`viridis`**.

```python
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            hue="body_mass_g", palette="viridis")
plt.show()
```

### Matching palette to variable (the key gotcha)

- Using a **qualitative** palette like `colorblind` on a **continuous** `hue` (`body_mass_g`) looks wrong, because qualitative palettes are built for **categorical** data.
- For a continuous variable, reach for a **sequential** palette: **`mako`**, **`rocket`**, or **`viridis`**. These give a smooth gradient, so it is easy to see values change across the plot.

---

## Reversing a palette (`_r`)

Add **`_r`** to any palette name to reverse the direction of its colors.

```python
# Reversed
sns.scatterplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
                hue="body_mass_g", palette="mako_r")
plt.show()

# Regular, for comparison
sns.scatterplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
                hue="body_mass_g", palette="mako")
plt.show()
```

- `mako` and `mako_r` use the same colors in opposite order, which flips which end of `body_mass_g` gets the light vs dark shade.

---

## Figure background style (`sns.set_style`)

Change the whole figure's background theme with **`sns.set_style(...)`**. It is a global setting: once called, it applies to plots drawn afterward.

```python
sns.set_style("dark")        # solid (non-white) background, no grid
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm", hue="body_mass_g")
plt.show()

sns.set_style("whitegrid")   # white background WITH grid lines
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm", hue="body_mass_g")
plt.show()

sns.set_style("white")       # white background, no grid lines (the default look)
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm", hue="body_mass_g")
plt.show()
```

- **`"dark"`**: shaded background, no grid.
- **`"whitegrid"`**: white background with **grid lines**, handy for reading off exactly where points sit.
- **`"white"`**: white background, **no** grid lines. Per this notebook, this is the **default** style.

---

## Quiz-ready facts

- Set the palette with **`palette="name"`**; set the background with **`sns.set_style("name")`**.
- **`hue`** accepts a **continuous** variable (gradient shading), not only a categorical one.
- Palette types: **qualitative** (categorical), **sequential** (continuous), **diverging** (continuous around a center). Match type to variable.
- **`colorblind`** and **`viridis`** are accessibility-friendly; `colorblind` is **qualitative** (for categories), so it is a poor fit for a continuous `hue`.
- Sequential palettes for continuous data: **`mako`**, **`rocket`**, **`viridis`**.
- Append **`_r`** to reverse any palette (e.g. `mako_r`).
- Background styles: **`dark`**, **`whitegrid`**, **`white`**; the notebook calls **`white`** (white background, no grid) the default.
- `sns.set_style()` is global: it affects plots created after the call.

> **See also:** the Module 10 note "Correlation Plots - Notebook (Cliff Notes)" for the `cmap` argument (the heat map equivalent of `palette`) and diverging palettes in action, and "Pairplots - Notebook (Cliff Notes)" for another multi-variable Seaborn plot.

---

*Source: CMPIF2100 notebook (personal study use).*
