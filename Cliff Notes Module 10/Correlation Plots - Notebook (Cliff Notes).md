# Correlation Plots (Lab Notebook): Quantifying and Visualizing Correlation

**Module 10 Cliff Notes** | Source: lab notebook `Module_10_correlation_plots_start_Lab.ipynb`

---

## TL;DR

- A **correlation coefficient** quantifies the linear relationship between two continuous variables. It runs from **-1 to +1** (near +1 positive, near -1 negative, near 0 none).
- Get the whole **correlation matrix** in pandas with **`dataframe.corr()`**.
- **Eyeball** correlation with a scatter plot plus regression line (**`sns.lmplot`**); **visualize** the matrix as a heat map (**`sns.heatmap`**) with a **diverging** palette centered on 0.
- This is the **lab starter** version: one step (computing the correlation matrix that feeds the heat map) is left for you to fill in. See the note below.

> **Companion note:** the lecture-based "Continuous-to-Continuous Relationships - Correlation Plots (Cliff Notes)" covers the same concepts in more depth. This note focuses on what the lab notebook itself runs.

---

## Setup and dataset

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Wine dataset, read straight from a URL; the raw file has no header,
# so column names are supplied explicitly.
wine_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine_names = ['Cultivar', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash',
              'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
              'Proanthocyanin', 'Color_intensity', 'Hue', 'OD280_OD315', 'Proline']
wine = pd.read_csv(wine_url, names=wine_names)
wine.info()
```

- **wine**: **178 rows, 14 columns**. All columns are numeric (`int64` / `float64`), so every one is a continuous-ish variable suitable for correlation. (`Cultivar` is really a class label stored as an int.)
- These are the notebook's actual column names, e.g. `Total_phenols`, `Flavanoids`, `Alcalinity_of_ash`, `Hue`, `Ash`.

---

## Eyeballing correlation: `lmplot` (scatter + regression line)

`sns.lmplot` draws a scatter plot with a fitted **linear regression line**, so you can see the direction of a relationship. The notebook walks three cases:

```python
# No / weak correlation
sns.lmplot(data=wine, x='Hue', y='Ash', height=3.5)
plt.show()

# Positive correlation
sns.lmplot(data=wine, x='Total_phenols', y='Flavanoids', height=3.5)
plt.show()

# Negative correlation
sns.lmplot(data=wine, x='Alcalinity_of_ash', y='Total_phenols', height=3.5)
plt.show()
```

- `Hue` vs `Ash`: line roughly flat, essentially **no** correlation.
- `Total_phenols` vs `Flavanoids`: line rises left to right, a **positive** correlation (a strong one, ~0.86).
- `Alcalinity_of_ash` vs `Total_phenols`: line falls left to right, a **negative** correlation (mild, ~-0.32).
- **`height`** just sizes the figure.

---

## Quantifying it: correlation coefficients

- A **correlation coefficient** is a quantitative measure of the relationship between two variables, always in **[-1, +1]**. Close to **-1** = negative correlation; close to **+1** = positive correlation.
- In pandas, get all pairwise coefficients at once with **`dataframe.corr()`**, called the **correlation matrix**:

```python
wine_corr = wine.corr()   # every column is numeric, so this works directly
wine_corr
```

> **Lab-starter note:** the notebook's heat map cells use a variable named **`wine_corr`**, but the starter never defines it (the cell that would compute it is left blank for you). The line above is that fill-in step. Its diagonal is all `1.0` and the matrix is symmetric.

---

## Visualizing it: correlation plots (heat maps)

A **correlation plot** renders the correlation matrix as a **heat map**.

```python
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(wine_corr, vmin=-1, vmax=1, center=0, ax=ax)
plt.show()
```

- **`sns.heatmap`** is an **axes-level** function, so you build the figure/axes with `plt.subplots` and pass **`ax=ax`**.
- **`vmin=-1, vmax=1`** pin the color scale to the full correlation range; **`center=0`** anchors the palette's midpoint at zero.
- A **diverging** color palette is best here: it uses **two colors**, one for values above the center (0) and one for below, and the **color intensity** shows how strong the positive or negative correlation is. Change the palette with the **`cmap`** argument (see Seaborn's [color palettes tutorial](https://seaborn.pydata.org/tutorial/color_palettes.html)).

### Add the numbers with `annot`

```python
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(wine_corr, vmin=-1, vmax=1, center=0, ax=ax, annot=True)
plt.show()
```

- **`annot=True`** writes the actual correlation coefficient inside each cell, so you get color and number together.

---

## Quiz-ready facts

- **Correlation coefficient** = quantitative measure of a linear relationship; range **[-1, +1]** (near +1 positive, near -1 negative, near 0 none).
- **`dataframe.corr()`** returns the **correlation matrix**; diagonal is all **1.0** and it is **symmetric**.
- **`sns.lmplot`** = scatter plot with a fitted **regression line** (for eyeballing direction); **`height`** sizes the figure.
- **`sns.heatmap`** is **axes-level**: create the figure with `plt.subplots`, pass **`ax=`**, and use **`vmin=-1, vmax=1, center=0`**.
- Use a **diverging** palette (two colors from a neutral center; intensity = strength); swap it with **`cmap`**.
- **`annot=True`** overlays the numeric coefficient on each heat map cell.
- wine dataset: **178 rows, 14 columns**, all numeric; actual column names include `Total_phenols`, `Flavanoids`, `Alcalinity_of_ash`, `Hue`, `Ash`.

> **See also:** "Continuous-to-Continuous Relationships - Correlation Plots (Cliff Notes)" (lecture note, same module) for the full walkthrough of positive vs negative correlation and the correlation-matrix examples, and "Pairplots - Notebook (Cliff Notes)" for scanning many pairwise relationships visually before quantifying them.

---

*Source: CMPIF2100 lab notebook (personal study use).*
