# Scatter Plots (Lab Notebook): Continuous-to-Continuous Relationships

**Module 9 Cliff Notes** | Source: lab notebook `Module_9_scatterplots_start.ipynb`

---

## TL;DR

- Scatter plots visualize **continuous-to-continuous** relationships (two numeric variables).
- Figure-level function: **`sns.relplot`**; axis-level: **`sns.scatterplot`**. Default `kind` is `"scatter"`.
- Add categorical variables with **`hue`** (color), **`col`** (column facets), and **`row`** (row facets): up to 2 continuous plus 3 categorical in one figure.

---

## Setup

```python
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset('penguins')
penguins.info()
```

- **penguins**: 344 rows, 7 columns.
  - Categorical: `species`, `island`, `sex`.
  - Continuous: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- Some columns have missing values: bill / flipper / body_mass are 342 non-null, sex is 333 non-null.

---

## Basic scatter plot

```python
sns.relplot(data=penguins, x='body_mass_g', y='bill_length_mm', kind='scatter', height=3)
plt.show()
```

- **`relplot`** is the figure-level function for relationships between continuous variables; the underlying axis-level function is **`sns.scatterplot`**.
- **`kind='scatter'` is the default**, so it is optional (this cell passes it explicitly).
- `height` just sizes the figure.
- This cell pairs `body_mass_g` against `bill_length_mm`. For how to read correlation direction from the point cloud, see the lecture note cross-referenced at the bottom.

---

## Incorporating categorical variables

Scatter plots can show 3 or more variables: 2 continuous, plus up to 3 categorical using color and facets.

### Color by species (`hue`)

```python
sns.relplot(data=penguins, x='flipper_length_mm', y='bill_depth_mm', hue='species', height=3.5)
plt.show()
```

### Column facets (`col`)

```python
sns.relplot(data=penguins, x='flipper_length_mm', y='bill_depth_mm', col='species', height=3.5)
plt.show()
```

- One panel per species.

### Color plus column facets

```python
sns.relplot(data=penguins, x='flipper_length_mm', y='bill_depth_mm',
            hue='species', col='island', height=3.5)
plt.show()
```

- Color by species, one column per island.

### Three categoricals: color plus column plus row facets

```python
sns.relplot(data=penguins, x='flipper_length_mm', y='bill_depth_mm',
            hue='species', col='island', row='sex', height=3.5)
plt.show()
```

- `hue='species'`, `col='island'`, `row='sex'`: two continuous axes plus three categorical groupings. This gets granular fast, and many species / island / sex combinations have few or zero data points.

---

## Quiz-ready facts

- Figure-level function: **`sns.relplot`**; axis-level: **`sns.scatterplot`**. Default `kind="scatter"`.
- penguins = 344 rows, 7 columns; the continuous columns are `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- Add categoricals with **`hue`** (color), **`col`** (columns), **`row`** (rows): up to 2 continuous plus 3 categorical variables in one figure.
- One dot = one row, placed by its X and Y values.

> **See also:** the lecture-based note "Continuous-to-Continuous Relationships - Scatterplots (Cliff Notes)" for reading correlation direction and the within-group trend reversal (Simpson's paradox).

---

*Source: CMPIF2100 lab notebook (personal study use).*
