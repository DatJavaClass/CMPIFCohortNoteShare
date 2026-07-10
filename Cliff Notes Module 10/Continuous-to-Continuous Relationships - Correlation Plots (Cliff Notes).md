# Correlation Plots: Continuous-to-Continuous Relationships

**Module 10 Cliff Notes** | Source: lecture transcript on correlation plots (continuous-to-continuous relationships)

---

## TL;DR

- A **correlation** is a single summary statistic that quantifies the **linear** relationship between two continuous variables. It runs from **-1 to +1**.
- **+1** = perfect positive correlation (both rise together), **-1** = perfect negative correlation (one rises as the other falls), **0** = no linear relationship. The closer to either end, the stronger the relationship.
- Compute a whole **correlation matrix** with pandas **`.corr()`**, then visualize it as a **correlation plot** (a **heat map** via **`sns.heatmap`**) using a **diverging** color palette centered on zero.
- Scatter plots + a **regression line** (**`sns.lmplot`**) let you *eyeball* correlation; `.corr()` and the heat map *quantify* it, which scales far better once you have 20 or 30 variables.

> **Why bother:** eyeballing scatter plots is fine for a couple of variables, but when you have lots of continuous variables a single correlation number (and a heat map of many of them at once) tells you which pairs relate without squinting at every scatter plot.

---

## Setup and dataset

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# The wine dataset, read straight from a URL (pandas can read a URL directly).
# The raw file has no header and poor column names, so supply your own list.
# (Standard UCI wine dataset URL; the transcript references but does not show it.)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
column_names = [...]   # a custom list of names the lecturer defined
wine = pd.read_csv(url, names=column_names)
wine
```

- The dataset has **178 rows** (178 types of wine), each with chemistry-style characteristics: alcohol content, grape cultivar, ash, and so on.
- **Almost every column is a continuous variable** (ints and floats), which is exactly what correlation is built for.

> **Note on column names:** the transcript spells the columns aloud (`hue`, `ash`, `total_phenols`, `flavonoids`, `alkalinity_of_ash`, etc.) but the exact identifier strings come from a custom list the lecturer typed, which is not shown in the transcript. Treat the names in the code below as stand-ins for that list.

---

## Eyeballing correlation: scatter plots + regression line

Start with a plain scatter plot, then drop a **linear regression line** on top with **`sns.lmplot`** (same idea as `relplot`, but it draws the best-fit line).

### No / weak correlation

```python
# Plain scatter plot
sns.relplot(data=wine, x="hue", y="ash")

# Same variables, now with a regression line
sns.lmplot(data=wine, x="hue", y="ash")
```

- The dots are scattered all over with no clear pattern; the regression line is roughly **flat (slope near zero)**. **This data is uncorrelated.**
- Practical note from the lecturer: on later assignments, if data does not *look* correlated, it is fine to just say it is not very correlated. You do not have to squint.

### Positive correlation

```python
sns.lmplot(data=wine, x="total_phenols", y="flavonoids")
```

- As **total phenols** (x) increase, **flavonoids** (y) increase too. The points climb together and the regression line runs **bottom-left to top-right**. That is a **positive correlation**.

### Negative correlation

```python
sns.lmplot(data=wine, x="alkalinity_of_ash", y="total_phenols")
```

- Notice the empty white space in the **bottom-left** and **top-right** corners, a hint of the opposite pattern.
- The regression line runs **top-left to bottom-right**: as **alkalinity of ash** increases, **total phenols** generally decrease. That is a **negative correlation** (one variable decreases as the other increases). It is not a strong relationship here, but the direction is clear.

---

## Quantifying it: the correlation coefficient

- The **correlation coefficient** is a single number in **[-1, +1]** that measures the linear relationship. (The lecture does not derive the formula, just how to use it.)
  - Near **+1** = strong positive, near **-1** = strong negative, near **0** = little to no linear relationship.
- In pandas, select the columns you want and call **`.corr()`**:

```python
cols = ["hue", "ash", "total_phenols", "flavonoids", "alkalinity_of_ash"]
wine[cols].corr()
```

This returns a **correlation matrix**:

- The **diagonal is all 1.0** (every variable is perfectly correlated with itself), a quick sanity check.
- The matrix is **symmetric** (the value for A-vs-B equals B-vs-A).
- Reading the transcript's examples off the matrix:
  - `hue` and `ash`: essentially **no** correlation.
  - `hue` and `total_phenols`: some positive correlation; `hue` and `flavonoids`: **stronger** positive.
  - `total_phenols` and `alkalinity_of_ash`: about **-0.3**, a mild negative correlation (matching the scatter plot above). A value like -0.9 would have shown up as a much more obvious downward trend.

---

## Visualizing it: correlation plots (heat maps)

Once you get to 20 or 30 variables, a raw matrix is hard to read. A **correlation plot** renders the matrix as a **heat map**.

```python
# heatmap is an AXES-level function, so set the figure size with matplotlib first
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    wine[cols].corr(),
    vmin=-1, vmax=1,   # pin the scale to the full correlation range
    center=0,          # center the diverging palette on zero
    ax=ax,             # draw onto the axes we made
)
```

- Because `sns.heatmap` is **axes-level** (not figure-level like `relplot`/`lmplot`), you create the figure/axes with matplotlib and pass **`ax=ax`** so it draws onto them. Setting `figsize` is optional but handy.
- **`vmin=-1, vmax=1`** fix the color scale to the true range of a correlation; **`center=0`** anchors the midpoint of the palette at zero.

### Diverging color palette

- The default is a **diverging** palette: **two colors** (e.g. red and blue) meeting at a neutral center. The **intensity** of the color shows how far a value sits from the center (zero).
- So a strong positive pair shows deep red, a strong negative pair deep blue, and near-zero pairs wash out to the neutral center color.

### Changing the palette (`cmap`)

```python
sns.heatmap(wine[cols].corr(), vmin=-1, vmax=1, center=0,
            cmap="coolwarm", ax=ax)   # the lecturer's favorite; 0 reads as light gray

# other nice diverging options:
#   cmap="vlag"   -> a more muted red/blue
```

- Use the **`cmap`** argument to swap palettes. The lecturer prefers **`coolwarm`** because zero shows up as a light gray rather than black. **`vlag`** is another good, more muted diverging option. Seaborn has many more; pick by preference.

### Showing the numbers (`annot`)

```python
sns.heatmap(wine[cols].corr(), vmin=-1, vmax=1, center=0,
            cmap="coolwarm", annot=True, ax=ax)
```

- **`annot=True`** prints the actual coefficient inside each cell, so you get color *and* the number at a glance.
- Example payoff: you can read a **~0.86** correlation between **total phenols** and **flavonoids** directly off the annotated cell.

---

## Quiz-ready facts

- **Correlation** = a single summary statistic for the **linear** relationship between two **continuous** variables; range **[-1, +1]**.
- **+1** perfect positive, **-1** perfect negative, **0** none; magnitude = strength, sign = direction.
- **Positive correlation**: both variables rise together (scatter/regression line goes **bottom-left to top-right**).
- **Negative correlation**: one variable decreases as the other increases (line goes **top-left to bottom-right**).
- **`sns.lmplot`** draws a scatter plot **with a regression line** (for eyeballing correlation).
- **`df[cols].corr()`** (pandas) returns the **correlation matrix**; its **diagonal is all 1.0** and it is **symmetric**.
- A **correlation plot** = the matrix drawn as a **heat map** with **`sns.heatmap`**.
- `sns.heatmap` is **axes-level**: build the figure with matplotlib and pass **`ax=`**. Use **`vmin=-1, vmax=1, center=0`**.
- Heat maps use a **diverging** palette (two colors from a neutral center); color **intensity** = distance from zero. Swap it with **`cmap`** (e.g. `"coolwarm"`, `"vlag"`).
- **`annot=True`** overlays the numeric coefficient on each cell.
- In the wine data: `total_phenols` vs `flavonoids` ≈ **0.86** (strong positive); `total_phenols` vs `alkalinity_of_ash` ≈ **-0.3** (mild negative); dataset has **178 rows**.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
