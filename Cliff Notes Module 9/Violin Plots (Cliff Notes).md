# Violin Plots: Visualizing Distribution Across Categories

**Module 9 Cliff Notes** | Source: notebook "Module_9_Violin_Plots.ipynb"

---

## TL;DR

- A **violin plot** shows how the **distribution** of a continuous variable changes across the categories of a categorical variable.
- It layers two things:
  1. A **KDE (Kernel Density Estimate)** mirrored on each side: wider = more data there, narrower = less. This reveals **shape, skew, and multiple peaks**.
  2. A **mini boxplot** down the middle: median, IQR, and the rest of the spread.
- It is the go-to alternative to a boxplot when you care about the **shape** of the data, not just the summary stats.
- Built with **`catplot`** and **`kind="violin"`**.

---

## What a violin plot is for

- Use it to compare **distributions** of a continuous variable across categories.
- A boxplot only gives you median, quartiles, and whiskers. A violin plot adds the full **density curve**, so you can see:
  - **Shape** of the distribution
  - Whether the data is **skewed**
  - Whether there are **multiple peaks (multimodal)**
- The KDE is drawn symmetrically (mirrored) on both sides of the center line, giving the "violin" silhouette.

---

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Example 1: Diamonds (price across color)

Uses Seaborn's built-in **`diamonds`** dataset (columns include `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `price`, `x`, `y`, `z`) to compare the distribution of `price` across `color`.

```python
diamonds = sns.load_dataset("diamonds")

sns.catplot(data=diamonds, x="color", y="price", kind="violin", aspect=2)
plt.show()
```

- **`kind="violin"`** is what turns `catplot` into a violin plot.
- **`x`** = the categorical variable (color).
- **`y`** = the continuous variable (price).
- **`aspect=2`** makes the figure twice as wide as it is tall (helpful when you have several categories side by side).
- Result: the diamond price distributions are **right-skewed**, with many more datapoints at lower prices. A plain boxplot would **not** capture that skew, but the violin's KDE does.

---

## Reading the inner boxplot

By default (`inner="box"`) each violin has a little black boxplot down its center. Read it like this:

- **White dot** = the **median**. (The notebook's markdown calls this a "white line," but in current Seaborn the default `inner="box"` renders the median as a white dot inside the black IQR bar; the white *line* form is what `inner="quartile"` uses.)
- **Thick center bar** = the **interquartile range (IQR)**, i.e. the middle **50%** of the data.
- **Thin lines (whiskers)** = the rest of the distribution.

So you still get all the usual boxplot information, plus the density shape wrapped around it.

---

## Showing quartiles inside the violin (`inner="quartile"`)

Swap the default inner boxplot for horizontal dashed quartile lines with **`inner="quartile"`**:

```python
sns.catplot(data=diamonds, x="color", y="price",
            kind="violin", aspect=2, inner="quartile")
plt.show()
```

This draws horizontal lines at:

- The **median** (50th percentile)
- The **25th percentile**
- The **75th percentile**

Useful when you want the quartile cut points marked directly on the density curve instead of a boxplot glyph.

---

## Why violin over boxplot?

| A boxplot shows | A violin plot adds |
| --- | --- |
| Median, IQR, whiskers | The full **KDE shape** |
| A five-number summary | Whether the data is **skewed** |
| A single implied hump | Whether it is **multimodal** (multiple peaks) |

Bottom line: reach for a violin plot whenever the **shape** of the distribution matters, not just its summary numbers.

---

## Example 2: GapMinder (life expectancy across years)

The second example loads the GapMinder dataset (1704 rows; columns `country`, `continent`, `year`, `lifeExp`, `pop`, `gdpPercap`) from a URL and plots how life expectancy is distributed across years.

```python
gap_url = "https://raw.githubusercontent.com/chendaniely/pandas_for_everyone/master/data/gapminder.tsv"
gap_df = pd.read_csv(gap_url, sep="\t")

sns.catplot(data=gap_df, x="year", y="lifeExp", kind="violin", aspect=2)
plt.show()
```

- Note the file is **tab-separated**, so `pd.read_csv(..., sep="\t")`.
- **`year`** is an integer but is treated as a **category** here (a limited number of discrete years).

**Interpretation:** over time, not only does the **median life expectancy increase**, but the **shape shifts**: the distribution develops a dominant peak toward the higher end. That kind of shape change is exactly what a violin plot reveals and a boxplot would miss.

---

## Practice exercises (from the notebook)

The notebook ends with three practice prompts. Two are solved with code:

```python
# 1. price vs cut (diamonds)
sns.catplot(data=diamonds, x="cut", y="price", kind="violin", aspect=2)
plt.show()

# 2. lifeExp vs continent (gapminder)
sns.catplot(data=gap_df, x="continent", y="lifeExp", kind="violin", aspect=2)
plt.show()
```

Exercise 3 is a written prompt (no code): compare the violin plots against a boxplot and describe the extra information the violin provides (answer: distribution **shape**, **skew**, and **multimodality**).

---

## Other common options (Seaborn reference, NOT shown in this notebook)

> These arguments are standard on Seaborn violin plots but are **not** demonstrated in Module_9_Violin_Plots.ipynb. Included for study reference; verify against the current Seaborn docs before quoting exact behavior.

- **`hue="..."`**: split each category by a second categorical variable, mapped to color (same idea as in point plots).
- **`split=True`**: when `hue` has exactly **two** levels, draw one level on each side of the violin instead of side-by-side violins (makes the two halves directly comparable).
- **`inner=`**: besides `"quartile"`, Seaborn also accepts `"box"` (the default mini boxplot), `"point"` / `"stick"` (marks individual observations), and `None` (no interior).
- **`cut=`**: how far past the most extreme datapoints the KDE is allowed to extend (in bandwidth units); `cut=0` clips the density to the actual data range.
- **`bw_adjust=`**: scales the KDE smoothing bandwidth (values < 1 = bumpier/less smooth, > 1 = smoother).
- **`sns.violinplot(...)`**: the axes-level function that draws the same plot without `catplot`'s figure-level wrapper.

---

## Quiz-ready facts

- A violin plot = **KDE (mirrored on both sides)** + an inner **mini boxplot**.
- Built with **`catplot`** and **`kind="violin"`** (`x` = categorical, `y` = continuous).
- **Wider** part of the violin = **more data** at that value; **narrower** = less. This exposes **skew** and **multiple peaks**.
- Inner boxplot (default `inner="box"`): **white dot = median**, **thick center bar = IQR (middle 50%)**, **thin whiskers = the rest**.
- **`inner="quartile"`** replaces the inner box with horizontal lines at the **median, 25th, and 75th percentiles**.
- Main advantage over a boxplot: it shows the **shape** of the distribution (skew, multimodality), not just the summary numbers.
- **`aspect=2`** widens the figure; it does not change the statistics.
- GapMinder example: over the years, **median life expectancy rises AND the distribution's shape shifts** toward a dominant high-end peak.
- GapMinder file is **tab-separated**: read it with **`sep="\t"`**.

---

*Source: University of Pittsburgh course notebook Module_9_Violin_Plots.ipynb (personal study use).*
