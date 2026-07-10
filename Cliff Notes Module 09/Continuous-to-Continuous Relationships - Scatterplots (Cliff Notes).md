# Scatter Plots: Continuous-to-Continuous Relationships

**Module 9 Cliff Notes** | Source: lecture transcript on scatter plots (continuous-to-continuous relationships)

---

## TL;DR

- A **scatter plot** is the fundamental visualization for a **continuous-to-continuous** relationship: two numeric variables plotted against each other, one dot per row.
- Figure-level function: **`sns.relplot`** (default `kind="scatter"`). Axis-level function: **`sns.scatterplot`**.
- Layer in **categorical** variables with **`hue`** (color), **`col`** (column facets), and **`row`** (row facets) to show three or more variables at once.

> **Note:** scatter plots specifically visualize **continuous-to-continuous** relationships. Categorical variables are layered on top (color, facets), they are not the core of the plot.

---

## What a scatter plot is for

- Use it to see how **two numeric (continuous) variables** relate: plot one against the other.
- Every **dot = one data point = one row** of the DataFrame, positioned by its X and Y values.
- Switching which variable is X vs Y does not matter much; the relationship reads the same either way.

---

## Basic code

```python
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
# each row = one measured penguin (species, island, plus continuous measurements)

# relplot is figure-level; kind="scatter" is the default, so you can omit it
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm")
```

- **`relplot`** is the figure-level function for relationships between continuous variables.
- **`kind="scatter"` is the default**, so you do not have to pass it. Add `kind="scatter"` only if you want to be explicit.
- The underlying axis-level function is **`sns.scatterplot`**.
- A **`height`** argument just sizes the figure. It is optional; the lecturer added it only to fit his screen.

---

## Reading a scatter plot and eyeballing correlation

- Each dot's position = that row's values for the two variables.
- Example: a point near flipper length just above 200 and bill depth just above 14 is one penguin with those two measurements.
- Look for **direction**: a cloud trending from **bottom-left to top-right** suggests a **positive correlation**.
  - Transcript example: **body mass vs bill length** trends bottom-left to top-right, hinting that higher body mass goes with longer bill length.
- Coming later in the module: drawing a **best-fit / linear regression line**, and directly **calculating correlations** and visualizing them with a **heat map**.

---

## Adding categorical variables (three or more variables)

Scatter plots can show more than two variables by layering categoricals on top of the two continuous axes:

- **`hue`** = color by a category.
- **`col`** = split into column facets (side-by-side panels).
- **`row`** = split into row facets (stacked panels).

### Color by category (`hue`)

```python
sns.relplot(
    data=penguins,
    x="flipper_length_mm",
    y="bill_depth_mm",
    hue="species",
    height=3.5,   # optional, just sizes the figure
)
```

- Coloring by **species** reveals the groups: points with **small bill depth but large flipper length** are **Gentoo**, which separates out neatly. **Adelie** and **Chinstrap** stay mixed together.

### Key insight: the trend can reverse within groups

- **Without color**, the overall cloud looks like **higher bill depth goes with lower flipper length** (a line from top-left to bottom-right, a negative trend).
- **With color by species**, the relationship flips: **within each species**, a longer bill depth goes with a longer flipper length (a positive trend inside each group).
- Lesson: an overall trend can hide, or even reverse, the real within-group relationship. Coloring by category exposes it.

> **Aside (not stated in the lecture):** this reversal is the classic example of **Simpson's paradox**: a relationship that holds within every subgroup can flip when the subgroups are pooled together.

### Facets (`col`, `row`) and combining them

```python
# Column facets: one panel per species
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm", col="species")

# Combine facets with color: facet by island, color by species
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            col="island", hue="species")

# Three categoricals: add row facets (very granular; some panels end up empty)
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            col="island", row="sex", hue="species")
```

- Faceting by **island** with color by **species** shows, for example, that **Torgersen** island contains only **Adelie** penguins (its facet has just one species' color).
- Pushing to **three categorical variables** by adding **`row`** (e.g. `row="sex"`) gets granular fast: many species / island / sex combinations have few or zero data points, so some panels come up empty.

---

## Quiz-ready facts

- Scatter plots visualize **continuous-to-continuous** relationships (two numeric variables).
- Figure-level function: **`sns.relplot`**; axis-level: **`sns.scatterplot`**. `relplot`'s default `kind` is **`"scatter"`**.
- **One dot = one row**; its position = its X and Y values. Swapping X and Y does not change the relationship.
- A **bottom-left to top-right** cloud = **positive** correlation.
- Add categorical variables with **`hue`** (color), **`col`** (column facets), **`row`** (row facets); a scatter plot can show **three or more** variables this way.
- **Gentoo** separates cleanly by bill depth vs flipper length; **Adelie** and **Chinstrap** overlap.
- An overall trend can **reverse within groups**: coloring by species flips the bill-depth vs flipper-length relationship from negative overall to positive within each species.
- Later in the module: **regression best-fit line**, and **correlation** values shown on a **heat map**.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
