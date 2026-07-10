# Module 10 - Cliff Jumper Notes

*One continuous lesson stitched from the eight Module 10 cliff notes (Pair Plots;
Correlation Plots, lecture and notebook; Styles and Color Palettes; Introduction
to Standardization, lecture and notebook; Standardization with Scikit-learn,
lecture and notebook). The through-line: **Module 10 is about working with many
continuous variables at once, in five moves.** First **scan** every pairwise
relationship in one grid (the **pair plot**). Then **quantify** a single
relationship as one number (the **correlation coefficient** and the **`.corr()`**
matrix). Then **visualize** the whole matrix as a **heat map** (`sns.heatmap`),
which is best shown with a **diverging** palette centered on zero. Then **style**
those figures so they read well (seaborn **color palettes** and **background
styles**, the general toolkit the heat map's `cmap` is one instance of). Finally
**prepare** the continuous data for the predictive analytics coming later in the
course by putting every variable on the same scale (**standardization** with
scikit-learn's `StandardScaler`). Module 9 gave you the single-pair relationship
plots; Module 10 scales them up and gets the data ready to model.*

> All code here targets **seaborn**, **pandas**, **matplotlib**, and (for
> standardization) **scikit-learn** with the standard aliases (`sns`, `pd`, `plt`,
> `np`; `StandardScaler` from `sklearn.preprocessing`). Datasets are the ones the
> source notes use: **penguins** loads straight from seaborn (`sns.load_dataset`),
> and the **wine** dataset reads from its UCI URL with column names supplied by
> hand (the raw file has no header). Described plot readings (which group separates
> out, which correlation is strong) are the observations the source notes report,
> not fresh reads of rendered images. All numbers below were reproduced by
> execution (seaborn 0.13.2, pandas 2.3.3, scikit-learn 1.7.2).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")   # 344 rows x 7 cols; 4 continuous, 3 categorical

wine_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_names = ["Cultivar", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash",
              "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols",
              "Proanthocyanin", "Color_intensity", "Hue", "OD280_OD315", "Proline"]
wine = pd.read_csv(wine_url, names=wine_names)   # 178 rows x 14 cols, effectively all numeric
```

---

## 1. The map: five moves on many continuous variables

Module 9 crossed from marginal plots into **relationship plots** for a single pair
of variables. Module 10 keeps the continuous-to-continuous thread but **scales it
up to many variables at once**, and then adds a step Module 9 never needed:
preparing the data itself. Name the move you want and the tool falls out:

| The move | Reach for | Section |
|---|---|---|
| **Scan** every pairwise relationship at once | **pair plot** (`sns.pairplot`) | 2 |
| **Quantify** one pair as a single number | **correlation coefficient**, `dataframe.corr()` (eyeball first with `sns.lmplot`) | 3 |
| **Visualize** the whole correlation matrix | **correlation heat map** (`sns.heatmap`) | 4 |
| **Style** how any figure looks | **color palettes** (`palette=`) and **styles** (`sns.set_style`) | 5 |
| **Prepare** variables onto one scale for modeling | **standardization** (`StandardScaler`) | 6 |

The first three moves are one arc (scan a relationship, put a number on it, draw
the whole matrix), section 5 is the styling toolkit that makes that matrix
readable, and section 6 pivots toward the modeling still to come. Walk them in
order.

---

## 2. Scan everything at once: the pair plot

A **pair plot** (**`sns.pairplot`**) draws a grid that visualizes the
relationships among **multiple continuous variables** in one figure. It is the
fastest way to see, at a glance, which pairs are worth a closer look.

```python
sns.pairplot(data=penguins)
plt.show()
```

- **Off-diagonal** cells: a **scatter plot** for every pair of numeric columns, so
  you see all pairwise relationships at once.
- **Diagonal** cells: each variable's **marginal distribution**, drawn as a
  **histogram by default**.
- Only the **numeric** columns become axes; the categoricals (`species`, `island`,
  `sex`) are skipped unless you use them for grouping.
- Missing values are present (the four measurement columns are **342 non-null**,
  `sex` is **333 non-null**); missing points are simply not drawn in each cell.

**Reading it.** One glance surfaces patterns worth chasing: linear trends,
non-linear shapes, clusters. From there you build focused scatter plots (section 3)
on just the pairs of interest.

**Conditioning on a category with `hue`.** Color the points by a category to expose
group structure:

```python
sns.pairplot(data=penguins, hue="species")
plt.show()
```

- **`hue="species"`** colors every point by species.
- Side effect worth memorizing: with `hue` set, **the diagonal switches from
  histograms to KDE curves**, one curve per group (a conditional KDE).

**Removing the sample-size effect on the diagonal.** By default those diagonal KDE
curves keep the effect of **group size** (groups with fewer rows draw
smaller-looking curves). Turn that off to compare **shape**:

```python
sns.pairplot(data=penguins, hue="species", diag_kws={"common_norm": False})
plt.show()
```

- **`diag_kws`** passes keyword arguments through to the diagonal plotting function
  (here, the KDE). **`common_norm=False`** normalizes each group's diagonal curve
  separately, so smaller groups no longer look artificially low and distribution
  **shapes** compare directly.

> **Cross-module link (verified):** with `hue` set the pairplot diagonal **is** a
> conditional KDE, so `diag_kws={"common_norm": False}` is the **same argument with
> the same meaning** as Module 9's `sns.displot(..., kind="kde", common_norm=False)`.
> Both remove the sample-size effect so every group's curve has area 1 and the
> shapes line up. It is literally the same `kdeplot` parameter reached through a
> different door.

**Do not overload it.** Pair plots crowd fast (the grid is variables x variables):

- Best with a **small to moderate** number of variables, **3 to 6** for most
  exploratory work, **about 20 at most**.
- Restrict the grid with the **`vars`** argument:

```python
sns.pairplot(
    data=penguins,
    vars=["bill_length_mm", "bill_depth_mm", "body_mass_g"],
    hue="species",
    diag_kws={"common_norm": False},
)   # a tidy 3x3 grid
plt.show()
```

---

## 3. Put a number on it: the correlation coefficient and matrix

A pair plot shows relationships qualitatively. To **quantify** one, use a
**correlation**: a single summary statistic for the **linear** relationship between
**two continuous** variables, always in the range **[-1, +1]**.

- **+1** = perfect positive (both rise together), **-1** = perfect negative (one
  rises as the other falls), **0** = no linear relationship.
- **Magnitude = strength, sign = direction.**

**Eyeball it first with a regression line.** **`sns.lmplot`** is like `relplot` but
it draws the best-fit line on top of the scatter, so you can see the direction:

```python
sns.lmplot(data=wine, x="Total_phenols", y="Flavanoids")        # positive
sns.lmplot(data=wine, x="Alcalinity_of_ash", y="Total_phenols") # negative
plt.show()
```

- A cloud and line running **bottom-left to top-right** is a **positive**
  correlation; **top-left to bottom-right** is **negative**.

**Then quantify it.** In pandas, **`dataframe.corr()`** returns the **correlation
matrix**, every pairwise coefficient at once:

```python
wine_corr = wine.corr()   # every column is numeric, so this works directly
wine_corr
```

- The **diagonal is all 1.0** (every variable correlates perfectly with itself), a
  quick sanity check, and the matrix is **symmetric** (the A-vs-B value equals
  B-vs-A).
- The **wine** dataset is **178 rows by 14 columns**, effectively all numeric,
  which is exactly what correlation is built for.
- Reading the verified examples off the matrix: `Total_phenols` vs `Flavanoids`
  is about **0.86** (a strong positive, the climbing `lmplot` above);
  `Total_phenols` vs `Alcalinity_of_ash` is about **-0.32** (a mild negative);
  `Hue` vs `Ash` is about **0** (essentially no linear relationship).

---

## 4. Draw the whole matrix: the correlation heat map

Once you have 20 or 30 variables, reading a raw matrix of numbers is painful. A
**correlation plot** renders that matrix as a **heat map** with **`sns.heatmap`**,
so color does the reading for you.

```python
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(wine_corr, vmin=-1, vmax=1, center=0, cmap="coolwarm", annot=True, ax=ax)
plt.show()
```

- **`sns.heatmap` is axes-level**, not figure-level like `relplot`/`lmplot`, so you
  build the figure and axes yourself with `plt.subplots(...)` and pass **`ax=ax`**
  so it draws onto them. `figsize` is optional but handy.
- **`vmin=-1, vmax=1`** pin the color scale to the full correlation range;
  **`center=0`** anchors the palette's midpoint at zero.
- The palette is **diverging**: two colors meeting at a neutral center, and the
  **color intensity shows how far a value sits from zero**. Deep on one end is
  strong positive, deep on the other is strong negative, near-zero washes out to
  the center color.
- Swap palettes with the **`cmap`** argument. The lecturer's favorite is
  **`coolwarm`** (zero reads as a light gray rather than black); **`vlag`** is a
  more muted red/blue.
- **`annot=True`** prints the actual coefficient inside each cell, so you get color
  and the number together (you can read the ~0.86 for `Total_phenols` vs
  `Flavanoids` straight off its cell).

> **Bridge to the next section (verified):** the heat map's **`cmap`** is nothing
> more than the heat-map twin of the **`palette`** argument, and "a diverging
> palette centered on 0" is exactly the case the styles lesson calls **diverging**
> (a continuous quantity that spreads out from a midpoint). So the heat map is your
> first live use of the palette-matching rules in section 5.

---

## 5. Make it read well: color palettes and background styles

Two levers control how any seaborn figure looks: the **color palette**
(**`palette=`**) and the **background style** (**`sns.set_style()`**).

**`hue` is not only for categories.** It also accepts a **continuous** variable,
and seaborn then shades the points along a gradient (conceptually like the bins of
a histogram) with a colorbar:

```python
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            hue="body_mass_g", palette="viridis")
plt.show()
```

**Match the palette type to the variable type.** This is the section's core rule:

| Palette type | Shows | Best for |
|---|---|---|
| **qualitative** | separate colors for separate categories | categorical variables |
| **sequential** | shades within one or two colors | continuous variables |
| **diverging** | shades within two colors from a center | continuous variables around a midpoint (like 0) |

- The **diverging** row is exactly the correlation heat map from section 4.
- **Accessibility:** **`colorblind`** and **`viridis`** are built to be friendly for
  color-vision deficiencies. But **`colorblind` is qualitative**, so using it on a
  continuous `hue` like `body_mass_g` looks wrong. For a continuous variable reach
  for a **sequential** palette: **`mako`**, **`rocket`**, or **`viridis`**.
- **Reverse any palette** by appending **`_r`** to its name (e.g. `mako_r`), which
  flips which end of the variable gets the light versus dark shade:

```python
sns.scatterplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
                hue="body_mass_g", palette="mako_r")   # same colors, reversed order
plt.show()
```

**Change the whole background with `sns.set_style`.** It is a **global** setting:
once called, it applies to every plot drawn afterward.

```python
sns.set_style("dark")        # solid gray background, NO grid
sns.set_style("whitegrid")   # white background WITH grid lines
sns.set_style("white")       # white background, NO grid (this notebook's default)
```

- **`"dark"`**: shaded (gray) background, no grid.
- **`"whitegrid"`**: white background with grid lines, handy for reading off exactly
  where points sit.
- **`"white"`**: white background, no grid lines. This notebook calls it the
  default.

> **Watch this (a name collision worth pinning down):** seaborn's *own* default,
> the one you get from **`sns.set_theme()`**, is not `white`; it is a fourth style
> called **`darkgrid`** (gray background with grid lines). And `darkgrid` is not
> the same as the `"dark"` above: **both share the identical gray background, and
> they differ only by the grid** (`dark` has none, `darkgrid` has the grid). So the
> four names sort cleanly into a 2x2: white vs gray background, crossed with grid
> vs no grid. `white` (no grid) and `whitegrid` on the light side; `dark` (no grid)
> and `darkgrid` on the gray side.

---

## 6. Get the data ready to model: standardization

The last move pivots from *looking at* continuous variables to *preparing* them.
Most of the predictive techniques coming later in the course want their input
reshaped first, a step called **preprocessing** (preparing, cleaning), and it eats
a large share of real data-science time. The key preprocessing technique here is
**standardization**: putting different **continuous** variables on **similar
scales** so no single big-numbered column dominates.

**Why it matters.** In penguins, `body_mass_g` runs into the **thousands** (mean
about **4201.75**) while the millimeter measurements sit in the tens to low
hundreds (means about **43.92 / 17.15 / 200.92**). On a box plot `body_mass_g`
flattens the others into slivers. That scale gap distorts anything scale-sensitive: **distances
between groups**, and **how much a change in one variable affects another**.

> **Aside:** `describe()` by default summarizes only the **continuous (numeric)**
> columns, which is why it is the quick way to see the scale problem.

**Z-standardization** fixes it by transforming each column to a **mean of 0 and a
standard deviation of 1**, converting each value to its **z-score** (the number of
standard deviations it sits from the mean):

```text
z = (value - mean) / standard_deviation
```

Two steps per column: **(1) center** (subtract the mean, making the new mean 0),
then **(2) scale** (divide by the standard deviation, making the new std 1).

**Scikit-learn does it for you.** The package name is **`sklearn`**, and every
scikit-learn transform follows the same **three steps**: **initialize** an object,
**fit** it to a dataset (it *estimates the parameters it needs*), then **transform**
the dataset with the fitted object.

```python
from sklearn.preprocessing import StandardScaler

penguins_scaler = StandardScaler()                                # 1. initialize
penguins_continuous = penguins.select_dtypes("number").copy()     # continuous cols only
penguins_scaler.fit(penguins_continuous)                          # 2. fit: learn mean & std

penguins_scaler.mean_    # [  43.92,   17.15,  200.92, 4201.75]
penguins_scaler.scale_   # [   5.45,    1.97,   14.04,  800.78]
```

- **`select_dtypes("number").copy()`** grabs just the four numeric columns
  (standardization is for continuous variables only) and copies them so the fit
  never mutates the original frame.
- **Fitting** stores what it learned as **`.mean_`** and **`.scale_`** (NumPy arrays
  of four values, one per column). The trailing underscore is scikit-learn's mark
  for "learned during fit."

> **Precision note (population vs sample std, verified):** `scale_` is the
> **population** standard deviation (NumPy `ddof=0`), **not** the sample std that
> pandas `.std()` reports by default (`ddof=1`). On these columns `scale_` equals
> `std(ddof=0) = [5.45, 1.97, 14.04, 800.78]`, and differs slightly from
> `std(ddof=1) = [5.46, 1.97, 14.06, 801.95]` (body mass: 800.78 vs 801.95). Same
> idea, different divisor (`n` vs `n-1`). If a lecture calls `scale_` the "sample
> standard deviation," that wording is imprecise.

**Transform, then re-wrap.** The fitted scaler now standardizes the data:

```python
penguins_std_arr = penguins_scaler.transform(penguins_continuous)
penguins_std_arr.shape   # (344, 4)

penguins_std = pd.DataFrame(penguins_std_arr, columns=penguins_continuous.columns)
```

- **`transform` returns a new NumPy array of shape (344, 4)** and does **not**
  mutate the input; missing values (NaN) are ignored during fit and pass straight
  through as NaN.
- A raw array has no column names, so rebuild a DataFrame and hand it the original
  **`.columns`**.

**Confirm the result.** Run `penguins_std.describe()`:

- **Means are essentially 0**, printed in scientific notation like **`8.31e-17`**
  (that is basically zero).
- **The standard deviation is 1.** One wrinkle: `describe()` actually prints
  **`1.0014652`**, not a clean `1.000`, because pandas `.std()` uses the **sample**
  formula (`ddof=1`) on the **342** non-null rows: `sqrt(342/341) = 1.0014652`. In
  the **population** sense (`ddof=0`) the standardized std is **exactly 1**, which
  is the "should be 1" a quiz means. So the **quiz answer is 1**; do not be thrown
  when `describe()` reads `1.0015`.
- Standardizing changes the **scale**, not the **shape**: box plots now sit on one
  comparable scale around 0, and a histogram of any single column keeps its shape,
  only its axis ticks move (now centered on 0, with negatives).

> **Why this is the last move (verified framing):** standardization is not a
> visualization at all; it is preparation. Putting every variable on the same scale
> is what the distance-based and pattern-finding methods later in the course need,
> which is why Module 10 ends by setting the data up rather than plotting it.

---

## One-paragraph recap

Module 10 works with **many continuous variables at once**, in five moves. **Scan**
every pairwise relationship in one grid with **`sns.pairplot`** (off-diagonal
scatter, diagonal marginal that is a histogram by default and switches to a
conditional KDE under `hue`; `diag_kws={"common_norm": False}` drops the
sample-size effect exactly as Module 9's KDE did; keep it to 3 to 6 vars via
`vars=`). **Quantify** one linear relationship as a **correlation coefficient** in
**[-1, +1]** (sign = direction, magnitude = strength), eyeballed with the
regression line of **`sns.lmplot`** and computed for all pairs at once by
**`dataframe.corr()`** (diagonal all 1.0, symmetric; in wine, `Total_phenols` vs
`Flavanoids` about **0.86**, vs `Alcalinity_of_ash` about **-0.32**). **Visualize**
the matrix as a **heat map** with the axis-level **`sns.heatmap`**
(`vmin=-1, vmax=1, center=0`, `annot=True`, a **diverging** `cmap` like
`coolwarm`). **Style** any figure with **`palette=`** and **`sns.set_style()`**:
`hue` takes continuous variables too, palette **type must match variable type**
(qualitative / sequential / diverging), `colorblind` and `viridis` are accessible
(`colorblind` is qualitative), sequential `mako`/`rocket`/`viridis` suit continuous
data, `_r` reverses a palette, and the background names sort into a 2x2 where
`dark` and `darkgrid` share a gray background and differ only by the grid
(`darkgrid` being `set_theme()`'s real default). Finally **prepare** the data:
**standardization** via scikit-learn's **`StandardScaler`** (init, fit, transform)
z-standardizes each continuous column to mean 0 and std 1 using the **population**
std in `scale_`, returns a `(344, 4)` array to re-wrap as a DataFrame, and leaves
the distribution shape unchanged, getting the variables onto one scale for the
modeling still to come.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
