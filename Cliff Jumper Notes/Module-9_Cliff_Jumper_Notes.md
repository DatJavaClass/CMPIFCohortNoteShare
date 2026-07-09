# Module 9 - Cliff Jumper Notes

*One continuous lesson stitched from the seven Module 9 cliff notes
(Categorical-to-Continuous Point Plots, lecture and notebook;
Continuous-to-Continuous Scatterplots, lecture and notebook; Visualizing Point
Density in Scatterplots; Faceted Histograms and Conditional KDE; Violin Plots).
The through-line: **Module 9 is about relationship plots, visualizing how two
variables move together, and the plot you pick falls out of the data types of the
pair.** When **both variables are continuous**, you reach for a **scatterplot**.
When **one is categorical and one is continuous**, you choose between comparing a
**summary statistic** (a point plot of group means with confidence intervals) and
comparing **whole distributions** (faceted histograms, conditional KDE curves, or
violin plots). Three seaborn figure-level functions carry the module: **`relplot`**
for continuous-vs-continuous, **`catplot`** for a categorical x, and **`displot`**
for distributions, each with the same figure-level/axis-level pairing and the same
`hue` / `col` / `row` facet controls introduced in Module 8.*

> All code here targets **seaborn**, **pandas**, and **matplotlib** with the
> standard aliases (`sns`, `pd`, `plt`). Datasets are the ones the source notes
> use: **penguins** and **diamonds** load straight from seaborn
> (`sns.load_dataset`), and **gapminder** comes from the *Pandas for Everyone* TSV
> (read with `sep="\t"`). Described readings (which species separates out, where
> life expectancy clusters, and so on) are the observations the source notes
> report, not fresh reads of the rendered images.

---

## 1. The map: relationship plots by variable type

Module 8 taught **marginal plots**, one variable's distribution on its own (bar
charts, histograms). Module 9 crosses fully into **relationship plots**: how do
**two** variables move together? The single most useful habit for the whole module
is to name the **data types of the pair** first, because that choice picks the
plot:

| The pair | Reach for | Section |
|---|---|---|
| Continuous vs continuous | **scatterplot** (`relplot`) | 2, plus density in 3 |
| Categorical vs continuous, comparing **averages** | **point plot** (`catplot`, `kind="point"`) | 4 |
| Categorical vs continuous, comparing **whole distributions** | **faceted histogram** / **conditional KDE** (`displot`), **violin plot** (`catplot`, `kind="violin"`) | 5, 6 |

Two pieces of Module 8 vocabulary carry straight over and are worth restating,
because every plot below is built from them:

- **Figure-level vs axis-level.** Each seaborn plot type has a **figure-level**
  function (goal-oriented, and the one that can split into multiple panels) and an
  **axis-level** function underneath it that draws into a single axes. The three
  figure-level functions in this module and their axis-level partners:

  | Figure-level | Axis-level | Used for |
  |---|---|---|
  | `sns.relplot` | `sns.scatterplot` | scatterplots (relationships) |
  | `sns.catplot` | `sns.pointplot`, `sns.violinplot` | categorical x (points, violins) |
  | `sns.displot` | `sns.histplot`, `sns.kdeplot` | distributions (hist, KDE) |

- **`hue` / `col` / `row`.** Layer extra categorical variables onto any
  figure-level plot: **`hue`** maps a category to **color**, **`col`** splits it
  into **column** panels, **`row`** into **row** panels. These are the facet
  controls that make figure-level plotting worth using.

With that map in hand, walk the pairs.

---

## 2. Continuous vs continuous: the scatterplot

A **scatterplot** is the fundamental picture of a **continuous-to-continuous**
relationship: two numeric variables plotted against each other, **one dot per
row**. Build it with the figure-level **`sns.relplot`**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")   # each row = one measured penguin

# relplot is figure-level; kind="scatter" is the DEFAULT, so you can omit it
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm")
plt.show()
```

- **`kind="scatter"` is the default** for `relplot`, so passing it is optional.
- The axis-level partner is **`sns.scatterplot`**.
- A **`height`** argument just sizes the figure; it is optional.

**Reading it.** Each dot's position is that row's two values (a dot near flipper
length just above 200 and bill depth just above 14 is one penguin with those
measurements). **Swapping which variable is x vs y does not change the
relationship.** Look for **direction**: a cloud trending from **bottom-left to
top-right** signals a **positive correlation**. (In the lecture's `body_mass_g` vs
`bill_length_mm` example, the cloud trends up to the right, so heavier penguins
tend to have longer bills.)

**Adding categorical variables.** A scatterplot can show three or more variables
by layering categoricals onto the two continuous axes with `hue`, `col`, and
`row`:

```python
# Color by species
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            hue="species", height=3.5)

# One column panel per species
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm", col="species")

# Facet by island, color by species
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            col="island", hue="species")

# Three categoricals: add row facets (granular; some panels come up empty)
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            col="island", row="sex", hue="species")
plt.show()
```

- Coloring by **species** reveals the groups: points with **small bill depth but
  large flipper length** are **Gentoo**, which separates cleanly; **Adelie** and
  **Chinstrap** stay mixed together.
- Faceting by **island** shows, for example, that **Torgersen** holds only
  **Adelie** penguins (its panel has just one color).
- Pushing to **three** categoricals with `row="sex"` gets granular fast: many
  species / island / sex combinations have few or zero rows, so some panels are
  empty.

**The insight worth carrying: an overall trend can reverse within groups.**
Without color, the whole penguins cloud looks like **higher bill depth goes with
lower flipper length** (a negative, top-left to bottom-right trend). Color by
species and it **flips**: **within each species**, longer flippers go with deeper
bills (positive). Pooling the groups hid, and even reversed, the real within-group
relationship.

> **Aside (not stated in the lecture):** this reversal is the classic
> **Simpson's paradox**, a relationship that holds inside every subgroup can flip
> when the subgroups are pooled together.

Still to come later in the course: a **best-fit / regression line**, and computing
**correlation** values shown on a **heat map**.

---

## 3. When the dots pile up: showing density in a scatterplot

Scatterplots break down when many points **overlap**: a solid blob hides where the
data is actually densest. Two fixes, both on penguins (`flipper_length_mm` vs
`bill_depth_mm`).

**Fix 1: transparency (`alpha`).** Lower each point's opacity so overlapping dots
stack and darken:

```python
sns.relplot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            s=250, alpha=0.33, edgecolor="none")
plt.show()
```

- **`alpha=0.33`** reduces opacity, so stacked points make dense regions look
  **darker** while sparse regions stay **light**.
- **`s=250`** sets point size; **`edgecolor="none"`** removes point borders so only
  the fill stacks.
- Keeps every individual point visible, good for **moderate** overlap.

**Fix 2: a 2D histogram.** Bin the x-y plane into rectangles and color each tile by
how many points land in it:

```python
sns.displot(data=penguins, x="flipper_length_mm", y="bill_depth_mm",
            kind="hist", cbar=True)
plt.show()
```

- Same figure-level **`displot`** used for ordinary histograms, with
  **`kind="hist"`**; passing **both `x` and `y`** is what makes it 2D.
- **`cbar=True`** adds the color bar, which is the **count scale**. Each tile is a
  region, and **color intensity = number of points** (darker = more).
- Summarizes into bins rather than showing every dot, better for **heavy** overlap.

> **Watch the function pairing:** the practice answer for a 2D histogram uses the
> **axis-level `sns.histplot(..., x=, y=, cbar=True)`** instead of figure-level
> `displot`. Both draw the same 2D histogram; it is the same
> `displot`/`histplot` (and `relplot`/`scatterplot`) figure-level vs axis-level
> split seen throughout.

Rule of thumb: **transparency keeps individual points visible (moderate overlap);
the 2D histogram bins them into a density map (heavy overlap).**

---

## 4. Categorical vs continuous, comparing averages: point plots

Now switch the pair to **one categorical + one continuous** variable, and ask the
first of two questions: **do the group averages differ?** The tool is the **point
plot**: it plots the **mean** of the continuous variable for each category, with an
**error bar** around each mean. Built with **`catplot`** and **`kind="point"`**, on
gapminder:

```python
import pandas as pd

gap_url = "https://raw.githubusercontent.com/chendaniely/pandas_for_everyone/master/data/gapminder.tsv"
gap = pd.read_csv(gap_url, sep="\t")   # tab-separated file

sns.catplot(
    data=gap,
    x="year",          # integer year, treated as CATEGORICAL
    y="lifeExp",       # continuous variable being averaged
    kind="point",
    linestyle="none",  # do NOT connect the means with a line
)
plt.show()
```

- **`year` is an integer but is treated as categorical**, because `catplot` treats
  the x variable as categorical **by default**, so each distinct year becomes one
  discrete point on the axis. (It is reasonable here because year has only a
  handful of unique values, the Module 8 "few unique values" rule of thumb.)
- **`y="lifeExp"`** is the continuous variable being averaged; one point = the mean
  life expectancy for that year (it trends upward over time).
- **`linestyle="none"`** stops seaborn from drawing the connecting line it adds by
  default. The axis-level partner is **`sns.pointplot`**.

> **Watch this (a lecture slip):** the lecturer's opening line calls a point plot a
> picture of "median and spread." A seaborn point plot plots the **mean** by
> default. Answer **mean**.

**Styling the error bars:**

```python
sns.catplot(data=gap, x="year", y="lifeExp", kind="point", linestyle="none",
            capsize=0.1,                # add end caps to the error bars
            err_kws={"linewidth": 1})   # restyle the bars (here, thinner)
plt.show()
```

**Three variables at once (`hue`).** Compare one continuous plus **two**
categorical variables by mapping the second category to color:

```python
sns.catplot(data=gap, x="year", y="lifeExp", hue="continent",
            kind="point", linestyle="none", capsize=0.1)
plt.show()
```

Now you get a mean-per-year for **each continent**, with an auto-generated legend.

### The statistics: what the error bars mean

This is the payload of the section. The default error bar is a **95% confidence
interval (CI)** around the mean, and it comes from the **standard error of the mean
(SEM)**:

- The **95% CI** spans about **2 x SEM** on each side of the mean; the **68% CI**
  spans about **1 x SEM** each side.
- **Why 2 x SEM?** The SEM is the **standard deviation of the sampling distribution
  of the mean** (imagine re-sampling many times and taking the mean each time). By
  the **Central Limit Theorem**, that sampling distribution is approximately
  **Gaussian**, and the SEM is **one standard deviation** of it, so about 2 SEM
  covers roughly 95%.
- **CI width is driven mostly by sample size (n):** the more data points averaged,
  the smaller the error bar.

> **Precision note:** the lecture rounds the 95% multiplier to **2**; the exact
> value is **1.96** standard errors. For quiz answers, use the lecture's framing
> (about 2 SE for 95%, about 1 SE for 68%).

> **Mechanism note (seaborn internals):** the lecture explains the CI as
> mean +/- 2 x SEM, and that is the quiz answer. Under the hood, seaborn's default
> `errorbar=("ci", 95)` computes a **bootstrap** percentile interval (resample
> about 1000 times, take the 2.5 / 97.5 percentiles), which for large, roughly
> symmetric samples lands very close to +/- 2 x SEM. For a literal standard-error
> bar, use `errorbar=("se", 2)` (about 95%) or `errorbar=("se", 1)` (about 68%).

**Reading significance off the plot:**

- **Error bars do NOT overlap** -> the means are **significantly different** at the
  95% level; you can be 95% confident the true means differ.
- **Error bars overlap** -> the means *look* different, but you are **not** 95%
  confident they truly differ.
- Reading one CI: "I am 95% sure the true mean sits inside this little range."
- Lecture examples: **Oceania** sits at the top with error bars so small they are
  hard to see, above **Europe** across most years (in 1952 and 2007 the bars
  clearly do not overlap). In **1967**, the **Americas** mean looks higher than
  **Asia**, but their bars **overlap**, so the difference is not confident.

Change the interval with the **`errorbar`** argument:

```python
sns.catplot(data=gap, x="year", y="lifeExp", kind="point",
            linestyle="none", capsize=0.1,
            errorbar=("ci", 68))   # 68% CI instead of the default 95%
plt.show()
```

A 68% CI gives **smaller** bars, but "68% sure" is weaker than "95% sure," so 95%
stays the usual default.

> **Caveat on the significance rule:** "non-overlapping bars = significant" is the
> quiz answer, but the reverse is only a rule of thumb. Overlapping bars usually
> mean "not confident of a difference," yet two 95% CIs can overlap slightly and
> still be significantly different.

---

## 5. Categorical vs continuous, comparing whole distributions: faceted histograms and conditional KDE

A point plot compresses each group to a single number (its mean). Sometimes you
need the **whole distribution** per category, its shape, spread, and skew, not just
the average. Two tools built on **`displot`** answer "does the distribution of a
continuous variable change across categories?" First clean the columns you will
plot:

```python
penguins_clean = penguins.dropna(subset=["flipper_length_mm", "species"])
penguins_clean.shape   # (342, 7): dropna dropped the 2 rows missing those values
```

- **`dropna(subset=[...])`** drops rows missing a value in **only the listed
  columns**, not every column. Here 344 rows in, 342 out (2 dropped).

**Faceted histograms: one histogram per category, side by side.**

```python
# Facet across columns
sns.displot(data=penguins_clean, x="flipper_length_mm", col="species", kind="hist")

# Facet across rows, shrinking each panel
sns.displot(data=penguins_clean, x="flipper_length_mm", row="species",
            kind="hist", height=2.5)
plt.show()
```

- **`col="species"`** gives each species its own column panel; **`row="species"`**
  stacks them. One facet per unique category value (three species -> three panels).
- **`kind="hist"`** makes each facet a histogram (y-axis = count).
- **`height=2.5`** sets each facet's height in inches, handy so stacked rows do not
  get too tall.

**Conditional KDE: smooth curves, one per category, overlaid.** A **KDE (kernel
density estimate)** smooths a distribution into a continuous curve instead of bars.
**Conditional** means one curve per category, overlaid via `hue` so shapes compare
directly:

```python
sns.displot(data=penguins_clean, x="flipper_length_mm", hue="species", kind="kde")
plt.show()
```

- **`hue="species"`** draws one colored curve per species with an auto legend.
- The y-axis now reads **Density**, not count: it reflects where rows concentrate
  along the x-axis, not a raw row count. Do not read a KDE peak as a count.

**The one gotcha to memorize: `common_norm`.** By default seaborn factors in how
many rows each group has (the **sample-size effect**):

- **`common_norm=True`** (the **default** for conditional KDE): all curves are
  scaled together so the **total area across all groups combined equals 1**. Groups
  with fewer rows get **shorter** curves (in the notebook, the Chinstrap peak sits
  lower than Adelie because there are fewer Chinstraps). Use it when the question
  involves overall frequency ("which species is more common, and where do its
  values sit?").
- **`common_norm=False`**: each group's curve is normalized **independently so
  every curve has area 1**. Height no longer encodes group size, so shapes compare
  directly (the Chinstrap peak jumps up once the sample-size effect is removed).
  Use it for "within each species, where do values cluster?" regardless of how
  common the species is.

```python
sns.displot(data=penguins_clean, x="flipper_length_mm", hue="species",
            kind="kde", common_norm=False)
plt.show()
```

> **Source note:** the specific peak comparisons (Chinstrap lower, then higher,
> than Adelie) are the notebook's written observations, not a fresh read of the
> rendered images.

So: **faceted histogram** = separate, non-overlapping panels; **conditional KDE** =
smooth overlaid curves; and `common_norm` decides whether curve height still
carries group-size information.

---

## 6. Categorical vs continuous, distribution shape in one glyph: violin plots

A **violin plot** packs a whole distribution into a single shape per category,
combining two ideas, the KDE from section 5 and a mini boxplot:

1. A **KDE mirrored** on both sides of a center line, giving the "violin"
   silhouette. **Wider = more data** at that value, **narrower = less**, which
   exposes **shape, skew, and multiple peaks (multimodal)**.
2. A **mini boxplot** down the middle: median, IQR, and the spread.

It is the go-to alternative to a plain boxplot when the **shape** matters, not just
the five-number summary. Built with **`catplot`** and **`kind="violin"`** (`x` =
categorical, `y` = continuous):

```python
diamonds = sns.load_dataset("diamonds")

sns.catplot(data=diamonds, x="color", y="price", kind="violin", aspect=2)
plt.show()
```

- **`kind="violin"`** turns `catplot` into a violin plot; the axis-level partner is
  **`sns.violinplot`**.
- **`aspect=2`** makes the figure twice as wide as tall (helpful with several
  categories side by side); it does **not** change the statistics.
- Result: diamond `price` is **right-skewed** (many more points at lower prices),
  which a plain boxplot would flatten but the violin's KDE reveals.

**Reading the inner boxplot** (default `inner="box"`):

- **A short white horizontal tick = the median.** (The notebook's markdown calls it
  a "white line"; in current seaborn the default `inner="box"` renders the median
  as a small white tick, a short white line marker, inside the dark-gray IQR bar.
  The `inner="quartile"` form instead draws dashed lines across the violin at the
  25th, 50th, and 75th percentiles.)
- **Thick center bar = the IQR** (middle 50% of the data).
- **Thin lines = the rest of the distribution.**

Swap the inner box for dashed quartile lines with **`inner="quartile"`** (draws
horizontal lines at the **median, 25th, and 75th** percentiles):

```python
sns.catplot(data=diamonds, x="color", y="price",
            kind="violin", aspect=2, inner="quartile")
plt.show()
```

**Second example (gapminder), where the shape story pays off:**

```python
sns.catplot(data=gap, x="year", y="lifeExp", kind="violin", aspect=2)
plt.show()
```

Over the years the **median life expectancy rises** *and* the **shape shifts**: the
distribution develops a dominant peak toward the high end. That shape change is
exactly what a violin shows and a boxplot would miss. (As in the point plot, `year`
is an integer treated as a category.)

> **Reference (standard seaborn, not shown in the Module 9 notebook):** `hue=`
> splits each category by a second categorical (color); **`split=True`** puts the
> two levels of a 2-level `hue` on opposite sides of one violin; `inner=` also
> accepts `"point"` / `"stick"` (mark individual observations) and `None`; `cut=`
> controls how far past the data the KDE extends (`cut=0` clips to the data range);
> `bw_adjust=` scales the smoothing (below 1 bumpier, above 1 smoother). Verify
> exact behavior against current docs before quoting.

---

## One-paragraph recap

Module 9 is the **relationship** half of visualization, and the plot falls out of
the **data types of the pair**. **Continuous vs continuous** -> a **scatterplot**
(`sns.relplot`, default `kind="scatter"`; axis-level `sns.scatterplot`), one dot
per row, a bottom-left to top-right cloud meaning positive correlation, with
categoricals layered on via `hue` / `col` / `row` (and the within-group trend
reversal, Simpson's paradox, as the cautionary tale). When the dots overlap, add
density: **`alpha`** transparency (plus `s` and `edgecolor="none"`) for moderate
overlap, or a **2D histogram** (`sns.displot(kind="hist", cbar=True)`, or
axis-level `sns.histplot`) for heavy overlap. **Categorical vs continuous** splits
by what you compare: for **averages**, a **point plot** (`sns.catplot(kind="point",
linestyle="none")`, axis-level `sns.pointplot`) shows each group mean with a **95%
CI** error bar, about **2 x SEM** on each side (68% CI about 1 SEM each side), built from the standard
error of the mean via the Central Limit Theorem, where **non-overlapping bars =
significantly different** and CI width shrinks with sample size; adjust it with
`errorbar=("ci", 68)`, and add a second category with `hue`. For **whole
distributions**, compare shapes with **faceted histograms** (`sns.displot(col=/row=,
kind="hist")`), **conditional KDE** curves (`sns.displot(hue=, kind="kde")`, y-axis
= density, `common_norm=True` keeps the sample-size effect, `False` removes it), or
**violin plots** (`sns.catplot(kind="violin")`, axis-level `sns.violinplot`) that
wrap a mirrored KDE around a mini boxplot (white tick = median, thick bar = IQR) to
expose skew and multimodality. Three figure-level functions, `relplot`, `catplot`,
`displot`, each with an axis-level partner and the shared `hue` / `col` / `row`
facet controls, cover every relationship in the module.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
