# Module 8 - Cliff Jumper Notes

*One continuous lesson stitched from the seven Module 8 cliff notes (Introduction
to Matplotlib; What Plot Type Should You Use; Bar Charts for One Categorical
Variable; Histograms for One Continuous Variable; Using Pandas to Plot Bar Charts
and Histograms; Categorical-to-Continuous Relationships Boxplots; Plotting
Combinations of Categorical Variables). The through-line: **Module 8 is the
visualization half of EDA, and you pick a plot from two facts about your
variables plus one about your question.** The two facts are each variable's
**data type** (categorical or continuous) and its **number of unique values**;
the question is whether you are showing **one variable on its own** (a marginal
plot) or **how variables move together** (a relationship plot). Three tools stack
up to do the drawing: **matplotlib** is the low-level figure/axes scaffold,
**seaborn** sits on top of it with goal-oriented plots, and **pandas** offers a
quick built-in shortcut that calls matplotlib under the hood.*

> All code here targets **matplotlib**, **seaborn**, and **pandas** with the
> standard aliases. Datasets are the ones the source notes use: **penguins** and
> **diamonds** load straight from seaborn (`sns.load_dataset`), and **gapminder**
> comes from the *Pandas for Everyone* TSV. Outputs described (which species
> dominates, where life expectancy clusters, and so on) are the readings the
> source notes report.

---

## 1. The foundation: matplotlib figure and axes

matplotlib is the foundational plotting package in Python. It is clunky, but
seaborn and pandas plotting are both built on top of it, so the vocabulary here
carries through the whole module. The universal imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # you import the pyplot module, aliased plt
import seaborn as sns             # customary alias, introduced in section 3
```

Every matplotlib plot follows a **four-step workflow**:

1. **Define a figure and axes** (set up the canvas you will draw on).
2. **Plot the data** into the axes.
3. **Modify** axes/figure attributes (size, labels, ticks).
4. **Show** the figure.

The single most confusing piece is **figure vs. axes**:

```python
fig, ax = plt.subplots()   # returns two objects: a figure and an axes
```

- **figure** is the whole picture, the outer frame/canvas. It can hold multiple
  plots.
- **axes** is one individual subplot, the actual plotting region. It is **not**
  the x-axis or y-axis. The word "axes" (the subplot) and the word "axis" (the
  x/y direction) are different things, and the naming is genuinely confusing.

Calling `plt.subplots()` immediately renders an empty canvas. Ask for several
subplots and `axes` becomes a **NumPy array** of axes objects:

```python
fig, axes = plt.subplots(2)                       # 2 rows, 1 column
fig, axes = plt.subplots(2, 3)                    # 2 rows, 3 columns
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)  # drop repeated tick labels
```

Index the array the NumPy way, **rows first, then columns**: `axes[0]`,
`axes[1]`, or for a grid `axes[row, col]` (top-middle of a 2x3 grid is
`axes[0, 1]`). Plot by grabbing the target axes and calling a method on it, then
display the figure:

```python
data = np.linspace(0, 10, 100)   # 100 evenly spaced points from 0 to 10
fig, axes = plt.subplots(2, figsize=(4, 4))   # figsize=(width, height) in inches
axes[0].plot(np.sin(data))
axes[1].plot(np.cos(data))
plt.show()
```

`figsize=(width, height)` is in **inches** (horizontal first), so `(4, 4)` is a
square. Hold on to two terms from here, because seaborn reuses them: a **figure**
can contain many **subplots** (also called **facets**), each of which is an
**axes**.

---

## 2. Choosing a plot: data type, uniqueness, and relationships

Before drawing anything, decide *what* to draw. The honest answer is "it
depends," and it depends on your **purpose** plus, the focus of this course, the
**data itself**. Three properties of the data drive the choice.

**Factor 1: the variable's data type.** Two types matter here:

| Type | Description | pandas dtype |
|---|---|---|
| **Categorical** | Non-numeric, usually strings, either/or values (dog/cat, yes/no) | `object` (see the version note) |
| **Continuous (numeric)** | Integers and floats, specific values, decimals, percentages | `int`, `float` |

Check types directly with `.dtypes` (faster than scanning `.info()`):

```python
df.dtypes
```

> **Version note on the string dtype.** Older course material says a text column
> shows up as `object`. That was true on pandas 1.x and 2.x. On pandas 3.0, text
> columns default to a dedicated `str` dtype. If a quiz keys to the `object`
> answer, give the course's answer, but know a current install may print `str`.

**Factor 2: the number of unique values.** A numeric column with **few** unique
values can be treated as categorical. Rule of thumb: fewer than roughly **20**
unique values is a candidate. In gapminder, `year` has only about **12** unique
values, so treating it as a category is reasonable. It is a judgment call (a year
collected in just 2021/2022 is clearly categorical, a wide continuous range is
not).

**Factor 3: one variable, or a relationship?** Everything that plots a single
variable's distribution is a **marginal plot** (also "marginal distribution"):
you ignore every other variable and look at one. Plots that show how **two or
more** variables move together are **relationship plots**. This single question
splits the rest of the module in half.

That gives a compact decision guide:

| Your data | Reach for |
|---|---|
| One categorical variable | **bar chart** of value counts (section 3) |
| One continuous variable | **histogram** (section 4) |
| Categorical vs. continuous | **boxplot** (section 6) |
| Categorical vs. categorical | **grouped counts / heatmap / facets** (section 7) |

---

## 3. One categorical variable: bar charts

For a single categorical variable, plot the **count of each unique value**: how
many rows fall into dog vs. cat, Adelie vs. Gentoo, and so on. This is a marginal
plot. This is also where **seaborn** enters. seaborn is built on matplotlib but
is friendlier and integrates tightly with pandas DataFrames. It offers two
levels of function:

- **Figure-level** functions are goal-oriented, the good default for quick looks,
  and their big advantage is easy **multiple subplots (facets)** in one figure.
- **Axes-level** functions draw one specific plot into one axes and give finer
  control.

Load the penguins dataset (each row is one measured penguin; `species`,
`island`, `sex` are categorical, the measurements are continuous):

```python
penguins = sns.load_dataset('penguins')
```

**Figure-level: `catplot`** is seaborn's categorical workhorse. For counts, pass
`kind='count'` and seaborn counts the rows per value for you (no manual
`value_counts()` needed):

```python
sns.catplot(data=penguins, x='sex', kind='count', height=3)
plt.show()   # plt.show() suppresses the printed description of the returned object
```

Size figure-level plots with `height` (height of each facet in inches, rough
because DPI affects the real size) and `aspect` (width = aspect x height, raise
it for wider bars):

```python
sns.catplot(data=penguins, x='island', kind='count', height=3, aspect=2)
plt.show()
```

**Axes-level: `countplot`** is what `catplot(kind='count')` calls underneath. It
draws into a single axes, so to control its size you make your own figure/axes
and pass `ax=`:

```python
fig, ax = plt.subplots(figsize=(2, 3))   # width, height in inches
sns.countplot(data=penguins, x='sex', ax=ax)
plt.show()
```

Swap `x=` for `y=` to flip the bars horizontal, which reads better with long
category labels:

```python
fig, ax = plt.subplots(figsize=(3, 2))
sns.countplot(data=penguins, y='sex', ax=ax)
plt.show()
```

So: **`catplot(kind='count')`** for the quick figure-level view, **`countplot`**
when you want axes-level control, `x=` for vertical bars and `y=` for horizontal.

---

## 4. One continuous variable: histograms

A continuous variable is the marginal-plot counterpart to a categorical one, but
you **cannot count every unique value**, because numeric data often has a
different value in nearly every row. Instead, build a **histogram**: chop the
value range into **intervals (bins)** and count how many values land in each.
That reveals the **distribution**, where values concentrate and where they are
sparse. Use the gapminder data (categorical `country`, `continent`, continuous
`lifeExp`, `pop`, `gdpPercap`, plus the borderline `year`):

**Figure-level: `displot`** (distribution plot) with `kind='hist'`:

```python
sns.displot(data=gap, x='lifeExp', kind='hist', height=4)
plt.show()
```

Life expectancy clusters around 70 to 75, with a secondary group near 40 to 50.
Control the number of intervals with **`bins`**:

```python
sns.displot(data=gap, x='lifeExp', kind='hist', bins=10)
plt.show()
```

Favor a **smallish** number of bins: you want the general shape at a glance.
Larger datasets can support more bins, but too many (say 200) gives choppy spikes
that add noise, not information.

**Axes-level: `histplot`** is what `displot(kind='hist')` calls underneath, and
takes a `figsize` through your own fig/ax:

```python
fig, ax = plt.subplots(figsize=(4, 3))
sns.histplot(data=gap, x='gdpPercap', ax=ax)
plt.show()
```

`gdpPercap` piles up at low values then drops off sharply: it is **right-skewed**
(a trait that comes back to bite boxplots in section 6).

**KDE (kernel density estimate)** is a smooth alternative to bars. Its **y-axis is
density, not counts**, where higher density means more rows concentrated there.
You can show it alone or overlay it on the histogram:

```python
sns.displot(data=gap, x='lifeExp', kind='kde')               # smooth, no bars
sns.displot(data=gap, x='lifeExp', kind='hist', kde=True)    # bars + smooth trend line
plt.show()
```

To sweep every numeric variable at once, grab the numeric column names and loop:

```python
num_cols = gap.select_dtypes('number').columns
for col in num_cols:
    sns.displot(data=gap, x=col, kind='hist', kde=True, height=3)
plt.show()
```

`select_dtypes('number')` selects columns by dtype and `.columns` gives their
names. For real assignments, consider dropping `year` first (its low unique
count makes it more categorical than continuous), and expect heavily skewed
variables like `pop` and `gdpPercap` to look odd, where fewer bins help.

---

## 5. The quick path: pandas built-in plotting

Both sections above set up seaborn. When you just want a fast look, **pandas has
plotting built in** that calls matplotlib under the hood, no seaborn or explicit
matplotlib calls required. Two recipes mirror the two marginal plots above.

**Bar charts** come from calling `.plot.bar()` on a **Series** (a single column,
or the output of `.value_counts()`). `.plot(kind='bar')` is equivalent:

```python
df['team'].value_counts().plot.bar()
plt.show()
```

`.value_counts()` produces counts per category, and `.plot.bar()` draws one bar
each. Pass styling arguments straight in:

```python
df['team'].value_counts().plot.bar(
    title='Team Frequency',
    xlabel='Team', ylabel='Count',
    rot=45,             # rotate x tick labels
    figsize=(3.5, 3.5)  # width, height in inches
)
plt.show()
```

For finer control (for example several datasets on one plot), make your own
`Figure`/`Axes` and pass `ax=`, then tweak the axes directly:

```python
fig, ax = plt.subplots(figsize=(3.5, 3.5))
df['team'].value_counts().plot.bar(ax=ax)
ax.set_title('Team Frequency')
ax.set_xlabel('Team'); ax.set_ylabel('Count')
ax.tick_params(axis='x', labelrotation=45)   # the fig/ax way to do rot=45
plt.show()
```

**Histograms** come from `.plot.hist()` (or `.hist()`) on a Series, and calling
`.hist()` on a whole **DataFrame** draws a grid of histograms for every numeric
column at once:

```python
gapminder.lifeExp.plot.hist()   # one continuous column
plt.show()
gapminder.hist()                # all numeric columns at once
plt.show()
```

Rule of thumb: pandas plotting is quick and simple, reach for seaborn or
matplotlib directly when you need more control or polish.

---

## 6. Categorical-to-continuous relationships: boxplots

Now cross the line from marginal plots to **relationship plots**. The first
relationship is **categorical to continuous**: how does a continuous variable
change across the values of a categorical one? The tool is the **boxplot**, which
shows the **median** and the **spread** of a distribution at a glance. Draw one
box per category, side by side, and you can immediately compare medians and
spreads. Back to penguins:

```python
sns.catplot(data=penguins, x='species', y='flipper_length_mm', kind='box', height=3)
plt.show()
```

Note the shape of the call: the same figure-level `catplot`, now with
`kind='box'`, the **categorical** variable on `x=` and the **continuous**
variable on `y=`. Vertical is the usual default these days (you can swap x/y for
horizontal). Across species you see the median flipper length (the line inside
each box) and the spread shift right away.

**Reading a boxplot**, element by element:

- **Median**: the line inside the box. Half the data points sit above it, half
  below. It is not necessarily the mean.
- **The box = the middle 50% of the data.** Sort the data into four equal-count
  **quartiles**; the box spans **Q1 to Q3**, and that width is the
  **interquartile range (IQR)**.
- **Whiskers**: lines reaching to the most extreme data point still **within
  1.5 x IQR** of the box (so they hit the true min and max only when there are no
  outliers).
- **Outliers**: any point beyond 1.5 x IQR shows up as an individual dot, read as
  far from the median. (The Adelie species shows a couple of these.)

Any distribution works, the data does **not** have to be Gaussian (a normal-curve
diagram is just a teaching aid). Add the mean on top with `showmeans=True`:

```python
sns.catplot(data=penguins, x='species', y='flipper_length_mm', kind='box',
            height=3, showmeans=True)
plt.show()
```

The mean (a small marker) usually differs slightly from the median, a hint of
asymmetry when the gap is wide.

**The big caveat: boxplots fail on skewed data.** They work best on roughly
**symmetric** distributions with most values near the middle. On a heavily skewed
variable they drown in outliers and hide the real shape. The diamonds dataset
(about 50,000 rows) makes the point, with `price` strongly right-skewed:

```python
diamonds = sns.load_dataset('diamonds')
sns.displot(data=diamonds, x='price', col='color', kind='hist', col_wrap=3)  # see the skew
plt.show()
sns.catplot(data=diamonds, x='color', y='price', kind='box', height=4)       # boxes look nasty
plt.show()
```

When you face skewed continuous distributions across categories, reach for
**faceted histograms** or **conditional KDE plots** instead. (As an aside,
diamonds carries a true pandas `category` dtype, an alternative to `object` for
categoricals, though `object` is still common and fine.)

---

## 7. Categorical-to-categorical relationships: grouped counts, facets, heatmaps

The second relationship is **categorical to categorical**: how do the counts of
one category break down across another? Three tools cover it, all on penguins.

**Grouped count plots** extend section 3's count plot with a second categorical
variable through `hue`:

```python
sns.catplot(data=penguins, x='species', hue='island', kind='count')
plt.show()
```

This colors the bars of each species by island, good when both variables have
relatively few unique values.

**Facets (subplots)** add a *third* categorical variable by splitting the plot
into panels with `col` or `row`. These arguments work in the figure-level
functions (`catplot`, `relplot`, `displot`), which is exactly why figure-level
plotting was worth setting up:

```python
sns.catplot(data=penguins, x='species', hue='island', col='sex', kind='count')
plt.show()
```

**Heatmaps** encode counts as color in a matrix of category combinations. Build
the count matrix with `pd.crosstab`, then draw it:

```python
ct = pd.crosstab(penguins['species'], penguins['island'])
sns.heatmap(ct, annot=True, fmt='d')   # annot prints the counts, fmt='d' as integers
plt.show()
```

When a categorical variable has **many** levels, heatmaps usually scale better
than grouped bar charts. (If you build count tables by hand with `unstack()`,
pass `fill_value=0` so missing combinations show as zero rather than `NaN`.)

---

## One-paragraph recap

Module 8 is EDA's visualization half, and the plot you pick falls out of two
facts about each variable plus one about your question: the **data type**
(categorical `object`/`str` vs. continuous `int`/`float`, checked with
`.dtypes`), the **number of unique values** (a numeric column with fewer than
about 20, like gapminder's `year` at 12, can be treated as categorical), and
whether you are drawing **one variable** (a marginal plot) or a **relationship**.
**matplotlib** supplies the scaffold, where a **figure** holds one or more
**axes** (subplots/facets) built by `fig, ax = plt.subplots(...)` and sized with
`figsize=(w, h)`. For one variable, use a **bar chart** of counts
(`sns.catplot(kind='count')` figure-level, `sns.countplot` axes-level, `x=`
vertical and `y=` horizontal) for categoricals, and a **histogram**
(`sns.displot(kind='hist', bins=)` figure-level, `sns.histplot` axes-level, plus
`kde` for a smooth density whose y-axis is density not counts) for continuous,
favoring few bins for shape. **pandas** gives a quick shortcut that calls
matplotlib under the hood, `series.value_counts().plot.bar(...)` and
`series.plot.hist()` (or `df.hist()` for every numeric column). For
**relationships**, a **boxplot** (`sns.catplot(kind='box')`, categorical on `x`,
continuous on `y`) shows median and spread via the box (Q1 to Q3, the IQR),
whiskers (out to 1.5 x IQR), and outliers beyond, with `showmeans=True` to add
the mean, but it misleads on skewed data, where faceted histograms or KDE plots
are better. And for two categoricals, reach for **grouped count plots**
(`hue=`), **facets** (`col=`/`row=`), or a **heatmap** of `pd.crosstab`
(`annot=True, fmt='d'`), with heatmaps scaling best when there are many levels.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
