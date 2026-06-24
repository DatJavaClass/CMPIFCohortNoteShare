# Module 6 — Cliff Jumper Notes

*One continuous lesson stitched from the six Module 6 cliff notes (Pandas Series;
DataFrames intro; DataFrame Columns; Filtering Rows; Visualizing Random Number
Generation; plus the HW6 notes that thread NumPy → Series → DataFrame together).
The through-line: **pandas is NumPy's numeric core with labels bolted on.** A
Series is values + an index; a DataFrame is several aligned Series sharing one
index; and almost every operation is "reach in through the labels and
select/add/filter." Two ideas from Module 5 come straight back — **`axis` = the
direction an operation collapses** (front and center in the capstone), and
**views vs. copies** (which resurfaces in the Copy-on-Write note on columns).*

> All code here was executed against **pandas 3.0.3 / NumPy 2.4.6** on Windows;
> outputs shown are real. Where current pandas differs from what an older lab
> printed — the **string dtype** and **Copy-on-Write** — it's flagged inline.

---

## 1. Why pandas: NumPy with labels

pandas is the Python package for data analysis. It's **built on top of NumPy**
but tailored to data work, and it borrows much of its feel from R while living
inside general-purpose Python. The universal convention:

```python
import pandas as pd          # you will type pd constantly
import numpy as np           # pandas sits on NumPy, so np comes along
import matplotlib.pyplot as plt   # used for the capstone plots in §7
```

The whole module is two data structures and the operations on them:

- **Series** — one-dimensional (a single labeled column of values).
- **DataFrame** — two-dimensional (a whole table; the way you'll normally work).

Underneath both is a NumPy array holding the **numeric core**; pandas adds
**labels** on top (an *index* for rows, plus *column names* for a DataFrame).
That "NumPy + labels" picture is the spine of everything below.

---

## 2. Series — labeled 1D data

A `Series` holds one-dimensional data — think of it as a labeled list. Build one
from a Python list (or a NumPy array):

```python
s = pd.Series([10, 20, 30, 40, 50, 60])
```

Every Series has **two parts**:

- **`s.index`** — the labels for each element (like dict keys). The default is a
  `RangeIndex(start=0, stop=6, step=1)` — i.e. labels `0 … n-1`.
- **`s.values`** — the actual data, returned as a NumPy `ndarray` (here
  `dtype int64`). This is the "NumPy underneath" made literal.

Both are **attributes** (no parentheses).

**Access is by index *label*, in brackets:**

```python
s[5]      # -> 60   (label 5 exists)
s[-1]     # -> KeyError: -1
```

This is the first real trap, and it's the opposite of Python lists / NumPy:
**negative indexing does not work.** With a default integer index, `s[-1]` looks
for a *label* called `-1`, doesn't find one, and raises `KeyError`. To grab the
last element by **position**, ask for the actual last label (`s[s.index[-1]]`) or
use `.iloc[-1]` (positional access, introduced in §6).

**Indexes are customizable and need not be unique:**

```python
s2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2['c']            # -> 3
s2[s2.index[-1]]   # -> 3   (last label, found dynamically)
```

Custom labels must match the element count. pandas even *allows* duplicate index
labels (`index=['x','x','y']` is legal, and `s['x']` then returns **both** `x`
rows) — bizarre, and worth avoiding. When in doubt, stick with the default
0-based range.

**Joining Series — `pd.concat`** takes a **list** of objects:

```python
pd.concat([a, b])                     # index -> [0, 1, 0, 1]  (keeps both schemes)
pd.concat([a, b], ignore_index=True)  # index -> [0, 1, 2, 3]  (fresh 0-based range)
```

By default concat **keeps each piece's original index**, so you can end up with
duplicate labels. `ignore_index=True` throws the old labels away and rebuilds a
clean range — a recurring argument you'll see across pandas operations.

---

## 3. DataFrame — labeled 2D data (aligned Series)

A `DataFrame` is the core structure: an entire **table** (2D — rows and columns),
and the usual way you interact with data. The mental model:

- each **column** is a variable/attribute;
- each **row** is a data point / observation;
- a DataFrame is effectively **a collection of Series put side by side** — every
  column is a Series, and they all **share one common index**.

So Series (§2) wasn't a detour: a DataFrame *is* aligned Series. That shared
index is what lets pandas line columns up by row.

**Constructing one** — you call the **constructor** `pd.DataFrame(...)` (a
constructor is the method you call on a type to make an instance of it). You can
build from:

```python
pd.DataFrame([[1, 2], [3, 4]])                    # a list of lists
pd.DataFrame(np.arange(12).reshape(4, 3))         # a NumPy array (RC -> 4 rows, 3 cols)
pd.DataFrame({"city": ["Mil", "StL"], "wins": [95, 93]})   # a dict
```

From a **dict**, keys become column names and values become column data — and
`len(some_dict)` (the number of key/value pairs) is the number of columns you'll
get. From a bare array you get **default integer column names** (`0, 1, 2, …`) and
a default RangeIndex. Data can be **categorical** (text values that fall into
named categories, e.g. baseball divisions) or **numeric** (e.g. air-quality
measurements).

**Three inspectors you'll reach for constantly:**

| Call | Kind | Shows |
|---|---|---|
| `df.shape` | attribute | `(rows, columns)` tuple — same **RC** order as NumPy |
| `df.dtypes` | attribute | the dtype of each column |
| `df.info()` | method | index range, each column's name, non-null count, dtype, memory |

> **Version note — string dtype (`object` → `str`).** Older lab material says a
> text column's dtype shows as **`object`**. That was true on pandas 1.x/2.x. On
> **pandas 3.0**, text columns default to the dedicated **`str`** dtype, so
> `df.dtypes` now prints `str` (not `object`) for string columns, while numeric
> columns still show `int64`/`float64`. Verified here:
> `pd.DataFrame({"txt": ["x"], "num": [1]}).dtypes` → `{'txt': 'str', 'num': 'int64'}`.
> If a quiz keys to the course's `object` answer, give the course's answer, but
> know that on a current install it's `str`.

---

## 4. Working with columns

Columns are the attributes of your data, and most manipulation is column work:
selecting, adding, removing.

**Selecting one column — two notations:**

```python
df["wins"]      # bracket notation — always works, name is a (usually quoted) string
df.wins         # dot notation — shorter, but FAILS on spaces/special characters
```

Dot notation is cleaner but only works when the name is a valid Python
identifier and isn't a DataFrame method name; `df.home team` is a literal
**SyntaxError**, so for `"home team"` you *must* use brackets. `df.columns` lists
all the names (handy when you forget them).

**Single vs. multiple — the bracket rule of thumb:**

```python
df["wins"]              # one column  -> a Series (1D)
df[["city", "wins"]]    # a LIST of names -> a DataFrame (2D)
```

Note the **double brackets** on the second: outer `[]` = "select," inner `[]` =
the list of names. **Single column → Series; list of columns → DataFrame.** (You
can build the name list beforehand and pass it in — cleaner for many columns.)

**Adding columns** is assignment to a new key:

```python
df["wins"] = [95, 93, 62]   # list/array; must match the row count, aligns positionally
df["season"] = 2022         # a SCALAR broadcasts -> same value in every row
```

A list (or NumPy array) must match the number of rows. A **scalar broadcasts** to
fill every row — so `df["season"] = 2022` adds a full column of `2022` and the
shape grows by one column.

**Dropping columns — `df.drop`:**

```python
df.drop(columns=["season"])                 # returns a NEW DataFrame; original UNCHANGED
df.drop(columns=["season"], inplace=True)   # mutates df in place; column truly gone
```

The key gotcha: by default `drop` **returns a new DataFrame and does not touch
the original** — run the original `df` afterward and the column is still there.
Two ways to make it stick:

1. **Reassign** (recommended for reproducibility — keep the raw set, derive a
   processed set): `baseball_dropped = df.drop(columns=["season"])`.
2. **`inplace=True`** — mutate the underlying DataFrame directly. Handy when you
   created a column by mistake and just want it gone.

> **Version note — Copy-on-Write and `.copy()`.** Older notes pair the
> reassign-pattern with `.copy()` (`df.drop(...).copy()`) to "avoid strange
> behavior" — the historical risk that editing a *selection* of another frame
> would either silently write back through a view or raise a
> `SettingWithCopyWarning`. On **pandas 3.0, Copy-on-Write is the default and is
> mandatory**, so that hazard is gone: a filtered subset is independent, and
> assigning to it neither warns nor mutates the original (verified — assigning a
> new column to `df[df["wins"] > 65]` raises no warning and leaves `df`
> untouched). Two practical points: `drop()` already returns a *new* frame, so
> the trailing `.copy()` is redundant there; and `.copy()` is still a fine habit
> when you want to state "this is an independent copy" explicitly — it's just no
> longer load-bearing for correctness.

---

## 5. Filtering rows — the two-stage mask

Filtering means selecting the **subset of rows** that meet a column condition —
like a SQL `WHERE`, or an if-statement per row (True = keep, False = drop). It's
always two stages, and it's the same boolean-mask idea you used on NumPy arrays
in Module 5.

**Stage 1 — build the mask.** A condition on a column returns a **Series of
True/False**, one per row (dtype `bool`):

```python
bb["wins"] > 65        # -> True, True, False, True, ...   (a boolean Series)
```

**Stage 2 — apply it.** Drop the mask into `.loc`, which takes **`[rows,
columns]`** (just like NumPy's RC indexing):

```python
bb.loc[bb["wins"] > 65]                          # all columns of the matching rows
bb.loc[bb["wins"] > 65, ["city", "team", "wins"]]  # ...trimmed to specific columns
bb[bb["wins"] > 65]                              # shorthand when you don't need to pick columns
```

The column part is optional: drop it (or use `:`) for all columns, or pass a
**list** to trim the output. `bb[mask]` is a common shorthand, but reach for
`.loc[mask, cols]` when you want to filter rows *and* choose columns in one shot.
Filtered rows **keep their original index labels** (gaps are normal), so you can
cross-reference back to the source row. Works on string conditions too
(`bb.loc[bb["team"] == "Brewers"]`).

**Combining conditions — element-wise operators, always parenthesized:**

```python
bb.loc[(bb["team"] == "Brewers") | (bb["team"] == "Cardinals")]   # OR  ->  |
bb.loc[(bb["team"] == "Cardinals") & (bb["wins"] > 55)]           # AND ->  &
bb.loc[~(bb["wins"] > 65)]                                        # NOT ->  ~
```

> **Critical operator trap.** Use `&` `|` `~`, **not** Python's `and` / `or` /
> `not` — those operate on whole objects and raise
> `ValueError: The truth value of a Series is ambiguous`. And **wrap each
> condition in its own parentheses**: `&`/`|` bind tighter than `>`/`==`, so
> without parens precedence silently wrecks the result. This is the same rule as
> NumPy masks in Module 5.

**Cleaner "match any of these" — `.isin`:**

```python
bb.loc[bb["team"].isin(["Brewers", "Cardinals"])]   # same as the chained | above
```

`.isin([...])` beats a long chain of `|` conditions when you have several
possible values.

**Partial string matches — the `.str` accessor.** `==` is an *exact* match; for
patterns use `.str` methods (case-sensitive by default):

```python
bb.loc[bb["team"].str.contains("ir")]      # any value containing "ir"  -> Pirates
bb.loc[bb["team"].str.startswith("Brew")]  # values beginning "Brew"    -> Brewers
```

Many more `.str` functions exist — check the pandas docs.

---

## 6. Position vs. label, sorting, and the attribute-vs-method tell

The HW6 work leans on a few more table operations worth pulling together.

**`.iloc` (integer position) vs. `.loc` (label).** Both can select rows; the
difference is *what the number means*:

```python
df.iloc[-1]     # last row by POSITION (this is how you get "the last row")
df.iloc[0]      # first row by position
df.loc[3]       # the row whose index LABEL is 3
```

A single selected row comes back as a **Series**, and its `Name:` is that row's
index label. (Recall from §2 that `.iloc[-1]` is also the right way to get a
Series' last element by position, since `s[-1]` fails.)

**Sorting — `sort_values`:**

```python
df.sort_values("A", ascending=False, ignore_index=True, inplace=True)
```

- `ascending=False` → descending;
- `ignore_index=True` → rebuild the index `0 … n-1` after the sort (same arg as
  concat in §2);
- `inplace=True` → mutate `df` and **return `None`**.

> **`inplace` trap.** Because `inplace=True` returns `None`, never write
> `df = df.sort_values(..., inplace=True)` — you'd overwrite `df` with `None`.
> Either reassign *without* `inplace`, or use `inplace=True` *without* assigning.
> (Same caution for any in-place method.) Note also that after an
> `ignore_index` sort, identical data can sit under a different label — the value
> didn't change, the label did.

**Attribute vs. method — a tell that prevents a lot of errors.** Attributes are
nouns (a stored fact); methods are verbs (an action you call):

| Attributes (no parentheses) | Methods (need parentheses) |
|---|---|
| `.index` `.columns` `.shape` `.dtypes` `.values` `.iloc` `.loc` | `.info()` `.head()` `.drop()` `.sort_values()` `.reset_index()` `.mean()` |

`df.shape` gives the tuple; `df.shape()` is an error. `df.info` shows the method
object; `df.info()` actually runs it.

**A few HW6 traps in one place:**

- `df[["A", "C"]]` needs **double brackets**; `df["A", "C"]` is an error.
- `'and'`/`'or'` in a mask → `ValueError`; use `&` and `|` (and `~` for not).
- Forgetting parentheses around each mask condition → wrong answer or error.
- `reset_index()` *without* `drop=True` shoves the old index in as a new column
  (sometimes you want that — see §7 — sometimes you don't:
  `reset_index(drop=True)` discards it).
- `RangeIndex(start=0, stop=5, step=1)` is just shorthand for labels `0,1,2,3,4`.

---

## 7. Capstone — visualizing the standard error by simulation

This notebook (no paired lecture) fuses the whole module and reconnects to
Module 5's standard-error idea. **Goal:** simulate repeated samples from a uniform
distribution, take each replication's mean, and *watch the spread of those means
shrink as the sample size grows* — i.e. show **why bigger samples give more
stable estimates.**

**Generate data with NumPy, wrap it in pandas:**

```python
rg = np.random.default_rng(2100)        # seeded generator -> reproducible (Module 5)
random_arr = rg.random((25, 7))         # 25 replications (rows) x 7 data points (cols), uniform [0,1)
df = pd.DataFrame(random_arr)           # default integer columns 0..6, RangeIndex
```

**Per-replication mean — `axis` is the crux (Module 5 callback):**

```python
rep_means = df.mean(axis=1)             # average ACROSS columns -> one mean PER ROW
```

`axis=1` **collapses the columns**, leaving one value per row (per replication) —
exactly the "axis = the direction you collapse" rule from Module 5. `axis=0` (the
default) would average *down* each column instead. The result is a **Series**.

**Turn it into a plottable table with an ID column:**

```python
rep_means = pd.DataFrame(rep_means, columns=['rep_mean'])               # Series -> 1-col DataFrame
rep_means = rep_means.reset_index().rename(columns={'index': 'replication'})
```

Here `reset_index()` is used *for* its side effect: on the default RangeIndex it
moves the index into a real column named `index` (and adds a fresh 0-based
index), then `rename(columns={...})` relabels it to `replication` — an explicit
ID column to plot against. (Confirmed: the resulting columns are
`['replication', 'rep_mean']`.)

**Plot straight off the DataFrame:**

```python
rep_means.plot.scatter(x='replication', y='rep_mean')
```

For the small sample (n=7) the 25 means **scatter widely** — observed spread
≈ 0.23 … 0.75 (their standard deviation ≈ 0.12), no real pattern; high
variability.

**Small vs. large — the payoff.** Build a large-sample set and overlay it:

```python
big = pd.DataFrame(rg.random((25, 500))).mean(axis=1)   # 500 points per replication
rep_means_all = pd.DataFrame(big, columns=['rep_mean500']).reset_index() \
                  .rename(columns={'index': 'replication'})
rep_means_all['rep_mean7'] = rep_means['rep_mean']      # add small means by index alignment

fig, ax = plt.subplots()
rep_means_all.plot.scatter(x='replication', y='rep_mean7',   color='blue', ax=ax)
rep_means_all.plot.scatter(x='replication', y='rep_mean500', color='red',  ax=ax)
ax.set_ylabel('Means'); plt.show()
```

Passing the **same `ax`** to both `.plot.scatter` calls overlays them on one set
of axes, and `rep_means_all['rep_mean7'] = rep_means['rep_mean']` lines the two
columns up **by their shared index**.

**The result:** the n=500 means (red) cluster tightly around 0.5 — observed
spread only ≈ 0.47 … 0.52 (standard deviation ≈ 0.014) — while the n=7 means
(blue) scatter broadly. This is the **standard error of the mean** falling as
*n* grows. The math confirms it: a uniform[0,1) population has mean 0.5 and SD
`1/√12 ≈ 0.289`, so the SEM is `0.289/√n` → **≈ 0.109 for n=7** and **≈ 0.013 for
n=500** — right in line with the observed scatters above, and exactly Module 5's
conclusion (`SEM ∝ 1/√n`), now produced by simulation instead of a formula.

---

## One-paragraph recap

pandas is **NumPy plus labels**: a **Series** is `values` (a NumPy ndarray) + an
`index` (labels — default `RangeIndex`, no negative indexing, `.iloc[-1]` for the
last by position), and a **DataFrame** is several aligned Series sharing one
index, where **columns are variables and rows are observations**. Build a frame
with `pd.DataFrame(...)` from lists-of-lists, arrays, or dicts; inspect with
`.shape`/`.dtypes`/`.info()` (text columns are now `str`, not `object`). Column
work is the heart of it: `df["c"]` → a **Series**, `df[["a","b"]]` → a
**DataFrame** (double brackets), add by assignment (a scalar **broadcasts**), and
`drop` returns a **new** frame unless you reassign or pass `inplace=True` (and
under pandas 3.0 Copy-on-Write the old `.copy()` ritual is no longer load-bearing).
**Filter** in two stages — build a boolean mask, then apply it via `df[mask]` or
`df.loc[mask, cols]` — combining conditions with **`&` `|` `~`, each in
parentheses** (never `and`/`or`), and reaching for `.isin([...])` or
`.str.contains/startswith` as needed. Mind **`.iloc` (position) vs `.loc`
(label)**, that **`inplace=True` returns `None`**, and the **attribute (no parens)
vs method (parens)** tell. The capstone fuses it all: a 2D random array →
DataFrame → `.mean(axis=1)` (per-row means, the Module 5 axis rule) →
`reset_index().rename(...)` for an ID column → an overlaid scatter showing the
sample means tighten from a ≈0.23–0.75 spread at n=7 to ≈0.47–0.52 at n=500 —
the standard error shrinking like `1/√n`.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
