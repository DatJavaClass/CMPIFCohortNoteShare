# Module 7 — Cliff Jumper Notes (Part 1)
### Pandas: Concatenation, Summary Statistics, Missing Values & EDA

A cross-topic synthesis of the four Module 7 notebooks. Every number and behavior
below was verified by execution on **pandas 2.3.3** (Diamond QC'd). Use this as the
fast master sheet; the per-topic `*_CliffNotes.txt` files have the full detail.

---

## The four notebooks in one line each
| Notebook | Core idea |
|---|---|
| **Pandas Concat** | Glue DataFrames together: **union the labels, fill every gap with NaN.** |
| **Summary (Series)** | Summarize ONE column: distribution stats for numbers, count stats for categories. |
| **Summary (DataFrame)** | Same stats across a whole frame; handle mixed types + custom functions. |
| **Missing Values & EDA** | Find/quantify/drop NaN, then run the "first look" on any dataset. |

---

## Five threads that tie the whole module together

### 1. NaN is the connective tissue
NaN shows up in every notebook, and the same facts keep paying off:
- **Concat** creates NaN wherever a label exists in one frame but not another (union-and-fill).
- **NaN is a float**, so the moment it lands in an integer column the whole column
  upcasts to `float64` (`10` → `10.0`). A column holds exactly one dtype.
- **Stats ignore NaN by default**: `mean`, `std`, `sem`, `value_counts`, `nunique`
  all skip it. That's *why* `df_2.mean(axis=1)` gives Lando Norris `3.0` (= 6/2, the
  NaN dropped — **not** 6/3).
- **EDA quantifies it**: `df.isna().sum()` = count missing; `df.isna().mean()` =
  proportion missing (the mean of a 0/1 Boolean column IS its True-rate).

### 2. `axis` means "the dimension you collapse"
- `axis=0` (default) → collapse **rows** → one answer **per column** (`df.mean()`).
- `axis=1` → collapse **columns** → one answer **per row** (`df.mean(axis=1)`,
  `df.apply(fn, axis=1)`, `df.isna().sum(axis=1)`).
- Concat uses the same axis flag: `axis=0` stacks vertically (union columns),
  `axis=1` stacks horizontally (union the index).
- ⚠️ The DataFrame-summary notebook's wording *"a value returned for each column,
  pass axis=1"* is misleading — `axis=1` returns a value **per row**.

### 3. `ddof=1` — pandas' sample-statistics default
- Pandas `.std()` and `.sem()` use **ddof=1** (divide by *n−1*, the **sample** stat).
  NumPy's `np.std()` defaults to **ddof=0** (divide by *n*, the **population** stat).
  Same data, different number: age std = `0.8165` (pandas) vs `0.7454` (numpy default).
- Consequence with thin data: a column with only **one** non-NaN value (`score_3`)
  returns **NaN** for std/sem — you can't get a sample spread from *n=1* (n−1 = 0).

### 4. "Numeric" is not one definition — mind the booleans
- `df.mean(numeric_only=True)` **includes bool** → `pass` (True/True/True/False)
  averages to **0.75**. Without `numeric_only`, a string column makes `mean()/std()/sem()`
  raise a **TypeError**.
- `df.select_dtypes('number')` **excludes bool** → returns only the int/float columns.
- 🔑 So the two "numeric" filters **disagree on booleans**. Don't treat them as
  interchangeable.

### 5. Series vs DataFrame is the same toolkit, one dimension apart
- A DataFrame column **is** a Series; `df["col"]` returns a Series (double brackets
  `df[["col"]]` returns a one-column DataFrame).
- Single-column methods (`.mean`, `.nunique`, `.value_counts`, `.map`, `.isna`)
  scale up to the whole frame, returning a Series indexed by column name.
- Pick the stat to match the dtype: **numbers** → `mean/min/max/std` (center+spread);
  **categories** → `nunique` (how many), `unique` (which), `value_counts` (how often).

---

## Method cheat-sheet (what to reach for)

| Goal | Call | Notes |
|---|---|---|
| Stack rows / columns | `pd.concat([...], axis=0/1)` | `ignore_index=True` to renumber |
| Average / spread of a column | `.mean()`, `.std()` | std is sample (ddof=1) |
| Standard error of mean | `.sem()` | = std / √n |
| Distinct count / values | `.nunique()` / `.unique()` | nunique drops NaN unless `dropna=False` |
| Total rows in a column | `.size` | counts NaN slots too |
| Frequency per category | `.value_counts()` | sorted high→low; drops NaN by default |
| Stats on mixed-type frame | `df.mean(numeric_only=True)` | includes bool; skips strings |
| Keep only numeric columns | `df.select_dtypes('number')` | **excludes bool** |
| One-to-one element transform | `Series.map(fn / dict / Series)` | can't see other columns |
| Row/column-wide custom calc | `df.apply(fn, axis=0/1)` | can combine multiple values |
| Flag missing | `.isna()` / `.isnull()` | identical aliases; True = missing |
| Count / proportion missing | `.isna().sum()` / `.isna().mean()` | per column |
| Drop missing rows (complete cases) | `df.dropna()` | new frame; `inplace=True` to mutate |
| Drop missing columns | `df.dropna(axis=1)` | |
| Drop only if particular cols missing | `df.dropna(subset=[...])` | |
| Drop only fully-empty rows/cols | `df.dropna(how='all')` | default is `how='any'` |
| Quick frame overview | `df.info()` | rows, dtypes, **non-null** counts, memory |
| Column labels / dtypes | `df.columns(.tolist())`, `df.dtypes` | |
| Numeric distribution | `df.describe()` | count/mean/std/min/quartiles/max |
| Categorical summary | `df.describe(include='object')` | count/unique/top/freq |

---

## Exam-trap checklist (the things that bite)
- **`10` → `10.0`** after concat: a NaN fill upcast the int column to float. Per-column dtype.
- **`object` dtype** just means a text/string column — a *separate* issue from the NaN→float upcast.
- **Repeating index** after vertical concat unless `ignore_index=True`.
- **pandas std ≠ numpy std** (ddof=1 vs ddof=0). Single-value column → NaN std/sem.
- **bool counts as numeric** for `mean()` but **not** for `select_dtypes('number')`.
- **`axis=1` = per row** (collapse columns), not per column.
- **`.map` is element-wise on one column/index** and can't see sibling columns; **`.apply`** can.
- **`nunique()` drops NaN** by default; `dropna=False` counts NaN as its own value.
- **`info()` reports non-null counts** → use `isna().sum()` for a direct missing count.
- **`describe()` `count` excludes NaN**; on an all-unique categorical, `top` is just the first value (arbitrary tie-break).
- **`dropna(axis=1)`** can wipe every column if each has at least one NaN; **`how='all'`** rarely drops anything.

---

## Worked anchors (memorize one example per concept)
- **Concat NaN island**: `pd.concat([df_c, df_a])` where only `df_c` has `extra` →
  `df_a`'s rows get `NaN` in `extra` (and the column becomes `float64`).
- **Row-wise mean drops NaN**: Lando `[1, 5, NaN]` → `mean(axis=1) = 3.0`.
- **DNF count (practice Q)**: `df_3.isna().sum(axis=1)` → Max 0, Lewis 1, Fernando 1, Lando 2.
- **Titanic first look**: `891 × 15`; missing `age=177`, `deck=688`, `embarked=2`;
  `pclass` value_counts `3rd=491 / 1st=216 / 2nd=184`; mean age ≈ `29.70`, max fare `512.33`.

---
*Part 1 of the Module 7 Cliff Jumper series. Covers: Concat · Summary (Series) ·
Summary (DataFrame) · Missing Values & EDA. All figures execution-verified under Diamond QC.*

---

*Authored and directed by **Victor Sverdlin (DatJavaClass)**: conceived, structured, formatted, fact-checked, and edited by Victor Sverdlin, with assistance by Claude.*
