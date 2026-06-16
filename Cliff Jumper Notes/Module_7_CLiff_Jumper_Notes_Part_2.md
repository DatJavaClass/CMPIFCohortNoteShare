# Module 7 — Cliff Jumper Notes (Part 2)
### Grouping, Aggregation & the Split-Apply-Combine Pattern

A cross-topic synthesis of the two Module 7 grouping notebooks (*Intro to Grouping &
Aggregation* and *Advanced Grouping & Aggregation*). Every number and behavior below was
verified by execution on **pandas 2.3.3** (Diamond QC'd). Use this as the fast master sheet;
the per-topic `*_CliffNotes.txt` / `.md` files have the full detail.

> **Where Part 1 left off:** Part 1 covered concat, summary statistics, and missing
> values/EDA — how to *describe a whole column or frame at once*. Part 2 is the natural next
> step: **describe each group separately.** Everything you learned about NaN-skipping,
> `ddof=1`, and `.sem()` in Part 1 still applies — it just now runs once per group.

---

## The two notebooks in one line each
| Notebook | Core idea |
|---|---|
| **Intro to Grouping & Aggregation** | `groupby` does **split-apply-combine** in one line: split rows into buckets by a key, reduce each bucket, stitch the answers into one row per group. |
| **Advanced Grouping & Aggregation** | `.agg()` removes the "one function, one column" ceiling: **many functions × many columns in a single call**, plus multi-key grouping and control over missing groups. |

---

## The one idea everything hangs on: Split → Apply → Combine

Grouping aggregation breaks a dataset into groups (rows that share a value for some
attribute — almost always a **categorical** variable), applies a function to each group, then
combines the results. The output has a **different shape** than the input: **one row per
group**, not one row per observation.

```python
df.groupby("E")["B"].mean()
#   └───┬────┘ └┬┘  └──┬──┘
#     split   select  reduce
```

| Slot | Code | Reads as |
|------|------|----------|
| **split** | `groupby("E")` | "one bucket per unique value of E" — takes the **column name**, not a value |
| **select** | `["B"]` | "in each bucket, look at column B" — column **selection**, not sorting |
| **reduce** | `.mean()` | "collapse **each** bucket to a mean" — keep the `()`; `.mean` alone is the uncalled function |

One line to memorize: **split by key → pick the column → collapse each bucket.**

There's an implied loop you never write:

```python
results = {}
for key in df["E"].unique():           # SPLIT
    bucket = df[df["E"] == key]
    results[key] = bucket["B"].mean()  # APPLY — reduce this bucket's slice only
# pandas then COMBINEs into a Series indexed by the keys
```

You describe the loop declaratively; pandas runs it in fast C. A hand-written `for` loop over
groups gives the same answer but runs far slower — same logic, wrong altitude.

**Syntax skeleton:** `df.groupby(<grouping column>)[<target column>].<function>()`
The target column is optional — drop it and the function applies across *all* other columns;
pass a list `[['B','C']]` for several. Dot notation works for the target too:
`titanic.groupby('pclass').survived.mean()`.

---

## Five threads that tie the two notebooks together

### 1. Group by the column NAME — never a value inside it
`groupby()` always takes the **name of the key column**. You never feed it one specific value.

```python
df.groupby("A")["B"].mean()    # ✅ "A" is a column → one bucket per id
df.groupby("id1")["B"].mean()  # ❌ KeyError: "id1" is a VALUE inside column A, not a column
```

If you only want **one** value (just `id1`), that's not a groupby — it's a **filter**:

```python
df[df["A"] == "id1"]["B"].mean()   # one number, only the id1 rows
```

> **Fork in the road:** `groupby` = "do it for *every* value at once" (many answers).
> `df[df[col] == val]` = "do it for *one* value" (single answer).

### 2. Each bucket sees only its OWN slice — never the whole column
The reduce runs **per bucket**, on that bucket's slice — not the whole column, not one grand
total. With our demo frame (`E` is the key, `B` the data):

```
low bucket → B = 10, 12, 7, NaN → mean 9.666667   (= 29/3; the NaN is skipped)
mid bucket → B = NaN, 9, 15      → mean 12.0       (mid never sees 10 or 7)
```

If every bucket averaged the *entire* column, every output row would be identical — which
makes groupby pointless. The word to burn in is **"each."**

### 3. NaN gets handled in TWO independent places — don't conflate them
This is the single biggest source of silent wrong answers, and it ties straight back to
Part 1's "stats skip NaN by default."

- **`dropna` controls the KEY column** (the thing you group on). It decides whether rows whose
  *grouping value* is missing get a bucket at all.
- **The reduce (`.mean()`, `.sum()`, `.count()`, …) skips missing VALUES** in the data column
  on its own — exactly as in Part 1 — regardless of `dropna`.

| `dropna` setting | Effect on rows whose **key** is missing (`E = None`) |
|---|---|
| `dropna=True` **(the silent default)** | the missing-key group is **deleted entirely** — those rows vanish from the result |
| `dropna=False` | `NaN` becomes its **own bucket** and gets a row like any other key |

```python
df.groupby("E")["B"].mean()                # NaN-key group silently dropped
df.groupby("E", dropna=False)["B"].mean()  # keeps it → adds  NaN  11.0
```

> **The danger is the default direction.** A plain `groupby` can quietly exclude rows and
> never tell you. Habit: *when a key column might contain NaN, ask whether the default just
> ate some rows.* In the Titanic data, `groupby('embark_town', dropna=False)` surfaces a
> survival rate for the 2 passengers whose boarding port was never recorded.

### 4. `size` vs `count` — bodies in the room vs non-null values
| Method | Counts | Cares about the selected column? |
|--------|--------|----------------------------------|
| `.size()` | **rows in the group** (NaNs included) | **No** — a property of the *group*; `["B"].size()` == `["C"].size()` |
| `.count()` | **non-NaN values in the chosen column** | **Yes** — point it at another column, get another answer |

```
low → size  = 4   (rows: 10, 12, 7, NaN)
low → count = 3   (the NaN is skipped)
```

**Handy identity:** `size − count` = number of missing values in that column for that group
(here `4 − 3 = 1`). `size` is immune to NaN-skipping because it counts *bodies in the room*,
not values.

### 5. The output is sorted by KEY, and the index is rebuilt from the keys
- **Sort:** by default groupby sorts the **group labels** in the output — *not* your original
  frame, and *not* by the computed values. That's why `low (9.67)` sits above `mid (12.0)`:
  sorted alphabetically by key (`low` < `mid`), not by number. Turn it off with
  `groupby("E", sort=False)`.
- **Combine ≠ concatenate:** the keys become the **index** and the reduced values become the
  data. The original rows are gone, replaced by one summary row per group. Group by several
  columns and the index becomes a **multi-index** (see below).

---

## `.agg()` / `.aggregate()` — build a whole report in one pass

Plain `.mean()` is one function on one column. `.agg()` (alias `.aggregate()`) removes that
ceiling: **multiple functions on multiple columns, one line.** Functions are passed as
**strings** (`'mean'`, `'std'`, `'sem'`, `'count'`, `'size'`, `'nunique'`, …) or as your own
callables.

> The bare `df.groupby("E").agg({'B': 'mean'})` is *deliberately underwhelming* — it equals
> `groupby("E")['B'].mean()`. The power shows up only with the two grammars below.

**List grammar** — same functions on every selected column (a symmetric grid):
```python
df.groupby("E", dropna=False)[['B', 'C']].agg(['mean', 'std'])
# → mean of B, std of B, mean of C, std of C
#        B                   C
#     mean       std      mean       std
# low  9.666667  2.516611  1.133333  0.115470
# mid 12.000000  4.242641  1.133333  0.251661
# NaN 11.000000       NaN  1.300000       NaN     ← n=1 → std is NaN (sample stat needs n≥2)
```

**Named-aggregation grammar** — a different function per column, with custom output names
(full control). Pattern: `output_name=('column', 'function')`:
```python
(df.groupby("E", dropna=False)
   .aggregate(
       B_avg=('B', 'mean'),
       B_sem=('B', 'sem'),
       B_numrows=('B', 'size'),
       B_nonmissing=('B', 'count'),
       A_nunique=('A', 'nunique'),
   )
   .reset_index())
#      E      B_avg     B_sem  B_numrows  B_nonmissing  A_nunique
#  0  low  9.666667  1.452966          4             3          3
#  1  mid 12.000000  3.000000          3             2          3
#  2  NaN 11.000000       NaN          1             1          1
```

Note the Part-1 carryover: `.sem()` is `std / √n` with **ddof=1**, where **n is the non-null
count, not `size`** (low group: `2.516611 / √3 = 1.452966`, using count = 3, not size = 4). So
the single-row `NaN` group returns `NaN` for both `std` and `sem` — you can't get a sample
spread from *n = 1*.

---

## Grouping by multiple columns

```python
df.groupby(["E", "F"], dropna=False)["B"].mean()
```
- Makes one bucket per **combination** of E and F values (only combinations that actually
  occur — e.g. `low/x`, `low/z`, `mid/y`, `NaN/x`).
- Returns a **multi-index** (2-level) Series.
- This course avoids multi-index, so flatten it back to a normal DataFrame with
  **`.reset_index()`**.

---

## Real-world anchor: survival on the Titanic (intro notebook)
Loaded from seaborn (`sns.load_dataset('titanic')`): **891 rows × 15 columns**.

- **Overall survival, no grouping:** `titanic['survived'].mean()` ≈ **0.384** (~38%). Because
  `survived` is coded 0/1, *the mean **is** the survival rate.* The high spread is what
  motivates breaking it down by group.
- **Group sizes (sanity check):** `groupby('pclass')['survived'].count()` → **1st = 216,
  2nd = 184, 3rd = 491.** Most passengers rode in 3rd class; the **fewest were in 2nd class**
  (184 < 216). *(`value_counts()` gives the same counts faster, but the groupby shows the
  pattern.)*
- **Survival rate by class:** `groupby('pclass')['survived'].mean()` → **1st ≈ 0.63,
  2nd ≈ 0.47, 3rd ≈ 0.24.** Clear stratification — survival tracked with class/ticket price.
- **Keeping missing keys:** `groupby('embark_town', dropna=False)['survived'].mean()` adds a
  `NaN` row for the 2 unrecorded-boarding passengers (both survived → rate 1.0), a group a
  plain groupby would have hidden.

---

## Method cheat-sheet (what to reach for)
| Goal | Call | Notes |
|---|---|---|
| One stat, one column, per group | `df.groupby(key)[col].mean()` | swap in `sum/min/max/std/median/count/nunique/size` |
| Keep missing-key rows as their own group | `groupby(key, dropna=False)` | default `dropna=True` silently drops them |
| Rows per group (NaNs counted) | `.size()` | property of the group; column-independent |
| Non-null values per group | `.count()` | depends on the selected column |
| Missing values per group, per column | `.size() − .count()` | the gap is the NaN count |
| Same functions on several columns | `df.groupby(key)[[c1,c2]].agg(['mean','std'])` | symmetric grid (list grammar) |
| Different function per column + names | `.agg(name=('col','func'), …)` | named-aggregation grammar |
| Standard error of the mean | `('col','sem')` | std / √(non-null count), ddof=1 → NaN when n=1 |
| Distinct values per group | `('col','nunique')` | drops NaN unless the data itself is kept |
| Group by several keys | `groupby([k1,k2])` | returns a multi-index Series |
| Flatten a multi-index result | `.reset_index()` | turns index levels back into columns |
| Don't sort the group labels | `groupby(key, sort=False)` | default sorts labels in the output |

---

## Exam-trap checklist (the things that bite)
- **`groupby("E")` takes the column NAME.** `groupby("id1")` (a value) → **KeyError**. One
  value is a **filter**, not a groupby.
- **Each bucket reduces its OWN slice** — never the whole column. Burn in the word "each."
- **`dropna` is about the KEY; the reduce skips missing VALUES on its own.** Two separate NaN
  handlers — don't conflate them.
- **`dropna=True` is the silent default** — a plain `groupby` can drop missing-key rows and
  never tell you. Use `dropna=False` to keep them; pair with `.reset_index()` when grouping by
  multiple keys.
- **`size` = rows in group (NaNs counted); `count` = non-null values in the chosen column.**
- **A single-row group → `NaN` for `std`/`sem`** (sample stats need n ≥ 2; ddof=1).
- **Output is sorted by KEY label, not by value** → `low` (9.67) sits above `mid` (12.0).
- **`.agg({'col':'func'})` alone is just the plain reduce** — the power is the list grammar
  (symmetric) and the named grammar (`name=('col','func')`, per-column control).
- **Grouping by 2+ columns returns a multi-index** → flatten with `.reset_index()`.
- **A Jupyter cell displays only its LAST expression.** End a cell with an assignment
  (`summary = df.groupby(...)`) and you see **nothing** — no error, no table. Put the variable
  name on its own last line, or use `display(summary)`.

---

## Worked anchors (memorize one example per concept)
- **Per-bucket mean drops NaN:** `low = [10, 12, 7, NaN] → 9.666667` (29/3), `mid = [NaN, 9,
  15] → 12.0`.
- **Missing-key group:** `groupby("E", dropna=False)` adds `NaN → 11.0` (the lone `E = None`
  row, B = 11).
- **size vs count:** `low → size 4, count 3` → one missing B in the low group.
- **n=1 → NaN spread:** the `NaN`-key group (one row) gives `std = sem = NaN`.
- **Multi-key:** `groupby(["E","F"], dropna=False)["B"].mean()` → only the combinations that
  occur: `low/x = 9.67`, `low/z = NaN`, `mid/y = 12.0`, `NaN/x = 11.0`.
- **Practice Q gotcha:** `groupby(['E','G'])` *without* `dropna=False` silently drops the
  `E = None` row, so it never appears in the result.
- **Titanic:** `891 × 15`; overall survival ≈ 0.384; class counts `1st 216 / 2nd 184 /
  3rd 491`; survival `1st 0.63 / 2nd 0.47 / 3rd 0.24`.

---

### TL;DR
**Split → apply → combine.** `groupby` takes a **column name**; `[]` **selects** a column;
each bucket reduces its **own slice** and the result is **one row per group**, indexed and
sorted by the keys. The default `dropna=True` **silently drops missing-key rows** — use
`dropna=False` to keep them, and remember that's a *separate* NaN handler from the reduce,
which skips missing values on its own. `size` = rows in group; `count` = non-null values;
their difference = missing count. `.agg()` runs **many functions × many columns in one line**
(list grammar = symmetric, named grammar = per-column control); single-row groups yield `NaN`
spread (ddof=1). Group by several keys → multi-index → `.reset_index()`. And end a cell with
the variable name or `display()`, or you'll stare at nothing.

---
*Part 2 of the Module 7 Cliff Jumper series. Covers: Intro to Grouping & Aggregation
(split-apply-combine, the Titanic walkthrough) · Advanced Grouping & Aggregation (`size`/`count`,
`.agg()` list & named grammars, multi-key grouping, `dropna` control). All figures
execution-verified on pandas 2.3.3 under Diamond QC.*
