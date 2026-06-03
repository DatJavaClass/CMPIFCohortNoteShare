# Module 5 — Cliff Jumper Notes

*One continuous lesson stitched from the seven Module 5 cliff notes (Representing
Data; 1D arrays; 2D arrays; Reshaping/Flattening/Transposing; Conditional
Filtering & Summary Stats; Random Number Generation; Simulating the Standard
Error). NumPy is the through-line, and one idea — **rows-first, columns-second
(RC), and "axis" = which direction an operation runs** — recurs in nearly every
section. Learn that spine and the rest hangs off it. The note ends with the
simulation that uses all of it at once.*

> All code here was executed against NumPy 2.4.6 on Windows; outputs shown are
> real. Where current NumPy differs from what an older lab printed, it's flagged.

---

## 1. Why NumPy: data as a numeric table

Data usually lives as a **table**: columns are attributes/fields, rows are
observations, each cell is one attribute's value for one row. (In raw Python
that's a list of dictionaries; the Parthenon as `{year_built: 438 BCE, …}`.) How
you structure it — combining columns, deriving new ones — is the analyst's
design choice and directly determines what analysis is possible.

Two Python tools hold tabular data: **NumPy arrays** (this module) and **pandas
DataFrames** (later, built on NumPy, and they keep headers/labels). NumPy
deliberately strips headers and metadata and holds the **numeric core** as an
`ndarray` (n-dimensional array), which is what enables fast math and
linear-algebra operations. Two attributes anchor everything:

- **`ndim`** — number of dimensions: 1D = a single list of numbers; 2D = a
  table (rows × columns); 3D+ exists (e.g. tables over time) but is out of scope.
- **`shape`** — `(rows, columns)`, and the axes match that order: **axis 0 =
  rows, axis 1 = columns** (zero-indexed). A 2-row, 3-column array is `(2, 3)`.

That `axis 0 = down the rows, axis 1 = across the columns` fact is the single
most reused idea in the module. Hold onto it.

---

## 2. Building 1D arrays

```python
import numpy as np          # the np alias is universal; you'll type it constantly
```

| Constructor | Result |
|---|---|
| `np.arange(6)` | `[0 1 2 3 4 5]` — like `range()` but returns an array directly (stop **exclusive**) |
| `np.arange(0, 10, 2)` | `[0 2 4 6 8]` — third arg is the step; stop still exclusive |
| `np.arange(0, 11, 2)` | includes 10 — push the stop past it |
| `np.array([...])` | from an existing list |
| `np.zeros(5)`, `np.ones(5)` | five 0s / five 1s |
| `np.linspace(0, 1, 50)` | 50 evenly spaced points, endpoint **inclusive** (you set the *count*, not the step) |

`np.arange` differs from Python's `range()`, which returns a lazy object you'd
wrap in `list()`; `arange` hands you the numbers immediately. Its element
`.dtype` **is** `int64` here — still just an integer (whether the repr actually
prints `dtype=` is a separate display rule; see §8).

**The three attributes** (called on the array itself): `arr.ndim` → `1`;
`arr.shape` → a **one-element tuple** like `(6,)`; `arr.size` → element count
(`6`), an alternative to `len()`.

**The real departure from lists — element-wise behavior:**

```python
[1, 2] + [1, 2]                       # lists CONCATENATE -> [1, 2, 1, 2]
np.array([1, 2]) + np.array([1, 2])   # arrays ADD        -> [2, 4]

np.array([1, 2, 3]) == np.array([1, 9, 3])   # -> [ True False  True]  (a bool ARRAY)
np.array_equal(a, b)                          # -> a single True/False for the whole array
```

`==` compares element by element and returns a **boolean array**, not one
True/False. For a single whole-array answer, use `np.array_equal`.

---

## 3. Building 2D arrays (matrices)

The most common build is from a **list of lists**:

```python
np.array([[1, 2, 3],
          [4, 5, 6]])      # a 2x3 matrix; whitespace/line breaks inside [] are ignored
```

A raw Python list of lists isn't special — `len([[1,2,3],[4,5,6]])` is just `2`
(two elements that happen to be lists). **NumPy** is what supplies matrix
structure. **Rows must be equal length**; ragged inner lists (`[1,2,3]` and
`[1,2]`) raise a `ValueError`.

- `shape` → `(rows, columns)` — mnemonic **RC**. `size` → total cells (a 4×3 has
  size 12).
- **From 1D arrays:** `np.concatenate([a, b])` stays **1D** (a longer flat list);
  `np.vstack([a, b])` stacks **vertically into 2D** — two 5-element arrays give
  shape `(2, 5)`. The arrays go **inside a list/tuple** as one argument; omitting
  that wrapper errors. `vstack` needs the **same number of columns** (rows may
  differ): a `(2,2)` over a `(3,2)` gives `(5,2)`.
- **Directly:** `np.zeros((2, 2))`, `np.ones((3, 2))` — pass the **shape as a
  tuple** (note the double parentheses). A single number gives a 1D array instead.

**Indexing is RC** — `arr[row, column]`, a single bracket with a comma (different
from plain Python). `:` means *all of that axis*:

```python
arr[0, :]     # first row, all columns
arr[-1, :]    # last row
arr[1, 2]     # single cell at row 1, column 2
arr[:2, :]    # first two rows
arr[:3, 1:]   # first three rows, columns from index 1 on (drops column 0)
```

---

## 4. Reshaping, transposing, flattening

All of these change an array's **shape** without changing its `size`. The key
distinction: **reshape preserves reading order; transpose does not.**

**Reshape** re-fences a flat stream into a grid:

```python
arr.reshape(rows, cols)     # rows * cols MUST equal .size, else ValueError
arr.reshape(2, -1)          # pass -1 for ONE dimension; NumPy solves it (12 -> 6 cols)
```

Only one dimension can be `-1` (NumPy solves for one unknown, not two). Reshape
is a **read, not a mutate** — it *returns* a new array; you must capture it
(`arr = arr.reshape(3, 5)`), or the result is thrown away.

**Transpose (`.T`)** flips rows and columns: shape `(rows, cols)` → `(cols,
rows)`. Every element keeps its value but its coordinates **swap**: position
`[row, col]` → `[col, row]`. So a whole row "stands up" into a column. It's a
**fold, not a spin** — the main diagonal (`[0,0], [1,1], …`) is a hinge that
stays put while the two triangular halves fold across it (not a rotation about
one corner).

Watch the shape change — this is where the source notes can mislead, so here it
is run for real. Transposing a **4×3** gives a **3×4** (not another 4×3):

```python
a = np.arange(12).reshape(4, 3)     # shape (4, 3)
#  [[ 0  1  2]
#   [ 3  4  5]
#   [ 6  7  8]
#   [ 9 10 11]]
a.T                                  # shape (3, 4)  <- rows and cols swapped
#  [[ 0  3  6  9]
#   [ 1  4  7 10]
#   [ 2  5  8 11]]
```

Contrast with **reshape**, which keeps the reading order `0,1,2,3,…` intact and
just redraws the fences. Reshape and transpose are different tools — a
`reshape(1, n)` followed by `reshape(-1, 1)` turns a single row into a single
column and *looks* like a transpose, but it's reshape doing it; don't conflate
them. `.T` is an **attribute** (no parentheses) and a read (original untouched).

**Flattening 2D back to 1D:**

- `.ravel()` → always returns a flat 1D array, **but it's a VIEW** — edits can
  hit the original. Use `.ravel().copy()` for an independent flat array.
- `.squeeze()` → only removes dimensions of **size 1** (e.g. `(1, 12)` → `(12,)`).
  It is **not** a general flattener: `.squeeze()` on a `(3, 4)` (no size-1 axis)
  returns it **unchanged**. For a guaranteed flat result, use `.ravel()`.

Traps: a plain Python list has no `.reshape`/`.T`/`.squeeze` (you'll get
`AttributeError: 'list' object has no attribute 'reshape'`) — box it first with
`np.array(...)`. And `np.array(15)` boxes the single value 15 as a 0-D array; to
*count* 0–14 use `np.arange(15)`.

---

## 5. The views-vs-copies trap (applies everywhere above)

A plain slice is a **VIEW** — a window onto the *same* memory. Mutating the
slice mutates the original:

```python
sub = arr[0:2, 0:2]    # VIEW
sub[:] = 999           # this ALSO changes arr   <- side effect

sub = arr[0:2, 0:2].copy()   # independent data
sub[:] = 999                 # arr stays untouched
```

Mental model: a **view** is your hand resting on part of the same box (poke it,
the original changes); a **copy** is a second box built independently. Ask: *did
a second box get built (`.copy()`), or am I holding part of the only box I had
(plain slice)?* (Note `.ravel()` above is a view for the same reason.) And
re-assigning `arr = np.array([...])` doesn't *repair* a damaged array — it builds
a new object and moves the **name** onto it; the variable name is just a
nameplate.

---

## 6. Conditional filtering — the two-stage move

Filtering is always two stages:

1. **Build a mask** — a boolean test, one `True`/`False` per item on an axis.
2. **Apply the mask** — drop it into the bracket; it returns the **values** you
   keep. (If your final output is still booleans, you forgot to apply it.)

**Where the mask goes decides what gets filtered** — and the `:` always sits in
the axis you are *not* filtering:

```python
arr[arr < 5]                 # filter across ALL elements -> a FLATTENED 1D array
arr[arr[:, 0] > 5, :]        # keep ROWS where column 0 > 5 (mask in the ROW slot)
arr[:, arr[0, :] > 5]        # keep COLUMNS where row 0 > 5 (mask in the COLUMN slot)
arr[:, arr[0, :] > arr[1, :]]  # keep columns where row 0 beats row 1, element-wise
```

The trap in "filter rows by a condition on one column": the column is only
*where you look*; you still keep **whole rows**. The mask's length must match the
axis it filters.

---

## 7. Summary statistics — `axis` is the direction you COLLAPSE

Here is the one genuinely counterintuitive point in the module, and it's the
payoff of the RC spine. For a reduction like `.mean()`, **`axis` is not a
selector — it's which way the calculation crushes the grid.** The result keeps
the axis you did *not* collapse:

```python
arr.mean(axis=0)      # collapse DOWN  (kill rows) -> one mean PER COLUMN
arr.mean(axis=1)      # collapse ACROSS (kill cols) -> one mean PER ROW
arr.mean(axis=None)   # collapse everything -> one single number
```

So **"the mean of each row" is `axis=1`**, not `axis=0` — to get a per-row answer
you sweep *across the columns*. Safety check: a 2-row array should give you 2
row-means → that's `axis=1` (`axis=0` would give one value per column instead).
`.std`, `.sum`, and friends all take the same `axis` argument. (Related: `np.sort`
returns a new sorted array while `arr.sort()` sorts in place; `axis=0` sorts each
column, `axis=1` each row.)

---

## 8. Random number generation

Instead of calling `np.random.something` repeatedly, build **one generator
object** and reuse it:

```python
rg = np.random.default_rng(seed=2100)   # a modern Generator; seed -> reproducible
```

| Call | Produces | Boundary |
|---|---|---|
| `rg.integers(low, high, size=...)` | random integers | `high` is **exclusive** → `low … high-1` |
| `rg.random(shape)` | floats in **[0, 1)** | uniform |
| `rg.normal(mean, std, shape)` | Gaussian floats | you give mean, std, shape |

**Boundary gotcha:** `integers(50, 100)` gives **50–99**, not 50–100. To include
100, use `integers(50, 101)`.

**The generator is stateful — it advances on every draw.** This is the core of
the lab's Practice Exercise 1, where the intuitive plan fails:

```python
# WRONG idea: one generator, draw twice, "the seed makes them match"
rand = np.random.default_rng(seed=2101)
draw1 = rand.random((3, 4))     # sequence item #1
draw2 = rand.random((3, 4))     # sequence item #2 -> DIFFERENT (gen moved forward)
# draw1 == draw2  ->  all False
```

A seed doesn't freeze the output to one value; it fixes the **starting point of a
sequence**. Reproducibility means **two separate generators, seeded the same,
each on its first draw**:

```python
genA = np.random.default_rng(seed=2101)
genB = np.random.default_rng(seed=2101)
genA.random((3, 4)) == genB.random((3, 4))    # -> all True (same seed, same first item)
```

(To "reset," re-create the generator with the same seed. And as in §2, `==` gives
a grid of booleans — collapse it with `np.array_equal(A, B)` or `(A == B).all()`,
never compare against `True`.)

### A note on what the notebook *prints* (and a version caveat)

Two separate mechanisms decide notebook output. **(A)** Jupyter auto-displays a
cell's last line *only if it's a bare expression* — an assignment (`arr = …`)
shows nothing; a bare `arr` on the last line displays. **(B)** What the text
*looks like* comes from the object's **`__repr__`** method (every class provides
one): `repr(arr)` is what the bare-line auto-display uses, `str(arr)` is what
`print()` uses.

The durable rule: **an array's `__repr__` shows `dtype=` only when the dtype is
NOT the default.** Verified on NumPy 2.4.6:

```python
repr(np.array([1, 2, 3], dtype=np.int32))   # array([1, 2, 3], dtype=int32)   <- shown (non-default)
repr(np.array([1., 2.], dtype=np.float32))  # array([1., 2.], dtype=float32)  <- shown (non-default)
repr(np.array([1, 2, 3]))                   # array([1, 2, 3])                <- hidden (default int)
repr(np.random.default_rng(0).random(3))    # array([0.63…, 0.26…, 0.04…])    <- hidden (float64 default)
```

> **Version caveat (the lab's example has aged):** older lab material says an
> integer array prints `dtype=int64` "because the default int is `int32` on
> Windows." That was true on **NumPy 1.x**, where Windows defaulted to `int32`,
> making `int64` non-default and therefore shown. On **NumPy 2.x** the default
> integer on Windows is now `int64`, so `dtype=int64` is **hidden** — the example
> no longer triggers on a current machine. The *rule* (non-default dtypes are
> shown) is what's durable; the specific `int64` case is version-dependent. If a
> quiz keys to the course's `int32`/`int64` example, answer the course's way, but
> know it depends on the NumPy version.

---

## 9. Capstone — simulating the standard error of the mean

This final lab uses almost everything above at once, and it reconnects to
Module 4's statistics. The **standard error of the mean (SEM)** is the standard
deviation of repeated sample means. Rather than resample infinitely, simulate it:

```python
rg = np.random.default_rng(2100)
num_reps, num_samples = 5000, 7

# 2D array: 5000 replications (rows) x 7 sample values (columns)
sample_means = rg.random((num_reps, num_samples)).mean(axis=1)   # axis=1 -> per-ROW mean
sem = sample_means.std(ddof=1)                                    # ddof=1 -> unbiased (Module 4)
# sem ~ 0.1096   (whole pipeline as one line below)

sem = rg.random((num_reps, num_samples)).mean(axis=1).std(ddof=1)
```

Every piece is a callback: a **2D array** shaped `(replications, sample_size)`
(§3); **`axis=1`** to take a mean *per row / per sample* (§7, the counterintuitive
one); **`ddof=1`** for the unbiased standard deviation (Module 4's `n−1`); and the
**generator** (§8). The simulated `sem ≈ 0.1096` for `n=7` matches the analytic
expectation: a uniform[0,1] has SD `1/√12 ≈ 0.289`, so SEM `= 0.289/√7 ≈ 0.109`.

**Effect of sample size** — sweep `n` and watch SEM fall:

```python
sample_sizes = 10 * np.arange(1, 11)          # 10, 20, …, 100  (arange from §2)
sems = [rg.random((num_reps, n)).mean(axis=1).std(ddof=1) for n in sample_sizes]
```

Plotting `sems` against `sample_sizes` gives a downward `1/√n` curve: **larger
samples → smaller standard error → a more confident estimate of the mean** —
exactly Module 4's conclusion, now produced by simulation instead of a formula.

---

## One-paragraph recap

NumPy holds tabular data as a numeric `ndarray` whose two anchors are `ndim` and
`shape = (rows, cols)`, with **axis 0 = rows, axis 1 = columns**; build 1D with
`arange`/`array`/`zeros`/`ones`/`linspace` and 2D from lists-of-lists or
`vstack`; index **RC** (`arr[row, col]`, `:` = all); **reshape** re-fences in
reading order (a (4,3) stays a stream) while **`.T` transposes** a (4,3) into a
(3,4) by folding rows into columns; slices and `.ravel()` are **views** (use
`.copy()` to be safe); **filter** by building a boolean mask and applying it in
the slot of the axis you're filtering; for stats, **`axis` is the direction you
collapse** so per-row means use `axis=1`; build random data from one stateful,
seeded `default_rng` (matching draws need two same-seeded generators); and the
capstone SEM simulation fuses a 2D random array, `axis=1` means, and `ddof=1` to
show SEM shrinking like `1/√n`.
