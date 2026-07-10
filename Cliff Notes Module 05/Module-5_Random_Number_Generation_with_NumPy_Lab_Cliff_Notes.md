# Module 5 — Random Number Generation with NumPy (Cliff Notes)

Quick-reference notes for the Module 5 lab. Focus areas: how NumPy's random
generator works, why notebooks print what they print (`__repr__`), and a deep
dive on Practice Exercise 1 — the part most people get wrong on the first try.

---

## 1. The random number generator object

Instead of calling `np.random.something` over and over, we build **one generator
object** and reuse it:

```python
import numpy as np
rg = np.random.default_rng(seed=2100)
```

- `default_rng(...)` returns a **Generator** object (the modern NumPy RNG).
- `seed=` makes results **reproducible** — same seed → same sequence of numbers
  every time you run the notebook. Great for labs, debugging, and grading.
- The generator is **stateful**: every time you draw from it, it advances.
  Remember this — it's the key to Exercise 1 below.

### Drawing numbers from it

| Call | Produces | Notes |
|------|----------|-------|
| `rg.integers(low, high, size=...)` | random **integers** | `high` is **exclusive** (you get `low` … `high-1`) |
| `rg.random(shape)` | random **floats in [0, 1)** | sampled from a uniform distribution |
| `rg.normal(mean, std, shape)` | random floats from a **Gaussian** | specify mean, std, and return shape |

```python
rg.integers(1, 11, size=10)     # 10 ints from 1 to 10 (11 is excluded)
rg.random((4, 5))               # 4x5 array of floats in [0, 1)
rg.normal(0, 1, (3, 5))         # 3x5 array from a normal dist (mean 0, std 1)
```

**Boundary gotcha:** `integers(50, 100)` gives 50–99, *not* 50–100. If a prompt
says "between 50 and 100" and you want 100 included, use `integers(50, 101)`.

---

## 2. Why does the notebook print `dtype=int64` sometimes but not always?

This trips people up. Two *separate* mechanisms are involved — keep them apart.

### Mechanism A — what causes anything to print at all

In Jupyter, if a cell's **last line is a bare expression** (a value sitting alone,
not assigned to anything), the notebook automatically displays it. No `print()`
needed.

```python
arr = rg.random((4, 5))   # assignment -> displays NOTHING (it's a statement)
arr                       # bare expression on the last line -> auto-displayed
```

- An **assignment** (`arr = ...`) does work silently; it returns no value to show.
- A **bare expression** is what triggers the auto-display.
- Only the **last** bare expression shows. `print()` is different — it's an
  explicit command and fires from *anywhere* in the cell.

### Mechanism B — what the displayed text looks like: `__repr__`

When the notebook auto-displays a value, it asks the object to describe itself by
calling that object's **`__repr__`** method. `__repr__` is a built-in method that
every object's *class* (its blueprint) provides — not something you write in your
cell.

- `repr(arr)` → the text the notebook auto-display uses (calls `arr.__repr__()`).
- `str(arr)`  → the text `print(arr)` uses.
- A NumPy array, an `int`, and a `Generator` each have their **own** `__repr__`,
  which is why they describe themselves differently
  (e.g. `Generator(PCG64) at 0x...`).

**The dtype rule lives inside NumPy's array `__repr__`:** it shows the dtype only
when it is **not the default**.

| Array | Default dtype? | Shows `dtype=`? |
|-------|----------------|-----------------|
| `rg.random((4,5))` → `float64` | yes (float64 is the default) | **hidden** |
| `rg.integers(...)` → `int64`   | no (default int is `int32` on Windows) | **shown** |

So `dtype=int64` appears not because of `print` vs. bare line, but because the
integer array's type differs from NumPy's default, and `__repr__` calls that out.

> Bonus: `print(int_array)` uses `str()`, which **never** shows an array's dtype.
> The bare line uses `repr()`, which can. Same array, two different "describe
> yourself" methods.

**Mental model:** `__repr__` is a *read* button bolted onto the object's class.
Jupyter presses the button (auto-display); NumPy decides what the button prints
(hide-default-dtype). The data lives on the instance; the button lives on the
blueprint, so every array of that type describes itself the same way.

---

## 3. Practice Exercise 1 — the big one

> **Task:** Repeat a random experiment twice using the same seed — confirm the
> results match.

### The trap most people fall into

The intuitive (wrong) plan: "make one generator, draw from it twice, the seed
will make them match." It won't. **A generator mutates on every draw.** Drawing
twice from the *same* generator walks forward through its sequence, so you get two
**different** blocks:

```python
rand = np.random.default_rng(seed=2101)
draw1 = rand.random((3, 4))     # sequence item #1
draw2 = rand.random((3, 4))     # sequence item #2  -> DIFFERENT
draw1 == draw2                  # -> all False
```

A seed does **not** freeze the output to one fixed value. It fixes the
**starting point of a sequence**. Call the generator again and it steps to the
next item.

### What "results match" actually requires

Reproducibility means: **two separate generators, seeded the same, each on its
first draw.** Same seed → same sequence → identical first item.

```python
genA = np.random.default_rng(seed=2101)   # fresh generator at the start
genB = np.random.default_rng(seed=2101)   # another fresh generator at the SAME start
A = genA.random((3, 4))         # genA's first draw
B = genB.random((3, 4))         # genB's first draw
A == B                          # -> all True
```

### Full worked solution

```python
# Case 1: ONE generator, two draws, NO reset -> results DIFFER
rand = np.random.default_rng(seed=2101)
draw1 = rand.random((3, 4))
draw2 = rand.random((3, 4))
NoEquals = draw1 == draw2       # all False

# Case 2: TWO generators, SAME seed, one draw each -> results MATCH
genA = np.random.default_rng(seed=2101)
genB = np.random.default_rng(seed=2101)
A = genA.random((3, 4))
B = genB.random((3, 4))
YesEquals = A == B              # all True

print("NoEquals (one generator, two draws):\n", NoEquals)
print("\nYesEquals (two generators, same seed):\n", YesEquals)
```

### Why the comparison returns a grid of True/False

`A == B` on NumPy arrays compares **element-by-element** and returns an array of
booleans, not a single True/False. To collapse it to one answer, use
`np.array_equal(A, B)` (returns a single bool) or `(A == B).all()`.

### Takeaways

- A seeded generator is **reproducible across separate generators**, not
  **self-repeating within one** generator.
- To reset, **re-create** the generator with the same seed (or make a second one).
- The reset must sit **between two copies of the same experiment**, and you
  compare the two saved results to each other — never against `True`.

---

## 4. Practice Exercises 2 & 3 — quick notes

### Exercise 2 — 1D array of 20 ints between 50 and 100

```python
da = rg.integers(50, 101, size=20)   # 101 so that 100 is included
```
`size=20` (a single number) gives a **1D** array of length 20. Watch the
exclusive-`high` boundary.

### Exercise 3 — concatenate two 1D arrays

```python
ca = rg.integers(0, 49, size=20)
da = rg.integers(50, 101, size=20)
np.concatenate([ca, da])             # glued head-to-tail -> length 40
```

- `np.concatenate` takes a **list of arrays** as its first argument: `[ca, da]`.
- For 1D arrays there's only one direction, so no `axis` needed.
- For 2D arrays later, add `axis=0` (stack rows) or `axis=1` (stack columns) to
  choose the direction.

---

## 5. One-screen cheat sheet

```python
rg = np.random.default_rng(seed=N)   # build a reproducible generator
rg.integers(low, high, size=...)     # ints in [low, high)  <- high EXCLUSIVE
rg.random(shape)                     # floats in [0, 1)
rg.normal(mean, std, shape)          # Gaussian samples

# Reproducibility: same seed in TWO generators -> matching first draws.
# A generator MUTATES on every draw; one generator drawn twice -> different.

# Display: bare last line auto-prints via repr(); print() uses str().
# repr() shows an array's dtype only when it's NOT the default (float64).

np.array_equal(A, B)                 # single True/False for two arrays
np.concatenate([a, b])               # join arrays (add axis= for 2D)
```

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
