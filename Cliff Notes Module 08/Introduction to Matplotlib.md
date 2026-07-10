# Introduction to Matplotlib

## Context
- matplotlib is the foundational plotting package in Python. Worth knowing even though it's clunky.
- Seaborn (covered later) is more intuitive but is built on top of matplotlib, so the basics carry over.

## Import convention
```python
import numpy as np
import matplotlib.pyplot as plt
```
- You import the `pyplot` module from matplotlib, aliased `plt`.

## The four-step plotting workflow
1. **Define figure and axes objects** — set up the canvas you'll plot on.
2. **Plot the data** into the axes object(s).
3. **Modify** axes/figure attributes as needed (size, labels, etc.).
4. **Show** the figure.

## Figure vs. axes (the confusing part)
```python
fig, ax = plt.subplots()
```
- `plt.subplots()` returns two objects: a **figure** and an **axes**.
- **Figure** = the whole picture / outer frame / canvas. Can hold multiple plots.
- **Axes** = an individual subplot (the actual plotting area). NOT the x/y axis.
  - "axes" here means the whole subplot; "axis" (x-axis, y-axis) is something different. The naming is genuinely confusing.
- Calling `plt.subplots()` immediately renders an empty canvas.

## Multiple subplots
```python
fig, axes = plt.subplots(2)        # 2 rows, 1 column
fig, axes = plt.subplots(2, 3)     # 2 rows, 3 columns -> rows first, then columns
```
- With multiple subplots, `axes` becomes a **NumPy array of axes objects**.
- Index it like a NumPy array: `axes[0]`, `axes[1]`, etc.
- For a grid, `axes[row, col]`. Top-middle of a 2x3 grid = `axes[0, 1]` (row 0, column 1).
- Convention matches NumPy: **rows first, then columns**.

### Cleaning up shared axes
```python
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
```
- `sharex` / `sharey` remove repeated tick labels across subplots for a cleaner look.

## Plotting data into axes
```python
data = np.linspace(0, 10, 100)     # 100 points from 0 to 10
sine_data = np.sin(data)
cosine_data = np.cos(data)

fig, axes = plt.subplots(2)
axes[0].plot(sine_data)
axes[1].plot(cosine_data)
fig                                 # display the figure
```
- Grab the target axes object and call `.plot()` on it.
- Display by evaluating the `fig` variable (or use `plt.show()`).

## Controlling figure size
```python
fig, axes = plt.subplots(2, figsize=(4, 4))
axes[0].plot(sine_data)
```
- `figsize=(width, height)` in **inches** (horizontal, vertical).
- `(4, 4)` produces a square figure.

## Key terms
- **figure**: entire image / canvas, can contain multiple subplots.
- **axes**: a single subplot region (multiple = a NumPy array).
- **subplots / facets**: the individual plots within a figure.
- **linspace(start, stop, n)**: NumPy helper for n evenly spaced points in a range.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
