# Standardization with Scikit-learn (Lab Notebook): running init / fit / transform on the penguins

**Module 10 Cliff Notes** | Source: lab notebook `Module_10_standardization_with_scikit-learn_start_lab.ipynb`

---

## TL;DR
- The notebook runs z-standardization for real with **`StandardScaler`** from **`sklearn.preprocessing`**, following the scikit-learn **3-step pipeline**: initialize, fit (estimate parameters), transform.
- Grab continuous columns only with **`penguins.select_dtypes('number').copy()`**: the four numeric penguin columns, 342 non-null of 344 rows each.
- **Fit** learns the per-column mean and std, stored as **`.mean_`** and **`.scale_`** (NumPy arrays of 4 values). `mean_ = [43.92, 17.15, 200.92, 4201.75]`, `scale_ = [5.45, 1.97, 14.04, 800.78]`.
- **Transform** returns a **new NumPy array of shape (344, 4)**, does **not** mutate the input, and passes NaN rows straight through. Wrap it back into a DataFrame with the original `.columns`.
- Result is **centered at 0** (means print as ~1e-16) with **std 1**; the **distribution shape is unchanged**, only the scale moves.

> **Companion note:** the lecture-based "Standardization with Scikit-learn (Cliff Notes)" narrates the same steps; this note focuses on the notebook's actual code, outputs, and numbers.

---

## Setup and the 3-step scikit-learn pipeline
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
```
scikit-learn (package name `sklearn`) transforms follow the same three steps, spelled out in the notebook's markdown:
1. **Initialize** a scikit-learn object with any desired specifications.
2. **Fit** it to a dataset: estimate the parameters from data. For z-standardization that is the **mean** and **standard deviation** of each column.
3. **Transform** a dataset using the fitted object.

The scaler is initialized with defaults (no arguments needed here):
```python
penguins_scaler = StandardScaler()
```
On its own this does nothing but put an unfitted scaler in the environment.

## Load the data and select continuous variables
```python
penguins = sns.load_dataset('penguins')
penguins.info()
```
`info()` reports a `DataFrame` of **344 rows x 7 columns**: 4 numeric (`float64`) and 3 object columns.

Standardization applies to continuous variables only, so pull just the numeric columns and copy them:
```python
penguins_continuous = penguins.select_dtypes('number').copy()
penguins_continuous.info()
penguins.head()
```
- `select_dtypes('number')` drops the object columns (`species`, `island`, `sex`), leaving the four numeric ones: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, each with **342 non-null** values (2 missing).
- `.copy()` protects the original `penguins` frame from any later in-place edits.
- The notebook's `penguins.head()` (the full original frame, not the numeric subset) shows row 3 is all `NaN` for the numeric columns: those missing values are real and stay in play.

## Fit: what `.mean_` and `.scale_` learn
```python
penguins_scaler.fit(penguins_continuous)
```
In Jupyter this just echoes `StandardScaler()`; nothing visibly runs, but the object is now fitted. Inspect what it learned:
```python
print(penguins_scaler.mean_)
print(penguins_scaler.scale_)
```
```text
mean_  = [  43.92192982   17.15116959  200.91520468 4201.75438596]
scale_ = [   5.45159602    1.97190392   14.04114057  800.78122924]
```
Both are NumPy arrays with **4 values** (one per input column). The trailing underscore (`mean_`, `scale_`) is scikit-learn's convention for "attribute learned during fit."

> **Precision note (population vs sample std, verified by running the pipeline):** `scale_` is the **population** standard deviation (NumPy `ddof=0`), not the sample std pandas `.std()` reports by default (`ddof=1`). Confirmed on these columns: `scale_` equals `penguins_continuous.std(ddof=0)` (to floating-point precision), and differs from `penguins_continuous.std(ddof=1) = [5.4596, 1.9748, 14.0617, 801.9545]`. Same idea (divide by the spread), different divisor (`n` vs `n-1`). The gap is tiny at n=342 but the exact divisor is `n`. If a lecture calls `scale_` the "sample standard deviation," that wording is imprecise: `StandardScaler` divides by the population std.

## Transform: NumPy array of shape (344, 4)
```python
penguins_std_arr = penguins_scaler.transform(penguins_continuous)
penguins_std_arr.shape        # (344, 4)
```
- `transform` returns a **new NumPy array**; it does **not** mutate `penguins_continuous` (its NaN row stays NaN, untouched).
- Shape is **(344, 4)**: all 344 rows and 4 columns, matching the input. Missing values pass through as NaN.

A raw array has no column names, so rebuild a DataFrame with the original columns:
```python
penguins_std = pd.DataFrame(penguins_std_arr, columns=penguins_continuous.columns)
penguins_std
```
Values are now small decimals scattered around 0 (row 0 is about `[-0.88, 0.79, -1.42, -0.56]`); row 3 stays `NaN` across all four columns.

## Confirm the result numerically
Running `penguins_std.describe()` on the standardized frame (this call is not in the notebook; computed separately here) shows the two hallmarks of z-standardization:
- **Means are essentially 0.** pandas prints them in scientific notation, e.g. `8.31e-17` and `-1.41e-15` (that is ~0.00000000000000008, i.e. zero for all practical purposes).
- **Std is right around 1.** `describe()` prints **1.0014652** for every column, not a clean `1.000`.

> **The 1.0015-vs-1.000 point (verified):** `describe()` shows `1.0014652` because pandas `.std()` uses the **sample** formula (`ddof=1`). In the **population** sense (`ddof=0`) the standardized std is **exactly 1.0**. Both describe the same result; the `n-1` divisor just nudges the printed figure slightly above 1. The intended quiz answer for "std after z-standardization" is **1**; do not be thrown when `describe()` reads `1.0015`.

## Distribution shape is unchanged (box plot)
The notebook's markdown notes that standardization does not change the **shape** of the distributions, then draws a box plot of the standardized frame:
```python
sns.catplot(data=penguins_std, kind='box', height=3, aspect=2)
plt.show()
```
With no `x`/`y` given, `catplot` draws a box for **every column on one shared axis**. Because the data is now standardized, all four boxes sit on a **comparable scale around 0** (instead of `body_mass_g` in the thousands dwarfing the mm columns). The relative spread and skew of each box is preserved: only the scale moved. `height=3, aspect=2` just size the figure.

> This notebook draws only the **after** (standardized) box plot. The **before** (raw-scale) box plot, where `body_mass_g` flattens the others, lives in the prerequisite "Introduction to Standardization - Notebook (Cliff Notes)."

## Quiz-ready facts
- **`StandardScaler`** comes from **`sklearn.preprocessing`**; scikit-learn's package name is **`sklearn`**.
- Every scikit-learn transform = **initialize -> fit -> transform**.
- **Fit** estimates parameters and stores them as **`.mean_`** and **`.scale_`** (NumPy arrays, 4 values here; trailing `_` marks a learned attribute).
- `scale_` is the **population** std (`ddof=0`), so it differs slightly from pandas `.std()` (`ddof=1`); e.g. body mass `800.78` (population) vs `801.95` (sample).
- **Transform** returns a **new NumPy array of shape (344, 4)** and **does not mutate** the input; wrap it in a DataFrame with `penguins_continuous.columns` to restore names.
- Standardize **continuous variables only**: select with **`select_dtypes('number')`** + **`.copy()`**. Missing values (NaN) are ignored during fit and pass through transform as NaN.
- After z-standardization: **mean approximately 0** (`describe()` shows sci-notation like `8.31e-17`) and **std = 1** (population). `describe()` prints `1.0015` because it uses the sample formula (`ddof=1`); the quiz answer is **1**.
- The **distribution shape is unchanged**; only the scale (axis values, now centered at 0 with negatives) changes.

---

> **See also:** the lecture note "Standardization with Scikit-learn (Cliff Notes)", and the prerequisite "Introduction to Standardization - Notebook (Cliff Notes)" / "Introduction to Standardization (Cliff Notes)".

---

*Source: CMPIF2100 lab notebook (personal study use).*
