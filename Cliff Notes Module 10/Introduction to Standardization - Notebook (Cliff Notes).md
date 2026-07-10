# Introduction to Standardization (Lab Notebook): visualizing the scale problem

**Module 10 Cliff Notes** | Source: lab notebook `Module_10_intro_standardization_start_lab.ipynb`

---

## TL;DR
- **Preprocessing** prepares data before analysis; **standardization** puts continuous variables on similar scales.
- The notebook loads the `penguins` dataset (344 rows, 7 columns) and uses `info()`, a box plot, and `describe()` to show the four numeric columns live on wildly different scales.
- `body_mass_g` (thousands of grams) dwarfs the bill and flipper measurements (tens to low hundreds of mm), so it would dominate any distance- or scale-sensitive method.
- **Z-standardization** rescales each feature to mean 0, standard deviation 1, via two steps: centering (subtract the mean) then scaling (divide by the standard deviation).
- This is an intro notebook: it visualizes and defines the problem but does **not** perform the transform. The companion scikit-learn notebook does that.

> **Companion note:** the lecture-based "Introduction to Standardization (Cliff Notes)" covers the same concept with the lecturer's analogy; this note focuses on what the notebook itself runs.

---

## Setup and dataset
```python
import seaborn as sns
import matplotlib.pyplot as plt
penguins = sns.load_dataset('penguins')
penguins.info()
```
`info()` reports a `DataFrame` of **344 rows x 7 columns**:

- **4 numeric** (`float64`): `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, each with **342 non-null** values (2 missing).
- **3 object** (categorical text): `species` and `island` (344 non-null, complete), and `sex` (333 non-null, 11 missing).

The missing values are just a side observation here; the notebook does not address them.

## The scale problem (box plot)
```python
sns.catplot(data=penguins, kind='box', height=3, aspect=2)
plt.show()
```
With no `x`/`y` specified, `catplot` draws a box for **every numeric column on one shared axis**. Because `body_mass_g` is measured in the thousands while the three mm columns sit in the tens to low hundreds, the body-mass box stretches the axis and flattens the others into thin slivers near the bottom. That visual mismatch is the whole point: the raw variables are not on comparable scales.

## describe() confirms it numerically
```python
penguins.describe()
```
`describe()` summarizes only the four numeric columns (count 342 each):

| column | mean | std | min | max |
|---|---|---|---|---|
| `bill_length_mm` | 43.92 | 5.46 | 32.1 | 59.6 |
| `bill_depth_mm` | 17.15 | 1.97 | 13.1 | 21.5 |
| `flipper_length_mm` | 200.92 | 14.06 | 172.0 | 231.0 |
| `body_mass_g` | 4201.75 | 801.95 | 2700.0 | 6300.0 |

The means and spreads span well over two orders of magnitude (from ~17 up to ~4200, roughly a 245x range), and `body_mass_g` dominates every statistic. This is exactly the imbalance standardization is meant to remove.

## Defining z-standardization
The closing markdown cell defines the technique:

> **Z-standardization** transforms data to a mean of 0 and a standard deviation of 1, by converting each feature's values to their **z-score** (the number of standard deviations from the mean).

Two steps, per feature:
1. **Centering:** subtract the mean from the value.
2. **Scaling:** divide the centered value by the standard deviation.

The result: every feature ends up centered at 0 with unit spread, so no single variable dominates by scale alone.

> **Precision note (sample vs population std):** the notebook cell says to divide by the *sample* standard deviation. When the transform is actually run with scikit-learn's `StandardScaler` (companion notebook), the divisor is the *population* standard deviation, `numpy` `ddof=0`, not the sample std that `pandas` `.std()` reports (`ddof=1`). On these columns `StandardScaler.scale_` equals `std(ddof=0)` (e.g. body mass 800.78) and not `std(ddof=1)` (801.95). The difference is tiny at n=342 and does not change the concept; the notebook's wording is kept as written, this is just the exact divisor to expect downstream.

## Quiz-ready facts
- **Preprocessing** = manipulating data before analysis; **standardization** is a key preprocessing technique for putting variables on similar scales.
- `penguins`: **344 rows, 7 columns**; 4 numeric (`float64`) + 3 object.
- `describe()` operates on the numeric columns only; `body_mass_g` has by far the largest mean (~4202) and std (~802).
- **Z-standardization target:** mean 0, standard deviation 1; each value becomes a **z-score** = (value - mean) / std.
- Two steps in order: **centering** (subtract mean), then **scaling** (divide by standard deviation).
- The notebook **defines and visualizes** standardization but does **not** compute it; the sklearn companion notebook performs the actual transform.

---

> **See also:** "Standardization with Scikit-learn - Notebook (Cliff Notes)" (same module) which actually performs the transform, and the lecture note "Introduction to Standardization (Cliff Notes)".

---

*Source: CMPIF2100 lab notebook (personal study use).*
