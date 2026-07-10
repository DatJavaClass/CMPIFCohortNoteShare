# Standardization with Scikit-learn: the init / fit / transform pipeline

**Module 10 Cliff Notes** | Source: lecture transcript on standardization with scikit-learn

---

## TL;DR

- **scikit-learn** (Python package name **`sklearn`**) does z-standardization for you via **`StandardScaler`**, imported from **`sklearn.preprocessing`**.
- Every scikit-learn transform follows the same **3-step pipeline**: (1) **initialize** an object, (2) **fit** it to a dataset (it estimates the parameters it needs), (3) **transform** the dataset with the fitted object.
- For `StandardScaler`, fitting estimates the **mean** and **standard deviation** of each column (stored as **`.mean_`** and **`.scale_`**); transforming subtracts the mean and divides by the std.
- Standardize **continuous variables only**: grab the numeric columns with **`select_dtypes`** and **`.copy()`** (so you do not accidentally modify the original frame).
- Result is **centered at 0 with std 1**, on a **NumPy array of the same shape** as the input; the **distribution shape is unchanged**, only the scale (axis values) moves.

> **Why bother:** the penguin variables live on wildly different scales (body mass is in the thousands, everything else is small), so standardizing puts them all on one comparable scale without distorting their shapes.

---

## Setup: importing scikit-learn

Same old `pandas`, `seaborn`, `matplotlib`. The new one is **scikit-learn**, whose Python package name is just **`sklearn`**. We pull the **`StandardScaler`** class out of its **`preprocessing`** module.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# scikit-learn -> package name is "sklearn"
from sklearn.preprocessing import StandardScaler
```

`StandardScaler` is the class that performs z-standardization.

---

## The scikit-learn 3-step pipeline

scikit-learn functions almost always follow the same three steps when transforming data:

1. **Initialize** a scikit-learn object (optionally with specifications; not needed here).
2. **Fit** the object to a dataset: it calculates the values it will need later. For `StandardScaler`, that means the **mean** and **standard deviation** of the data you pass in. (This is the "finding patterns" step.)
3. **Transform** the dataset using the fitted object.

scikit-learn keeps this friendly: the last two steps are literally **`.fit()`** and **`.transform()`**.

> Why mean and std? Z-standardization subtracts the mean from every value and divides by the standard deviation, so the fit step has to learn those two numbers first.

---

## Step 1: initialize the scaler

```python
penguins_scaler = StandardScaler()
```

This does nothing on its own; it just puts an (unfitted) scaler object in your environment, ready to be fitted.

---

## Step 2: select continuous columns, then fit

Z-standardization is a technique for **continuous variables only**, so first pull just the numeric columns. Use **`select_dtypes`** to select by data type, and add **`.copy()`** so that working on this subset never modifies the original frame.

```python
penguins = sns.load_dataset("penguins")

# continuous (numeric) columns only, copied so the original is untouched
penguins_continuous = penguins.select_dtypes(include="number").copy()

penguins_continuous.info()
penguins_continuous.head()
```

- This drops the object/categorical columns (`species`, `island`, `sex`), leaving the four numeric ones: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- There are some **missing values (NaN)**. Not a problem: they are simply ignored during fitting and pass straight through as NaN on transform.

Now fit the scaler to those columns:

```python
penguins_scaler.fit(penguins_continuous)
```

In a Jupyter notebook this prints a little diagram of the fitted estimator. Nothing visibly "runs," but the object is now fitted, which you can confirm by inspecting the values it learned.

### Inspect what fit learned: `.mean_` and `.scale_`

```python
penguins_scaler.mean_    # the per-column means
penguins_scaler.scale_   # the per-column standard deviations
```

Both are **NumPy arrays with 4 values** (one per input variable). Verified values:

```text
mean_  = [  43.92,   17.15,  200.92, 4201.75]   # body_mass mean is in the thousands
scale_ = [   5.45,    1.97,   14.04,  800.78]   # body_mass std is far larger too
```

The trailing underscore (`mean_`, `scale_`) is scikit-learn's convention for "attribute learned during fit."

> **Precision note (added, verified):** `scale_` is the **population** standard deviation (NumPy `ddof=0`), not the sample std that pandas `.std()` reports by default (`ddof=1`). That is why `scale_ = [5.4516, 1.9719, 14.0411, 800.7812]` differs slightly from `penguins_continuous.std()` = `[5.4596, 1.9748, 14.0617, 801.9545]`. Same idea, different divisor (`n` vs `n-1`).

---

## Step 3: transform the data

```python
penguins_standardized_array = penguins_scaler.transform(penguins_continuous)
penguins_standardized_array.shape   # (344, 4)
```

- Despite the name "transform," it **does not mutate** the input; it **returns a new NumPy array**, so the original `penguins_continuous` is safe.
- The output is a plain NumPy array of shape **(344, 4)**: 344 penguins (rows) and the 4 continuous columns, matching the input.

### Wrap it back into a DataFrame

A raw NumPy array has no column names, so rebuild a DataFrame and hand it the **original columns**:

```python
penguins_standardized = pd.DataFrame(
    data=penguins_standardized_array,
    columns=penguins_continuous.columns,   # NumPy dropped the names; reuse the originals
)
penguins_standardized
```

The values now look completely different from the thousands-scale originals: small decimals scattered around zero, some negative, some positive.

---

## Confirm the result: `describe()`

```python
penguins_standardized.describe()
```

- **Means are essentially 0.** pandas prints them in scientific notation, e.g. **`8.31e-17`** (that is `8.31 x 10^-17`, i.e. move the decimal 17 places left, so basically zero). All four column means read as tiny numbers like this.
- **Standard deviations are right around 1.** That is exactly what z-standardization delivers.

> **Quiz point:** after z-standardization the **standard deviation should be 1**.

> **The 1.0015 gotcha (added, verified):** `describe()` actually shows std = **1.0014652**, not a clean `1.000`, because pandas `.std()` uses the **sample** formula (`ddof=1`). In the **population** sense (`ddof=0`) the standardized std is **exactly 1** - which is the "should be 1" the quiz means. So do not be thrown when `describe()` prints `1.0015`; it is the same result, just measured with `n-1`.

---

## Before vs after: box plots

```python
# before: original scale
sns.boxplot(data=penguins_continuous)

# after: standardized
sns.boxplot(data=penguins_standardized)
```

- **Before:** `body_mass_g` towers in the thousands and crushes the other variables against the bottom; you cannot compare them.
- **After:** everything sits on the **same scale** around zero. That is the whole point.

---

## Shape is preserved: histogram + KDE

Standardizing changes the scale, **not the shape** of a distribution. Check `flipper_length_mm` before and after:

```python
# before
sns.displot(data=penguins_continuous, x="flipper_length_mm", kde=True, height=3)

# after
sns.displot(data=penguins_standardized, x="flipper_length_mm", kde=True, height=3)
```

- The two histograms (with KDE overlaid, `height=3` just to fit the screen) have the **same shape**.
- Only the **x-axis tick values change**: the original ran up around 200 to 300, whereas the standardized version is **centered near 0 and includes negatives**. That shift is the centering-and-scaling that z-standardization does.

---

## Quiz-ready facts

- **`StandardScaler`** comes from **`sklearn.preprocessing`** (`from sklearn.preprocessing import StandardScaler`); scikit-learn's package name is **`sklearn`**.
- Every scikit-learn transform = **initialize -> fit -> transform**.
- **Fit** estimates the parameters and stores them as attributes: **`.mean_`** and **`.scale_`** (both NumPy arrays, 4 values here = 4 input variables; the trailing `_` marks a learned attribute).
- **Transform** returns a **new NumPy array** of shape **(344, 4)** and **does not mutate** the input; wrap it in a DataFrame with the original `.columns` to restore names.
- Standardize **continuous variables only**: select them with **`select_dtypes`** + **`.copy()`**. Missing values (NaN) are ignored and passed through.
- After z-standardization: **mean approximately 0** (`describe()` shows scientific-notation values like `8.31e-17`) and **standard deviation = 1**.
- The **distribution shape is unchanged**; only the scale (axis values, now centered at 0 with negatives) changes.
- Precision detail: `scale_` uses the **population** std (`ddof=0`), so it differs slightly from pandas `.std()` (`ddof=1`); likewise `describe()` shows the standardized std as `1.0015` (sample) while the population std is exactly `1`.

---

> **See also:** the companion lab note "Standardization with Scikit-learn - Notebook (Cliff Notes)" (same module, the notebook version of this lesson), and the prerequisite "Introduction to Standardization (Cliff Notes)" / "Introduction to Standardization - Notebook (Cliff Notes)" which motivates why standardization is needed.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
