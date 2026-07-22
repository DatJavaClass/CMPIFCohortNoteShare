# Standardizing Variables for Linear Regression (Lab Notebook)

**Module 13 Cliff Notes** | Source: lab notebook `Module-13_Standardizing_variables_for_linear_regression.ipynb`

---

## TL;DR

- **Problem:** in an additive multi-predictor regression, you might compare **coefficient magnitudes** to judge which input matters more, but that only works if the inputs are on **similar scales**. A coefficient is "change in `y` per **1-unit** change in `x`," and "1 unit" means something totally different for a variable measured in millimeters versus grams.
- **Fix:** **z-standardize** the input variables first (rescale each to **mean 0, standard deviation 1**) with **`StandardScaler`** from scikit-learn, then fit. Now every coefficient is on the same footing.
- **Demo dataset:** Seaborn's **Palmer Penguins** (**344 rows**, Verified). Model predicts **`flipper_length_mm`** from **`bill_length_mm` + `body_mass_g`**, two inputs on wildly different scales (bill length ~32 to 60 mm, body mass ~2700 to 6300 g).
- **Without standardizing** (Verified coefficients): bill_length **0.549224**, body_mass **0.013051**. Naively you'd say bill_length matters ~40x more. **Wrong**, that gap is just a units artifact.
- **After standardizing** (Verified coefficients): bill_length **2.994150**, body_mass **10.450819**. Now the truth shows: **body_mass has the stronger association** (~10 vs ~3). Standardization **flipped the ranking**.
- **What does NOT change:** the two slopes' **p-values are identical** in both models (bill_length **3.31e-11**, body_mass **7.56e-75**, Verified) and **R-squared is identical** (**0.7884**, Verified). Standardization is a linear rescaling; it changes the coefficients' units, not the model's fit or significance.
- **New interpretation of a standardized coefficient:** change in `y` per **1 standard deviation** change in the input (not per 1 raw unit). So +1 SD of body_mass ~ **+10 mm** flipper; +1 SD of bill_length ~ **+3 mm** flipper.

> **The one takeaway:** to compare which input is more strongly associated with the outcome, standardize the inputs to mean 0 / SD 1 first (`StandardScaler().fit_transform(...)`), because raw coefficients are not comparable across variables on different scales; standardizing rescales the coefficients into "per 1 SD" units without changing p-values or R-squared.

---

## Setup and the standardization idea

The notebook imports the usual trio plus, later, `StandardScaler` and the statsmodels formula API:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
```

The motivating question (from the notebook's intro): in a store, how much does moving cheese to the front raise cheese sales **compared with** putting cheese on sale? To answer "which lever matters more," you compare the coefficients of an **additive** model. Two important caveats the notebook flags up front:

- **Correlated inputs break the comparison.** If the input variables are correlated, you cannot trust their relative coefficient values. Handling that is out of scope for this course, but be aware of it.
- **Scale breaks the comparison.** Even with uncorrelated inputs, comparing coefficients only makes sense if the variables are in the same general magnitude range. This note is about fixing the scale problem.

Recall from an earlier module: **z-standardization** preprocesses a variable to have **mean 0 and standard deviation 1** by subtracting the mean and dividing by the standard deviation. That is exactly the rescaling that makes coefficients comparable.

## The data: Palmer Penguins

```python
penguins = sns.load_dataset('penguins')
penguins.info()
```

Verified: **344 rows, 7 columns**. Four numeric (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, each **342 non-null**, so 2 rows have missing measurements) and three object columns (`species`, `island` at 344 non-null; `sex` at 333 non-null).

## Visualizing the scale (magnitude) of variables

To see whether variables live on different scales, box-plot all the numeric columns at once:

```python
sns.catplot(data=penguins, kind='box', aspect=2)
plt.show()
```

Passing a wide dataframe with no `x`/`y` draws **one box per numeric column** on a shared axis. Verified from the data, `body_mass_g` (values 2700 to 6300) towers over everything else; `bill_length_mm` (32 to 60), `bill_depth_mm` (13 to 22), and `flipper_length_mm` (172 to 231) get squashed into a thin band near the bottom. The plot makes the scale mismatch obvious at a glance: **body_mass_g is the variable with a vastly different scale.**

Confirm the eyeball read with numbers:

```python
penguins.describe()
```

Verified summary (numeric columns):

```text
       bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
count      342.000000     342.000000         342.000000   342.000000
mean        43.921930      17.151170         200.915205  4201.754386
std          5.459584       1.974793          14.061714   801.954536
min         32.100000      13.100000         172.000000  2700.000000
max         59.600000      21.500000         231.000000  6300.000000
```

`body_mass_g` has a mean near **4200** and SD ~**800**, while `bill_length_mm` sits near **44** with SD ~**5.5**. Those are not remotely comparable magnitudes. (Note the flipper mean **200.915205**, it comes back later.)

## Modeling WITHOUT standardization

Fit an additive two-predictor model on the raw data:

```python
import statsmodels.formula.api as smf

model = smf.ols(formula='flipper_length_mm ~ bill_length_mm + body_mass_g', data=penguins).fit()
print(model.pvalues)
print()
print(model.params)
```

Verified output:

```text
Intercept         1.697948e-138
bill_length_mm     3.306943e-11
body_mass_g        7.564660e-75
dtype: float64

Intercept         121.956042
bill_length_mm      0.549224
body_mass_g         0.013051
dtype: float64
```

Reading it:

- **Both inputs are statistically significant** (both p-values far below 0.05), so both are genuinely associated with flipper length.
- **The trap:** if you rank the two by coefficient size, bill_length (**0.549**) dwarfs body_mass (**0.013**), suggesting bill_length is ~40x more important. **This is a bad conclusion.** The coefficients are on vastly different scales, so the comparison is meaningless. body_mass's coefficient looks tiny only because a "1-gram" change is a trivially small move for a variable that ranges over thousands of grams.
- (Not printed by the notebook but Verified: this model's **R-squared is 0.7884**, and there are **342 observations** after the 2 missing rows drop out.)

## Standardizing the input variables

Standardize **only the inputs** (standardizing the outcome is generally unnecessary), and only the **continuous** ones (standardization does not apply to categorical variables). Steps:

**1. Drop the outcome, then keep only numeric inputs.** `df.select_dtypes('number')` is the easy way to grab all continuous columns:

```python
penguins_inputs = penguins.drop(columns='flipper_length_mm')
penguins_cont_inputs = penguins_inputs.select_dtypes('number')
penguins_cont_inputs.info()
```

Verified: this leaves **3 numeric columns**, `bill_length_mm`, `bill_depth_mm`, `body_mass_g` (the object columns `species`, `island`, `sex` are dropped by `select_dtypes('number')`).

> **Note:** `bill_depth_mm` rides along here because it is numeric, even though the model only uses `bill_length_mm` and `body_mass_g`. Standardizing an unused column is harmless.

**2. Standardize with `StandardScaler`.** It returns a NumPy array, so wrap it back into a DataFrame with the original column names:

```python
from sklearn.preprocessing import StandardScaler

penguins_scaler = StandardScaler()                    # initialize scaler
penguins_std_arr = penguins_scaler.fit_transform(penguins_cont_inputs)  # do the standardization
penguins_std = pd.DataFrame(penguins_std_arr, columns=penguins_cont_inputs.columns)  # array -> DataFrame
penguins_std.info()
penguins_std.head()
```

Verified `head()` (values are now z-scores, centered near 0; the missing row stays `NaN`):

```text
   bill_length_mm  bill_depth_mm  body_mass_g
0       -0.884499       0.785449    -0.564142
1       -0.811126       0.126188    -0.501703
2       -0.664380       0.430462    -1.188532
3             NaN            NaN          NaN
4       -1.324737       1.089724    -0.938776
```

`fit_transform` learns each column's mean and SD (Verified means **43.92 / 17.15 / 4201.75**, matching `describe()`) and rescales every value to `(x - mean) / std`. Missing rows are ignored when computing the statistics and remain `NaN` in the output.

> **Exam note (subtle but fair game):** `StandardScaler` divides by the **population** standard deviation (ddof = 0, dividing by N), whereas pandas `.std()` / `.describe()` use the **sample** standard deviation (ddof = 1, dividing by N-1). So if you call `.describe()` on the standardized frame, the SD reads **~1.0015**, not exactly 1.0000. The result is still "mean 0, SD 1" for all practical purposes; the tiny gap is just the ddof convention. It does not change any conclusion below.

**3. Re-check the scale with a boxplot:**

```python
sns.catplot(penguins_std, kind='box', aspect=2)
plt.show()
```

Verified from the standardized values: all three boxes are now **centered on 0 with comparable spreads** (roughly the -2 to +2 range, similar IQRs). The body_mass box no longer dominates. The variables are finally on the same footing.

**4. Add the (un-standardized) outcome back in:**

```python
penguins_std['flipper_length_mm'] = penguins['flipper_length_mm']
penguins_std.info()
```

Now `penguins_std` has the 3 standardized inputs plus the raw `flipper_length_mm` outcome, ready to model.

## Modeling WITH standardization

Same formula, same data shape, but standardized inputs:

```python
import statsmodels.formula.api as smf

model = smf.ols(formula='flipper_length_mm ~ bill_length_mm + body_mass_g', data=penguins_std).fit()
print(model.pvalues)
print()
print(model.params)
```

Verified output:

```text
Intercept         0.000000e+00
bill_length_mm    3.306943e-11
body_mass_g       7.564660e-75
dtype: float64

Intercept         200.915205
bill_length_mm      2.994150
body_mass_g        10.450819
dtype: float64
```

What changed and what did not:

- **The coefficient ranking flipped.** Now **body_mass (10.45) > bill_length (2.99)**, so **body mass has the stronger positive association** with flipper length. This is the opposite of the raw-coefficient impression, and it is the trustworthy answer because both inputs are now on a mean-0 / SD-1 scale.
- **The slope p-values are unchanged.** bill_length is still **3.306943e-11** and body_mass still **7.564660e-75**, byte-for-byte identical to the un-standardized model (Verified). Standardization is a **linear** transformation, so it cannot alter a coefficient's t-statistic or significance; it only rescales the coefficient's magnitude and units.
- **Model fit is unchanged.** R-squared is still **0.7884** (Verified identical to the raw model). You have re-expressed the same fit, not improved it.
- **The intercept is now the outcome's mean.** The standardized intercept is **200.915205**, which is exactly the mean of `flipper_length_mm` (Verified). With every input centered at 0, the prediction at "all inputs = 0" is the prediction at each input's mean, which for an OLS fit equals the mean of `y`. (Its p-value prints as `0.0` because testing "is the average flipper length different from 0?" is overwhelmingly significant.)

## Interpreting standardized coefficients

The one thing you give up by standardizing is the raw-unit meaning of the coefficient. The new reading:

- A standardized coefficient is the change in the outcome per **1 standard deviation** change in that input (instead of per 1 raw unit).
- So a **1-SD increase in `body_mass_g`** is associated with about a **+10 mm** increase in `flipper_length_mm`, while a **1-SD increase in `bill_length_mm`** is associated with about a **+3 mm** increase. Body mass moves the needle more, per comparable-sized change.

This "per 1 SD" interpretation is the whole point: it puts every predictor's effect in the same currency, so the coefficients are finally comparable.

---

## Quiz-ready facts

- **Why standardize:** raw regression coefficients are **not comparable across inputs on different scales**, because a coefficient is "change in `y` per 1-unit change in `x`" and 1 unit differs wildly between, say, millimeters and grams. Standardizing puts them on equal footing.
- **z-standardization** rescales a variable to **mean 0, standard deviation 1** via `(x - mean) / std`. Tool: **`StandardScaler`** from `sklearn.preprocessing`; `scaler.fit_transform(df)` returns a **NumPy array**, so wrap it back with `pd.DataFrame(arr, columns=...)`.
- **Standardize inputs only, and only continuous ones.** Standardizing the outcome is generally unnecessary; standardization does not apply to categorical variables. Select continuous columns with **`df.select_dtypes('number')`**.
- **Dataset:** Palmer Penguins from Seaborn, **344 rows**; numeric columns have 342 non-null (2 rows missing). Model: `flipper_length_mm ~ bill_length_mm + body_mass_g`.
- **Raw coefficients (Verified):** bill_length **0.549224**, body_mass **0.013051**, misleadingly suggesting bill_length dominates.
- **Standardized coefficients (Verified):** bill_length **2.994150**, body_mass **10.450819**, correctly showing **body_mass has the stronger association**. Standardization flipped the ranking.
- **Invariant under standardization:** slope **p-values** (bill_length 3.31e-11, body_mass 7.56e-75) and **R-squared (0.7884)** are **identical** in both models. A linear rescaling changes coefficient magnitudes/units, not significance or fit quality.
- **Standardized-coefficient interpretation:** change in `y` per **1 standard deviation** change in the input. +1 SD body_mass ~ **+10 mm** flipper; +1 SD bill_length ~ **+3 mm** flipper.
- **Standardized intercept = mean of the outcome** (here **200.915205** = mean `flipper_length_mm`), because centering all inputs at 0 makes the intercept the prediction at the inputs' means, which equals the mean of `y`.
- **Caveat:** even after standardizing, you **cannot trust relative coefficients if the inputs are correlated** (out of scope for this course, but be aware).
- **ddof gotcha:** `StandardScaler` uses the **population** SD (ddof = 0); pandas `.std()` uses the **sample** SD (ddof = 1), so a standardized column's pandas SD reads ~1.0015, not exactly 1.

---

> **See also:** "Additive Features in Statsmodels - Notebook (Cliff Notes)" for the additive multi-predictor models whose coefficients this note is trying to compare fairly, and "Categorical Inputs to Linear Regression - Notebook (Cliff Notes)" (plus the lecture note "Categorical Inputs to Linear Regression (Cliff Notes)") for handling the non-continuous inputs that standardization deliberately skips.

---

*Source: CMPIF2100 lab notebook (personal study use).*
