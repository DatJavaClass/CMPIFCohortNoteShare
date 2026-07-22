# Prediction Visualizations for Interaction Features (Lab Notebook)

**Module 13 Cliff Notes** | Source: lab notebook `Module-13_Prediction_visualizations_for_interaction_features.ipynb`

---

## TL;DR

- The notebook fits a linear regression **with an interaction term** and then **draws a picture of the fitted model** to show what "interaction" actually looks like. Dataset: an **apartment price** file (`apt_real_estate.csv`, **150 rows**), predicting **`price_usd`** from **`size_m2`** (float) and **`city_center`** (a boolean, in vs out of the city center).
- The formula uses the **`*` operator**: **`smf.ols('price_usd ~ size_m2 * city_center', data=data).fit()`**. In a statsmodels formula, **`a * b` expands to `a + b + a:b`**, so one `*` gives you **both main effects plus the interaction term** `size_m2:city_center`.
- Fitted model (Verified): **R-squared = 0.878**, and the interaction coefficient **`size_m2:city_center[T.True] = 870.57`** is significant (**t = 2.77, p = 0.006**). Positive interaction = the size-to-price slope is **steeper inside the city center**.
- The core technique is a **prediction grid**: build a `DataFrame` of every combination of a dense range of `size_m2` values (**`np.linspace(min, max, 101)`**) crossed with **both `city_center` values**, giving **202 rows** (Verified), then call **`model.predict(viz_grid)`** and plot the predictions.
- Plotting `pred` vs `size_m2` with **`hue='city_center'`** and **`kind='line'`** gives **two non-parallel lines**. The city-center line rises faster. **Non-parallel = interaction**; parallel lines would mean additive (no interaction).

> **The one takeaway:** to *see* an interaction, feed a fitted model a grid of input combinations (`np.linspace` for the numeric axis crossed with every group value), call `model.predict()` on the grid, and line-plot the predictions with `hue` set to the grouping variable. Interaction shows up as lines with **different slopes**; additive effects show up as **parallel** lines.

---

## The idea: what an interaction models

Premise: in the apartment dataset, the relationship between **size** and **price** may **depend on** whether the apartment is in the city center. An extra square meter downtown may add more dollars than an extra square meter in the suburbs. An **interaction term** is exactly the tool that lets the **slope of one variable change with the value of another**.

This notebook does not just report the interaction coefficient; it **visualizes** the fitted model so the effect is obvious to the eye.

## Setup and the data

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
```

`numpy` is the new import here (versus the plain-fit notebooks), because the prediction grid is built with **`np.linspace`**.

```python
data_url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(data_url)
data.info()
```

Verified structure: **150 rows, 3 columns, no missing values**:

```text
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   size_m2      150 non-null    float64
 1   city_center  150 non-null    bool
 2   price_usd    150 non-null    int64
```

Note the dtypes: **`size_m2` is numeric (float64)**, **`price_usd` is numeric (int64)**, and **`city_center` is a boolean** (`True`/`False`). Statsmodels treats that boolean as a **categorical input** automatically, which is why its coefficient shows up as `city_center[T.True]` later. (Extra context, Verified but not printed in the lab: 80 rows are `True`, 70 are `False`; `size_m2` ranges from **25.0 to 129.1**.)

## Visualize before fitting

```python
sns.relplot(data=data, x='size_m2', y='price_usd', hue='city_center')
plt.show()
```

A scatterplot of **price (y) vs size (x)**, with points **colored by `city_center`**. Verified from the data: both color groups slope **up and to the right** (bigger apartments cost more), but the **city-center points sit higher and climb more steeply**. That visual hint (two groups with different-looking slopes) is the motivation for adding an interaction term.

## Fit with an interaction: the `*` operator

```python
model = smf.ols('price_usd ~ size_m2 * city_center', data=data).fit()
print(model.summary())
```

The key is the **`*`** in the formula. In statsmodels' R-style formulas:

```text
size_m2 * city_center   ==   size_m2 + city_center + size_m2:city_center
```

- **`size_m2`** on its own = the **main effect** of size.
- **`city_center`** on its own = the **main effect** of being downtown.
- **`size_m2:city_center`** (the colon) = the **pure interaction term**, the part that lets size's slope differ by group.

So **`*` is the shortcut for "both main effects AND their interaction."** If you wrote just the colon (`size_m2:city_center`) you would get the interaction *without* the main effects, which is almost never what you want.

Verified summary:

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:              price_usd   R-squared:                       0.878
Model:                            OLS   Adj. R-squared:                  0.876
Method:                 Least Squares   F-statistic:                     351.8
Date:                Wed, 22 Jul 2026   Prob (F-statistic):           1.38e-66
Time:                        13:41:46   Log-Likelihood:                -1783.5
No. Observations:                 150   AIC:                             3575.
Df Residuals:                     146   BIC:                             3587.
Df Model:                           3
Covariance Type:            nonrobust
===============================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                    3.159e+04   1.51e+04      2.088      0.039    1686.867    6.15e+04
city_center[T.True]          9.287e+04    2.1e+04      4.427      0.000    5.14e+04    1.34e+05
size_m2                      2996.9921    222.241     13.485      0.000    2557.767    3436.217
size_m2:city_center[T.True]   870.5729    314.505      2.768      0.006     249.002    1492.143
==============================================================================
Omnibus:                        3.889   Durbin-Watson:                   2.090
Prob(Omnibus):                  0.143   Jarque-Bera (JB):                3.890
Skew:                          -0.358   Prob(JB):                        0.143
Kurtosis:                       2.671   Cond. No.                         636.
==============================================================================
```

### Reading the four coefficients

The model has **four terms** (that is why `Df Model = 3`: three predictor terms plus the intercept). Verified full-precision values:

- **`Intercept = 31586.41`**: predicted price of a **size = 0** apartment that is **NOT** in the city center (the baseline group). It is just the geometric anchor of the baseline line.
- **`city_center[T.True] = 92869.53`**: how much the **intercept shifts up** for city-center apartments. `[T.True]` means "treatment coding, the `True` level relative to the `False` baseline." So the downtown line starts about **\$92,870 higher** at size = 0.
- **`size_m2 = 2996.99`**: the **slope of size for the baseline (non-city-center) group**. Each extra square meter outside the center adds about **\$2,997**.
- **`size_m2:city_center[T.True] = 870.57`**: the **interaction**, i.e. how much **steeper** the size slope becomes for city-center apartments. It is a **difference of slopes**, not a slope itself.

So the two group lines are:

```text
Not city center:  price = 31586.41            + 2996.99            x size_m2
City center:      price = (31586.41+92869.53) + (2996.99+870.57)   x size_m2
                        = 124455.94           + 3867.57            x size_m2
```

Verified: the grid predictions reproduce these exactly (e.g. size = 25 gives \$106,511 outside the center and \$221,145 inside; the empirical slopes over the grid are **2996.99** and **3867.57**).

### Is the interaction real?

The notebook's markdown reads the interaction row as **positive and significant (p < 0.05)**: the coefficient `870.57` is **positive**, its **t = 2.77**, and its **p-value = 0.006** (Verified `6.37e-03`). A positive interaction means the size-to-price slope has a **steeper increase inside the city center**, which is what the scatterplot suggested.

> **Exam note (source wording):** the lab describes the interaction as "positive (2.78) and significant." The **coefficient** is `870.57` (that is what "positive" refers to); the number **2.78** is the **t-statistic** (`2.768`, more precisely rounding to 2.77), not the coefficient. Do not read "2.78" as the size of the effect. The effect size is **+\$870.57 per m² of extra steepness downtown**, with **t = 2.77, p = 0.006**.

## Build the prediction grid

To visualize the fitted model you feed it a **dense, systematic set of made-up inputs** and predict on them. The notebook builds every combination of a fine range of sizes with both `city_center` values using a nested list comprehension:

```python
viz_grid = pd.DataFrame(
    [ (size_m2, city_center)
        for size_m2 in np.linspace(data.size_m2.min(), data.size_m2.max(), 101)
        for city_center in [True, False]
    ],
    columns=['size_m2', 'city_center']
)
```

What the pieces do:

- **`np.linspace(data.size_m2.min(), data.size_m2.max(), 101)`** makes **101 evenly spaced sizes** from the data's **min (25.0)** to its **max (129.1)** (Verified step ≈ **1.041**). `linspace(a, b, n)` includes **both endpoints** and yields `n` points.
- The **nested `for`** (size on the outer loop, city_center on the inner) crosses each size with **both `True` and `False`**.
- Result: **101 × 2 = 202 rows** (Verified `[202 rows x 2 columns]`). Grid columns must be **named exactly like the training columns** (`size_m2`, `city_center`) so the model can match them.

Verified head/tail of the grid:

```text
     size_m2  city_center
0     25.000         True
1     25.000        False
2     26.041         True
3     26.041        False
...
200  129.100         True
201  129.100        False
```

## Predict on the grid

```python
viz_grid['pred'] = model.predict(viz_grid)
```

**`model.predict(new_df)`** runs the fitted model on the grid and returns one predicted **average price** per row. Assigning it back adds a **`pred`** column. Verified sample:

```text
   size_m2  city_center           pred
0   25.000         True  221145.066278
1   25.000        False  106511.212665
...
200  129.100         True  623758.583613
201  129.100        False  418498.093912
```

These are **model predictions**, not observed data points, which is the whole point: they trace the **fitted lines** smoothly, even at input combinations no real apartment had.

## Visualize the interaction

```python
sns.relplot(
    data=viz_grid,
    x='size_m2',
    y='pred',
    kind='line',
    hue='city_center',
)
plt.show()
```

Two arguments make this a model-visualization rather than a scatter:

- **`kind='line'`** connects the predictions into smooth lines (the fitted relationships).
- **`hue='city_center'`** draws **one line per group** in different colors.

Verified from the predictions: **two straight lines, both rising, but NOT parallel.** The `city_center = True` line is **steeper** (slope 3867.57) and **starts higher**, so it **pulls away** from the `False` line (slope 2996.99) as size grows. The visible **fanning-out** of the two lines is the interaction.

### The key contrast: interaction vs additive

- **Interaction (this notebook):** lines have **different slopes** and fan apart. The effect of `size_m2` **depends on** the value of `city_center`.
- **Additive only (`price_usd ~ size_m2 + city_center`):** the same picture would show **parallel lines**, one shifted vertically above the other. Parallel = the effect of size is the **same** in both groups; only the baseline level differs.

Seeing parallel vs fanning lines is the fast visual test for "did my model include an interaction, and does it matter?"

---

## Quiz-ready facts

- Dataset: `apt_real_estate.csv`, **150 rows**, columns **`size_m2` (float64)**, **`city_center` (bool)**, **`price_usd` (int64)**, no missing values.
- Interaction fit: **`smf.ols('price_usd ~ size_m2 * city_center', data=data).fit()`**. In a formula, **`a * b` = `a + b + a:b`** (both main effects **plus** the interaction); the bare **`a:b`** (colon) is the **interaction term alone**.
- A boolean/categorical input like `city_center` is auto-encoded; its coefficient prints as **`city_center[T.True]`** (treatment coding, `True` relative to the `False` baseline).
- Verified fit: **R-squared = 0.878** (Adj. 0.876), **F = 351.8**, **Prob(F) = 1.38e-66**. Coefficients: Intercept **31586.41**, `city_center[T.True]` **92869.53**, `size_m2` **2996.99**, `size_m2:city_center[T.True]` **870.57**.
- The **interaction coefficient is a difference of slopes**: baseline size slope = **2996.99**; city-center size slope = **2996.99 + 870.57 = 3867.57**. The **`city_center` main effect (92869.53) is a difference of intercepts** (vertical shift), **not** a slope.
- Interaction is **positive and significant**: **t = 2.77, p = 0.006** (Verified `6.37e-03`). Positive = size raises price **faster** inside the city center.
- Prediction grid: **`np.linspace(min, max, 101)`** (101 points, **both endpoints included**) crossed with **both `city_center` values** via a nested list comprehension = **202 rows**. Grid columns must be **named exactly** like the training columns.
- Predict with **`model.predict(viz_grid)`**; store as a new **`pred`** column. These are model predictions, not observed data.
- Visualize with **`sns.relplot(..., x='size_m2', y='pred', kind='line', hue='city_center')`**. **`kind='line'`** traces the fitted lines; **`hue`** splits by group.
- Reading the picture: **non-parallel (fanning) lines = interaction**; **parallel lines = additive (no interaction)**.

---

> **See also:** "Interaction Features for Linear Regression - Notebook (Cliff Notes)" for the mechanics of the `*` operator and how the interaction coefficient is derived; "Additive Features in Statsmodels - Notebook (Cliff Notes)" for the parallel-lines case this contrasts with; and "Categorical Inputs to Linear Regression - Notebook (Cliff Notes)" (plus the lecture note "Categorical Inputs to Linear Regression (Cliff Notes)") for why the boolean `city_center` becomes `city_center[T.True]`.

---

*Source: CMPIF2100 lab notebook (personal study use).*
