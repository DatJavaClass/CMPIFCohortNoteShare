# Interaction Features for Linear Regression (Lab Notebook)

**Module 13 Cliff Notes** | Source: lab notebook `interaction_features_for_linear_regression_start.ipynb`

---

## TL;DR

- An **interaction feature multiplies two inputs together** instead of just adding their separate influences. Use it when **the effect of one variable depends on the value of another** (e.g., an extra square meter is worth more in the city center than outside it).
- Plain addition (`a + b`) captures only **main effects** (each variable's stand-alone effect). An **interaction term** captures the extra push that happens **only when both conditions hold at once**.
- statsmodels' formula interface has **three ways to write the same interaction model**, and they produce **identical coefficients** (Verified):
  - **`a:b`** = the interaction term **only** (a bare colon, rarely wanted alone).
  - **`a + b + a:b`** = main effects **plus** the interaction (spelled out).
  - **`a * b`** = shortcut for **main effects plus all interactions** among them.
  - **`(a + b)**2`** = main effects plus interactions **up to degree 2** (the `**n` caps the interaction order).
- Model: **`price_usd ~ size_m2 * city_center`** on a 150-row apartment dataset. Fitted coefficients (Verified): **Intercept 31586.41, city_center[T.True] 92869.53, size_m2 2996.99, size_m2:city_center[T.True] 870.57**.
- **All four terms are significant** (p < 0.05, Verified). The positive interaction (870.57, p = 0.0064) means **each square meter in the city center adds about \$871 more to price than the same square meter outside it**, on top of the base \$2,997/m2.

> **The one takeaway:** an interaction term (`a:b`) lets the slope of one variable change depending on another variable; in statsmodels write `a * b` to get both main effects and their interaction in one shot, and a significant interaction coefficient means the combined effect is more (or less) than just adding the parts.

---

## The idea: additive effects vs. interactions

The default way to add a second input to a regression is **additive**: `price ~ size + city_center` says each variable pushes the prediction up or down **on its own**, and you just sum the pushes. Those stand-alone pushes are the **main effects**.

An **interaction** goes further: it **multiplies** two variables so that **the effect of one depends on the level of the other**. The notebook's motivating examples:

- **Weather comfort:** cold lowers comfort, rain lowers comfort (two main effects). But cold *and* rainy at the same time may feel *worse than the sum of the two* (an interaction).
- **Infection risk:** more viral exposure raises the chance of getting sick, but for an immunocompromised person that slope is *much steeper* (the exposure effect interacts with immune status).
- **This notebook's case (real estate):** an extra square meter is worth more in the city center than outside, so `size` and `city_center` interact.

Both **continuous and categorical** variables can interact, in any combination.

## Load data

The apartment dataset is read straight from a course GitHub URL (no local file):

```python
import pandas as pd

data_url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(data_url)
data.info()
```

Verified: **150 rows, 3 columns, no missing values**. The dtypes matter for what follows:

```text
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   size_m2      150 non-null    float64   <- continuous (apartment size, sq meters)
 1   city_center  150 non-null    bool      <- categorical (True/False)
 2   price_usd    150 non-null    int64     <- the target (price in USD)
```

Verified `head()`:

```text
   size_m2  city_center  price_usd
0     75.9        False     221000
1     84.3         True     467000
2    100.2        False     256000
3     78.8         True     434000
4     55.6        False     175000
```

Because `city_center` is a **`bool`**, statsmodels treats it as **categorical** automatically: it picks `False` as the **reference (baseline)** and creates the dummy term **`city_center[T.True]`**, whose coefficient is the effect of being `True` **relative to** `False`. (Verified split: **80 city-center rows, 70 not**.)

Goal: predict **`price_usd`** from **`size_m2`** and **`city_center`**, **with an interaction** between them.

## Way 1: explicit interaction with a colon (`a:b`)

To manually name an interaction between two variables, use **`variable1:variable2`** (a colon). You almost never want the interaction *alone*; you normally want the main effects too, so you list all three terms:

```python
import statsmodels.formula.api as smf

model = smf.ols(formula='price_usd ~ size_m2 + city_center + size_m2:city_center', data=data).fit()
model.params
```

Verified output:

```text
Intercept                      31586.409292
city_center[T.True]            92869.531786
size_m2                         2996.992135
size_m2:city_center[T.True]      870.572873
dtype: float64
```

Read the formula as: `size_m2` (main effect) `+ city_center` (main effect) `+ size_m2:city_center` (their interaction). The bare colon (`size_m2:city_center`) is **just the product term**; without the two main effects in front of it the model would be hard to interpret, so include them.

> **Note:** if the true relationship were purely additive, the interaction coefficient would come out **not significant**, and the main effects would soak up the variance. A significant interaction (as here) is evidence the effect really is more than additive.

## Way 2: the star shortcut (`a * b`)

For the common case of "give me the main effects **and** their interaction," use **`variable1 * variable2`**:

```python
model = smf.ols(formula='price_usd ~ size_m2 * city_center', data=data).fit()
model.params
```

Verified output (identical to Way 1):

```text
Intercept                      31586.409292
city_center[T.True]            92869.531786
size_m2                         2996.992135
size_m2:city_center[T.True]      870.572873
dtype: float64
```

**`a * b` expands to `a + b + a:b`.** Same model, same coefficients. This is the everyday syntax.

## Way 3: degree control with `(a + b)**2`

A third syntax is **`(variable1 + variable2)**2`**. The **`**2` sets the maximum *degree* of interaction** (how many variables inside the parentheses may be multiplied together at once):

```python
model = smf.ols(formula='price_usd ~ (size_m2 + city_center)**2', data=data).fit()
model.params
```

Verified output (again identical):

```text
Intercept                      31586.409292
city_center[T.True]            92869.531786
size_m2                         2996.992135
size_m2:city_center[T.True]      870.572873
dtype: float64
```

With only two variables all three syntaxes coincide, but the **degree matters once you have three or more inputs**:

- **`a * b * c`** gives **every** interaction up to the full order:
  - main effects: `a`, `b`, `c`
  - 2-way interactions: `a:b`, `a:c`, `b:c`
  - 3-way interaction: `a:b:c`
- **`(a + b + c)**2`** caps it at **2-way**: the three main effects plus the three 2-way interactions, but **no** `a:b:c` term.

So `**n` is a dial for the highest interaction order you allow. Higher-order interactions (3-way, 4-way, ...) get **harder to interpret**, so capping the degree is common.

## Checking significance: `model.pvalues`

The notebook reads significance off the coefficient p-values directly:

```python
model.pvalues
```

Verified:

```text
Intercept                      3.854865e-02   (0.0385)
city_center[T.True]            1.857520e-05   (0.0000186)
size_m2                        1.985029e-27   (~0)
size_m2:city_center[T.True]    6.370067e-03   (0.00637)
dtype: float64
```

```python
model.pvalues < 0.05
```

Verified:

```text
Intercept                      True
city_center[T.True]            True
size_m2                        True
size_m2:city_center[T.True]    True
dtype: bool
```

**Every term, including the interaction, is significant at the 0.05 level.** In particular the interaction's p-value is **0.0064 < 0.05**, so the interaction is real and worth keeping.

(Exam note: the notebook's closing markdown says "From the summary, you can see..." but the code actually pulls significance from **`model.pvalues`**, not from `model.summary()`. The values are the same either way; `model.summary()` would show these p-values in its `P>|t|` column. This model's overall fit, from a fresh `summary()` run, is **R-squared = 0.878, Adj. R-squared = 0.876, F = 351.8, Prob(F) = 1.38e-66, n = 150, Df Model = 3** (Verified).)

## Interpreting the coefficients

`city_center` is categorical, so its terms are read **relative to the `False` baseline**:

- **`Intercept = 31586.41`** is the base price for a hypothetical **0-m2, non-city-center** apartment (the reference geometry, not a real listing).
- **`city_center[T.True] = 92869.53`** (positive): being in the city center is associated with **about \$92.9k more** price than not, at the same size (the notebook rounds this to "~\$90k").
- **`size_m2 = 2996.99`** (positive): **outside** the city center, each extra square meter adds **about \$2,997** to price.
- **`size_m2:city_center[T.True] = 870.57`** (positive, the interaction): **inside** the city center, each square meter adds **about \$871 *more* than that**, i.e., the price-per-m2 slope is **steeper in the city center**.

Writing out the two fitted lines makes the interaction concrete:

```text
Not city center:  price = 31,586 + 2,997 * size
City center:      price = (31,586 + 92,869) + (2,997 + 871) * size
                        = 124,456 + 3,868 * size
```

So being in the city center **shifts the intercept up by ~\$92.9k** *and* **tilts the slope up from ~\$2,997/m2 to ~\$3,868/m2**. That slope change **is** the interaction term. Without an interaction, both lines would be forced to share one slope (parallel lines); the interaction lets each group have its own slope. Visualizing those two lines is the subject of the next notebook.

---

## Quiz-ready facts

- An **interaction feature multiplies two inputs**; it models cases where **one variable's effect depends on another's value**. Stand-alone effects are **main effects**.
- **`a:b`** (colon) = the **interaction term only**. **`a + b + a:b`** = main effects plus interaction. **`a * b`** = shortcut that **expands to `a + b + a:b`**. All three give **identical coefficients** (Verified).
- **`(a + b)**2`** = main effects plus interactions **up to degree 2**; the **`**n` caps the interaction order**. `a*b*c` gives all interactions including the 3-way `a:b:c`, whereas `(a+b+c)**2` **omits** the 3-way term.
- Higher-degree interactions are **harder to interpret**, which is why you often cap the degree.
- A **`bool`/categorical** input is auto-dummied with a reference level; here `city_center` uses `False` as baseline and reports **`city_center[T.True]`** (effect of `True` relative to `False`).
- Dataset: **150 rows, 3 columns** (`size_m2` float64, `city_center` bool, `price_usd` int64), **no missing values**, loaded from a URL.
- Fitted model `price_usd ~ size_m2 * city_center` (Verified): **Intercept 31586.41, city_center[T.True] 92869.53, size_m2 2996.99, interaction 870.57**.
- **All four p-values < 0.05** (Verified); interaction p = **0.0064**, so the interaction is significant. `model.pvalues < 0.05` returns all `True`.
- Interpreting the positive interaction: **each m2 in the city center adds ~\$871 more than outside**; the price/m2 slope rises from **~\$2,997** to **~\$3,868**. Overall **R-squared = 0.878** (Verified).
- Check coefficient significance with **`model.pvalues`** (and `model.pvalues < 0.05`); the same p-values appear in the `P>|t|` column of `model.summary()`.

---

> **See also:** "Additive Features in Statsmodels - Notebook (Cliff Notes)" for the plain `a + b` baseline this note extends, and "Prediction Visualizations for Interaction Features - Notebook (Cliff Notes)" for plotting the two non-parallel fitted lines the interaction produces (the "next notebook" this one points to). For how the `bool` input becomes `city_center[T.True]`, see "Categorical Inputs to Linear Regression - Notebook (Cliff Notes)" and the lecture note "Categorical Inputs to Linear Regression (Cliff Notes)". Related model-building tools: "Nonlinear Feature Transformations in Statsmodels - Notebook (Cliff Notes)" and "Standardizing Variables for Linear Regression - Notebook (Cliff Notes)".

---

*Source: CMPIF2100 lab notebook (personal study use).*
