# Module 13 - Cliff Jumper Notes

*One continuous lesson stitched from the seven Module 13 cliff notes: one
lecture note (Categorical Inputs to Linear Regression) and six lab notebook
notes (Additive Features in Statsmodels; Standardizing Variables for Linear
Regression; Categorical Inputs to Linear Regression, the companion lab;
Interaction Features for Linear Regression; Prediction Visualizations for
Interaction Features; and Nonlinear Feature Transformations in Statsmodels). The
through-line: **Module 12 fit one straight line to one numeric input; Module 13
makes the inputs rich. It adds multiple predictors together, standardizes them
so their coefficients can be compared fairly, encodes categorical inputs as
dummy variables, multiplies inputs together with interaction terms when one
input changes another's slope, draws those interactions, and bends the inputs
through nonlinear transforms, all while staying inside linear regression,
because "linear" means linear in the coefficients, not in the inputs.** The
whole module is one sentence stretched wide: keep `smf.ols('y ~ ...')`, and just
make the right-hand side smarter.*

> All code targets **statsmodels** (formula interface, `smf`), **pandas**,
> **numpy**, **seaborn**, and **matplotlib** with the standard aliases, plus
> **scikit-learn**'s `StandardScaler` for the standardization section. Six
> datasets appear: **synthetic seed-2100 additive data** (100 rows, built from a
> known truth); the **apartment prices CSV** (`apt_real_estate.csv`, 150 rows,
> loaded live from its course GitHub URL); seaborn's built-in **iris** (150
> rows) and **penguins** (344 rows); and two **unseeded synthetic curves** (a
> sine wave and a parabola). Every number below was reproduced by execution
> (python 3.13.9, statsmodels 0.14.5, pandas 2.3.3, numpy 2.3.5, seaborn 0.13.2,
> scikit-learn 1.7.2), with one honest exception: the nonlinear-transform data
> carries **no random seed**, so its digits are reported the way the source note
> does, as behavior-level ranges ("Verified (behavior)"), never as falsely exact
> numbers.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
```

---

## 1. The map: from one line to rich features

Module 12 fit `profit ~ population`: one numeric input, one slope, and the whole
model was a single straight line. Module 13 keeps that machinery untouched and
spends its entire effort on the **right-hand side of the tilde**. Everything new
is a way of building a better *feature* (an input to the regression) without
ever leaving ordinary least squares.

| The move | The idea | The formula | Section |
|---|---|---|---|
| **Add** | two inputs contribute independently | `'y ~ x1 + x2'` | 2 |
| **Standardize** | rescale inputs so coefficients compare fairly | `StandardScaler().fit_transform(...)` | 3 |
| **Encode** | turn a category into 0/1 dummies | `'petal_length ~ species'` | 4 |
| **Interact** | let one input change another's slope | `'price_usd ~ size_m2 * city_center'` | 5 |
| **Visualize** | draw the interaction as fanning lines | `model.predict(grid)` + `hue` | 6 |
| **Transform** | bend a straight line through a curve | `'y ~ np.sin(x)'`, `'y ~ np.power(x, 2)'` | 7 |

The unifying fact that makes all six moves "still linear regression": a linear
model is a **sum of coefficients times features**, `ŷ = β₀ + β₁f₁ + β₂f₂ + ...`,
and nothing says the features `fᵢ` have to be raw columns. They can be a second
raw column (section 2), a rescaled column (3), a 0/1 indicator (4), a product of
two columns (5), or a nonlinear function of a column (7). The estimator only
ever sees numbers to multiply by coefficients.

---

## 2. Add: two inputs, contributing independently

The simplest way past a single predictor is **addition**. The additive
two-input model is

```text
ŷ = β₀ + β₁x₁ + β₂x₂
```

where each input contributes on its own (β₁x₁ from `x1`, β₂x₂ from `x2`) and the
prediction is the sum of those contributions plus the intercept. "Additive"
means the effects **do not depend on each other**: the value of `x2` never
changes how much a one-unit move in `x1` shifts the prediction. In the formula
interface, additive inputs are just a **`+`** between them, and the intercept is
still added automatically, so you never write β₀:

```python
np.random.seed(2100)
x1 = np.random.uniform(0, 10, 100)   # uniform on [0, 10]
x2 = np.random.uniform(0, 5, 100)    # uniform on [0, 5]
y = 5 + 2*x1 - 3*x2 + np.random.normal(0, 2, 100)   # true: 5, +2, -3, noise sd 2
df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

model = smf.ols('y ~ x1 + x2', data=df).fit()
model.params
```

The synthetic data is built from a **known** truth (intercept 5, `x1` slope +2,
`x2` slope -3, plus Gaussian noise of sd 2), so we can check whether the fit
backs out the structure. It does. Verified full-precision coefficients:

```text
Intercept    4.726926
x1           1.967544
x2          -2.908475
```

The fitted line is `y = 4.7269 + 1.9675 x1 - 2.9085 x2`, close to the true 5,
+2, -3, with the gaps due to noise. Verified fit statistics: **R-squared =
0.944** (full precision 0.9441302, Adj. 0.9429782), **F = 819.6** (819.589),
**Prob(F) = 1.73e-61**, **Df Model = 2** (two predictors now), **Df Residuals =
97** (100 minus 1 intercept minus 2 slopes). All three terms are highly
significant; every `P>|t|` prints `0.000`, and the true p-values are Intercept
**2.02e-15**, x1 **1.36e-52**, x2 **9.93e-40** (a printed `0.000` means "< 0.0005,"
never literally zero, the same rule Module 12 taught). Verified 95% CIs, none
containing 0: Intercept **[3.735, 5.719]**, x1 **[1.843, 2.092]**, x2 **[-3.169,
-2.648]**.

**A slope here is a partial effect, not a marginal one.** Each coefficient is
the change in predicted `y` per one-unit change in that input *holding the other
input fixed*, unlike an `sns.lmplot` of `x1` against `y` alone, which draws the
**marginal** single-variable relationship. Here `x1` and `x2` were generated
independently, so the marginal slopes and the partial coefficients land close
together, but that is not guaranteed once predictors are correlated. That
distinction is exactly why section 3 works so hard to make two coefficients
comparable.

**Seeing the additive structure.** To visualize how predictions move across the
input space, build a grid of every combination of evenly spaced inputs with a
nested comprehension, predict on it, and line-plot with the second input as
color:

```python
viz_grid = pd.DataFrame(
    [(a, b)
     for a in np.linspace(df.x1.min(), df.x1.max(), 101)   # 101 x-axis values
     for b in np.linspace(df.x2.min(), df.x2.max(), 9)],   # 9 line colors
    columns=['x1', 'x2'])
viz_grid['pred'] = model.predict(viz_grid)
sns.relplot(data=viz_grid, x='x1', y='pred', hue='x2', kind='line')
```

`np.linspace(start, stop, N)` makes `N` evenly spaced values including both
endpoints; **101 values of `x1` crossed with 9 values of `x2` = 909 rows**
(Verified). The lab's rule of thumb is roughly 5 to 10 values for the coloring
variable so the plot does not drown in lines. The payoff is **9 parallel
lines**: at fixed `x1`, each step up in `x2` of about 0.623 drops the prediction
by about 1.81, which is exactly the slope times the step (-2.9085 x 0.623),
Verified against the first grid predictions (4.7692 down to 2.9573, a 1.8119
drop). `x2` changes only each line's height, never its slope. **Parallel lines
are the visual fingerprint of a purely additive model.** Remember that picture:
section 6 breaks it on purpose.

---

## 3. Standardize: making coefficients comparable

Section 2 left a temptation. If both inputs sit in the same additive model, why
not just compare their coefficients to decide which matters more? Because a
coefficient is "change in `y` per **one unit** of `x`," and one unit means
something wildly different across variables on different scales. This is a real
trap, and the penguins dataset shows it biting.

```python
penguins = sns.load_dataset('penguins')   # 344 rows, 7 columns
model = smf.ols('flipper_length_mm ~ bill_length_mm + body_mass_g', data=penguins).fit()
model.params
```

`bill_length_mm` runs about 32 to 60 mm; `body_mass_g` runs about 2700 to 6300
grams. (A boxplot of all numeric columns, `sns.catplot(data=penguins,
kind='box', aspect=2)`, makes the mismatch obvious: body mass towers over the
millimeter columns squashed near the bottom.) Verified raw coefficients:

```text
bill_length_mm    0.549224
body_mass_g       0.013051
```

Naively, bill length's coefficient (0.55) is about 40 times body mass's (0.013),
so bill length must matter far more. **That conclusion is wrong.** Body mass's
coefficient looks tiny only because a one-*gram* change is a trivially small
move for a variable that spans thousands of grams. The comparison is an artifact
of units, not of importance.

**The fix is z-standardization:** rescale each input to **mean 0, standard
deviation 1** via `(x - mean) / std`, then every coefficient is on the same
footing. Standardize the **inputs only** (the outcome is left in raw units) and
only the **continuous** ones (categoricals have no meaningful standardization):

```python
penguins_cont = penguins.drop(columns='flipper_length_mm').select_dtypes('number')
scaler = StandardScaler()
penguins_std = pd.DataFrame(scaler.fit_transform(penguins_cont),
                            columns=penguins_cont.columns)
penguins_std['flipper_length_mm'] = penguins['flipper_length_mm']   # raw outcome back in
model = smf.ols('flipper_length_mm ~ bill_length_mm + body_mass_g', data=penguins_std).fit()
model.params
```

`select_dtypes('number')` grabs the continuous columns (`bill_length_mm`,
`bill_depth_mm`, `body_mass_g`; note `bill_depth_mm` rides along harmlessly even
though the model does not use it). `fit_transform` returns a NumPy array, so wrap
it back into a DataFrame with the original column names. Verified standardized
coefficients:

```text
bill_length_mm     2.994150
body_mass_g       10.450819
```

**The ranking flipped.** Body mass (10.45) now clearly beats bill length (2.99):
per one standard deviation of change, body mass moves flipper length far more.
This is the trustworthy answer, and it is the exact opposite of the raw-unit
impression.

**What standardization does NOT change.** It is a *linear* rescaling, so it
cannot alter significance or fit. Verified: the slope p-values are identical to
displayed precision in both models (bill_length **3.31e-11**, body_mass **7.56e-75**), and
**R-squared is identical at 0.7884** in both. You have re-expressed the same fit,
not improved it. One neat side effect: with every input centered at 0, the
standardized **intercept equals the mean of the outcome**, Verified at
**200.915205**, which is exactly `penguins['flipper_length_mm'].mean()`.

**The new reading of a standardized coefficient:** change in the outcome per
**one standard deviation** of the input (not per one raw unit). So +1 SD of body
mass is about +10 mm of flipper, +1 SD of bill length about +3 mm. That "per 1
SD" currency is the whole point: it makes the predictors finally comparable.

> **Exam note (the ddof landmine).** `StandardScaler` divides by the
> **population** standard deviation (ddof = 0, dividing by N), while pandas
> `.std()` and `.describe()` use the **sample** standard deviation (ddof = 1,
> dividing by N-1). So if you call `.std()` on the standardized frame it reads
> about **1.0015**, not exactly 1.0000 (Verified 1.001465). The result is still
> "mean 0, SD 1" for every practical purpose; the tiny gap is only the ddof
> convention and changes no conclusion. (Caveat the note also flags: even after
> standardizing, relative coefficients are untrustworthy if the inputs are
> **correlated**, which is out of scope here but real.)

---

## 4. Encode: categorical inputs as dummy variables

So far every input has been numeric. But regression **multiplies each input by a
coefficient**, and you cannot multiply a coefficient by the word `"setosa"`. The
fix is **feature extraction**: replace a categorical column with **indicator
(dummy) variables**, each a 0/1 feature that is **1 when the row has that value
and 0 otherwise**. Now every feature is a number the model can multiply.

The catch that surprises everyone: for a category with `K` levels you do **not**
make `K` dummies, you make **K minus 1**. The dropped level is the **reference
category**, the stand-in for "zero" that a categorical variable otherwise lacks.
A numeric input has a natural 0; a category does not, so one level is elected to
play that role. When a row is at the reference, all of that variable's dummies
are 0 and the model falls back to the intercept. (The precise reason one *must*
be dropped: keeping all `K` dummies would make them sum to 1 and be perfectly
collinear with the intercept, the "dummy variable trap," leaving the fit
unidentifiable.)

Statsmodels does all of this automatically. Write the formula exactly like a
numeric one; it detects the `object`-dtype column, builds the dummies, and drops
one level. The demo is the classic **iris** dataset (150 rows, 50 per species),
predicting petal length from species:

```python
iris = sns.load_dataset('iris')          # 150 rows, species: setosa/versicolor/virginica
model = smf.ols('petal_length ~ species', data=iris).fit()
model.params
```

Verified output:

```text
Intercept                1.462
species[T.versicolor]    2.798
species[T.virginica]     4.090
```

One input produced **two coefficients**, and **setosa is missing**. That is not
a bug: setosa is the **reference**, chosen because statsmodels drops the level
that is **first alphabetically** (setosa < versicolor < virginica). The `[T.`
in `species[T.versicolor]` marks a **Treatment contrast**, statsmodels' name for
standard reference/dummy coding; the term is exactly the 0/1 dummy "is this row
versicolor?"

**Reading a categorical coefficient (the landmine).** It is **not** a slope. A
dummy coefficient is how far that level's average output sits from the
reference level's average output:

```text
coefficient(level) = mean(output at that level) - mean(output at the reference)
```

Confirm it against the group means, `iris.groupby('species')['petal_length'].mean()`
(Verified setosa **1.462**, versicolor **4.260**, virginica **5.552**):

- **Intercept 1.462 = setosa's mean.** With a single categorical input, the
  intercept *is* the reference category's average output.
- **versicolor coef 2.798 = 4.260 - 1.462** (Verified exactly).
- **virginica coef 4.090 = 5.552 - 1.462** (Verified exactly).

So a category's own prediction is intercept plus its dummy, which is just its
group average (versicolor: 1.462 + 2.798 = 4.260). Verified end to end:
`model.predict(pd.DataFrame({'species': ['setosa','versicolor','virginica']}))`
returns exactly **1.462, 4.260, 5.552**, the three group means. A categorical
regression predicts each category's training-set average.

> **Exam note (transcript garble).** The lecture audio says "4.26 minus the
> reference category, so the average value of the reference category, that's
> going to be 2.798." Read past it. **2.798 is the coefficient (the
> difference)**, not the reference category's mean. The reference (setosa) mean
> is **1.462**. The correct sentence: versicolor mean 4.260 minus setosa mean
> 1.462 equals the coefficient 2.798.

Significance is also **relative to the reference**: each p-value tests whether
that level's average differs from setosa's. Verified `model.pvalues`: versicolor
**5.25e-69**, virginica **4.11e-91**, Intercept **9.30e-53** (the lecture's
offhand "10 to the negative 53" matches the intercept's p-value; the species
terms are even tinier). `model.pvalues < 0.05` returns all `True`. And the model
overall is strong: Verified **R-squared 0.941**, **F 1180**, **Prob(F)
2.86e-91**, with 95% CIs that all exclude 0 (Intercept [1.342, 1.582],
versicolor [2.628, 2.968], virginica [3.920, 4.260]). Species barely overlap, so
species alone explains about 94% of petal-length variance.

You can force categorical treatment or change the reference with the `C()`
wrapper: `C(species)` reproduces the bare-form fit, and
`C(species, Treatment(reference='virginica'))` drops virginica instead
(Verified). These are the columns standardization (section 3) deliberately
skipped with `select_dtypes('number')`: dummies and standardization handle the
two different input types.

---

## 5. Interact: when one input bends another's slope

Section 2's additive model made a strong promise: the effect of one input never
depends on the other, which is why its grid lines were parallel. Often that
promise is false. An extra square meter of apartment is worth **more** in the
city center than in the suburbs; cold and rain together may feel worse than the
sum of each; more viral exposure raises risk faster for an immunocompromised
person. The tool for "the effect of one variable depends on the value of
another" is an **interaction feature**, which **multiplies two inputs together**
instead of just adding them. The stand-alone pushes are still the **main
effects**; the product is the extra push that appears only when both move
together.

The demo dataset is apartment prices, read straight from a course GitHub URL:

```python
url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(url)   # 150 rows, no missing values
```

Verified structure: **150 rows, 3 columns, no missing values**, with dtypes
**`size_m2` float64** (continuous), **`city_center` bool** (80 True, 70 False),
**`price_usd` int64** (the target). Because `city_center` is a `bool`,
statsmodels auto-encodes it exactly like the section-4 categorical: `False`
becomes the reference and the term prints as **`city_center[T.True]`**, the same
`[T.` treatment coding. So this section reuses everything from section 4, just
with a two-level boolean.

Statsmodels gives **three equivalent ways** to write the same interaction model,
and all three produce **identical coefficients** (Verified byte-for-byte):

```python
smf.ols('price_usd ~ size_m2 + city_center + size_m2:city_center', data=data).fit()  # spelled out
model = smf.ols('price_usd ~ size_m2 * city_center', data=data).fit()                # the shortcut
smf.ols('price_usd ~ (size_m2 + city_center)**2', data=data).fit()                   # degree-capped
```

- **`a:b`** (colon) is the **interaction term only**, the bare product. You
  almost never want it alone.
- **`a * b`** expands to **`a + b + a:b`**: both main effects plus their
  interaction. This is the everyday syntax.
- **`(a + b)**2`** is main effects plus interactions **up to degree 2**; the
  `**n` caps the interaction *order*, not an exponent. With only two inputs all
  three coincide, but with three or more the degree matters: `a*b*c` includes
  the 3-way `a:b:c`, whereas `(a+b+c)**2` stops at the three 2-way terms.
  Higher-order interactions get hard to interpret, so capping is common.

Verified coefficients for `price_usd ~ size_m2 * city_center`:

```text
Intercept                      31586.409292
city_center[T.True]            92869.531786
size_m2                         2996.992135
size_m2:city_center[T.True]      870.572873
```

Read against the `False` baseline:

- **Intercept 31586.41**: the price of a hypothetical 0-m2, non-city-center
  apartment (a geometric anchor, not a real listing).
- **city_center[T.True] 92869.53**: a **difference of intercepts**. Being
  downtown shifts the line up by about $92.9k at the same size.
- **size_m2 2996.99**: the size slope for the **baseline** group. Outside the
  center, each extra square meter adds about $2,997.
- **size_m2:city_center[T.True] 870.57**: the interaction, a **difference of
  slopes**. Inside the center, each square meter adds about $871 *more* on top
  of the base $2,997.

So the two group lines have **different slopes**:

```text
Not city center:  price = 31586.41            + 2996.99 x size_m2
City center:      price = (31586.41+92869.53) + (2996.99+870.57) x size_m2
                        = 124455.94           + 3867.57 x size_m2
```

Verified: the downtown slope is exactly 2996.99 + 870.57 = **3867.57**. Every
term is significant (Verified `model.pvalues`: Intercept **0.0385**,
city_center **1.86e-05**, size_m2 **1.99e-27**, interaction **6.37e-03**;
`model.pvalues < 0.05` all `True`). The interaction's p = 0.0064 < 0.05 means
the effect really is more than additive, not just noise. Overall fit (Verified):
**R-squared 0.878** (Adj. 0.876), **F 351.8**, **Prob(F) 1.38e-66**, **Df Model
3** (three predictor terms plus the intercept), **n = 150**.

---

## 6. Visualize: fanning lines are the interaction

Section 2 showed additive effects as **parallel** grid lines. The whole visual
point of section 5 is that an interaction **breaks the parallelism**. The
technique is the same prediction grid, now crossing the numeric axis with both
values of the group:

```python
viz_grid = pd.DataFrame(
    [(s, c)
     for s in np.linspace(data.size_m2.min(), data.size_m2.max(), 101)  # 101 sizes, 25.0 to 129.1
     for c in [True, False]],                                            # both groups
    columns=['size_m2', 'city_center'])
viz_grid['pred'] = model.predict(viz_grid)
sns.relplot(data=viz_grid, x='size_m2', y='pred', kind='line', hue='city_center')
```

`np.linspace(25.0, 129.1, 101)` makes 101 evenly spaced sizes (Verified step
about 1.041, both endpoints included) crossed with `[True, False]` = **202 rows**
(Verified). The grid columns must be named exactly like the training columns so
`model.predict` can match them; `kind='line'` traces the fitted relationships
and `hue='city_center'` draws one line per group. Verified predictions confirm
the two-line algebra above: at size 25, the city line predicts **$221,145** and
the non-city line **$106,511**; at size 129.1, **$623,759** versus **$418,498**.
The city-center line **starts higher and climbs faster**, so the two lines
**fan apart** as size grows. **Non-parallel (fanning) lines = interaction;
parallel lines = additive (no interaction).** That is the fast visual test for
whether an interaction is present and matters, and it is the direct contrast to
section 2's parallel picture.

> **Exam note (the t-stat vs coefficient landmine).** The lab describes the
> interaction as "positive (2.78) and significant." The **coefficient** is
> **870.57** (that is what "positive" means, and it is the effect size: about
> +$870.57 per m² of extra downtown steepness). The number **2.78** is the
> **t-statistic** (Verified 2.768, rounding to 2.77), **not** the coefficient.
> Do not read "2.78" as the size of the effect. The full row is coefficient
> 870.57, t = 2.77, p = 0.006. (A companion garble worth carrying: the lab's
> markdown says "from the summary you can see," but the code actually pulls
> significance from `model.pvalues`; the values are the same either way, since
> those p-values are the summary's `P>|t|` column.)

---

## 7. Transform: curves without leaving linear regression

Every model so far has drawn straight lines. The last move bends them, and the
surprise is that it is still linear regression. **"Linear" means linear in the
coefficients (β), not in the input `x`.** A model like

```text
ŷ = β₀ + β₁·sin(x)
```

is linear because β₁ multiplies `sin(x)` and does not sit *inside* the sine. So
you can fit curved data by first passing the raw input through a nonlinear
**feature transformation**, then fitting an ordinary least squares line to the
transformed feature. Statsmodels lets you write the transform **directly inside
the formula string**, with NumPy functions evaluated inline against the data.

> **Note (no random seed).** These two examples build data with
> `np.random.normal` and **never call `np.random.seed`**, so every coefficient,
> R-squared, t, and p shifts a little each run. The saved notebook values below
> are one such run; the rest are reported as behavior-level ranges, Verified by
> re-running, never as falsely exact digits.

**Example 1, a sine wave.** Data is `sin(x) + Normal(0, 0.2)` over x in [0, 7],
100 points (true slope 1, true intercept 0):

```python
data = pd.DataFrame({'x': np.linspace(0, 7, 100)})
data['y'] = np.sin(data['x']) + np.random.normal(0, 0.2, size=100)
model = smf.ols('y ~ np.sin(x)', data=data).fit()
```

Saved notebook run: `np.sin(x)` coefficient **1.013**, intercept **-0.017** (p =
0.370, correctly **not** significant, since the true intercept is 0), R-squared
**0.932**. **Verified (behavior):** across fresh unseeded runs the sine
coefficient lands near **1** (my re-runs about 0.94 to 1.06) and R-squared near
**0.93** (about 0.90 to 0.94); seeded runs (seeds 0/1/42) give slopes about
**1.06 / 0.96 / 0.95** with R-squared about 0.93 each. The slope always recovers
the true 1 and the intercept is reliably insignificant. A non-significant
intercept here is the honest, correct result, not a defect.

**Example 2, a parabola.** Data is `2 + 0.8·x² + Normal(0, 0.5)` over x in
[-3, 3], 100 points (true intercept 2, true x² coefficient 0.8):

```python
data = pd.DataFrame({'x': np.linspace(-3, 3, 100)})
data['y'] = 2 + 0.8*(data['x']**2) + np.random.normal(0, 0.5, size=100)
poly_model = smf.ols('y ~ np.power(x, 2)', data=data).fit()
```

Saved notebook run: intercept **1.909** (true 2), `np.power(x, 2)` coefficient
**0.814** (true 0.8), R-squared **0.954**, both terms significant. **Verified
(behavior):** unseeded re-runs give x² coefficient near **0.8** (about 0.76 to
0.87), intercept near **2** (about 1.8 to 2.2), R-squared near **0.95** (about
0.94 to 0.965); seeded runs (0/1/42) give x² coefficients about **0.87 / 0.79 /
0.81**, intercepts near 2. The fit always recovers the generating (2, 0.8).

> **Exam note (the patsy `x**2` landmine).** You might expect `'y ~ x**2'` to
> give a quadratic. It does **not**. In the formula (patsy) mini-language, `**`
> means **interaction expansion**, not exponentiation, and a variable interacted
> with itself is just itself, so `'y ~ x**2'` collapses to plain `'y ~ x'`.
> Verified (behavior) on the parabola data: fitting `'y ~ x**2'` yields a
> useless **R-squared near 0.0001** (about 7.9e-05) with only an `x` term, while
> `'y ~ np.power(x, 2)'` yields **R-squared about 0.96**. To raise a variable to a
> power inside a formula you must call a real function, **`np.power(x, 2)`**, or
> wrap the arithmetic in the identity operator, **`I(x**2)`**; the two give
> byte-for-byte identical fits (Verified). This is why the notebook uses
> `np.power`.

**Interpretation shifts under a transform**, another quiz trap: the coefficient
is no longer "change in output per one-unit rise in raw `x`," it is "change in
output per one-unit rise in the **transformed feature**" (per unit of `sin(x)`,
or of `x²`). And to draw the fitted curve you reuse the exact grid-and-predict
recipe from section 6: build a dense `np.linspace` of x-values, call
`model.predict` (which re-applies the stored transform, so you pass raw `x`),
and line-plot the predictions over the raw scatter. The result is a smooth curve,
not a straight line, which is the visible payoff. The closing cheat sheet of
shape to transform: **wave or cyclical -> `sin(x)` or `cos(x)`**; **simple
U-shaped curve -> `x²` via `np.power(x, 2)`**; **complex curve -> higher-order
polynomials such as `x³`**.

---

## One-paragraph recap

Module 13 keeps Module 12's `smf.ols('y ~ ...').fit()` untouched and makes only
the right-hand side richer, all while staying linear regression because a linear
model is a sum of coefficients times features and the features can be anything
numeric. **Add:** `'y ~ x1 + x2'` sums two independent contributions (on seed-2100
synthetic data the fit recovers the true 5/+2/-3 as Intercept 4.73, x1 1.97, x2
-2.91, R-squared 0.944), and a prediction grid draws **parallel** lines, the
signature of no interaction; a slope is a **partial** effect (holding the other
input fixed), not the marginal single-variable one. **Standardize:** raw
coefficients are not comparable across scales, so `StandardScaler` rescales
inputs to mean 0 / SD 1, which flips the penguins ranking from a misleading
bill_length 0.55 > body_mass 0.013 to the true body_mass 10.45 > bill_length
2.99, leaving p-values (3.31e-11, 7.56e-75) and R-squared (0.7884) untouched and
making the intercept the outcome mean (200.915), with the ddof gotcha that a
standardized column's pandas `.std()` reads 1.0015, not 1. **Encode:** a
categorical enters as **K minus 1 dummies** against an alphabetical reference
(iris `petal_length ~ species` drops setosa), the intercept is the reference
mean (1.462) and each dummy is that level's mean minus the reference (versicolor
2.798 = 4.260 - 1.462), read relative to the reference and never as a slope.
**Interact:** `'price_usd ~ size_m2 * city_center'` (= `a + b + a:b`, same as
`a:b` spelled out or `(a+b)**2`) lets one input change another's slope, so the
apartment interaction 870.57 (p 0.006) tilts the downtown price/m² from $2,997 to
$3,868 and shifts the intercept up $92.9k; watch the landmine that its **t-stat
2.78 is not the coefficient 870.57**. **Visualize:** the same 202-row grid plus
`model.predict` and `hue='city_center'` draws two **fanning** (non-parallel)
lines, the eye-test for interaction versus additive. **Transform:** `'y ~
np.sin(x)'` and `'y ~ np.power(x, 2)'` fit curves by transforming the input
first (unseeded data recovers slope near 1 and x² near 0.8 at R-squared near 0.93
and 0.95), interpreting each coefficient per unit of the *transformed* feature,
and dodging the patsy landmine that `'y ~ x**2'` collapses to `'y ~ x'` (R-squared
0.0001) so you must use `np.power(x, 2)` or `I(x**2)`.

---

## Errata: the recovered lab recording

After this weave shipped, the transcript of the **recorded lab session** behind
section 5's interaction notebook surfaced (it had been left out of the module by
a save-directory error). Its cliff note now sits in the module folder as
**"Interaction Features for Linear Regression (Cliff Notes)"**, the spoken
companion to the notebook note. Nothing in sections 1 through 7 changes: the
recording narrates the same lab, and every number in the material it narrates
was re-verified against the same 150-row CSV (the section 5 coefficients, the
identity of the three formula spellings, and all four p-values reproduce
exactly). What follows is what the recording adds or corrects.

- **The module's cliff-note count is now eight, not seven.** The intro
  describes this weave as stitched from "seven Module 13 cliff notes: one
  lecture note ... and six lab notebook notes," which is still true of what
  was stitched. With the recovered recording the module folder now holds
  **eight** cliff notes: two spoken-source notes (the categorical lecture and
  the interaction lab session) plus the six notebook notes.
- **The formula mini-language is R's, on the record.** The instructor states
  that the statsmodels formula interface is "drawn from the R programming
  language." That is why `+`, `:`, `*`, and `**` follow R's formula conventions
  instead of Python arithmetic, and it is the same R-heritage operator behind
  section 7's landmine, where `'y ~ x**2'` collapses to `'y ~ x'` because
  formula `**` means interaction expansion, not exponentiation.
- **The apartment data is synthetic.** The instructor says so in passing while
  reading the fitted parameters. The header's dataset roster can be read
  accordingly: the apartment prices CSV is synthetic like the seed-2100 additive
  data, which is likely why its interaction shows up so cleanly. Same file,
  same URL, so no verified number moves.
- **Spoken guidance now on record** (matching section 5 as written): you usually
  do not want the interaction term alone, you keep the **main effects** and add
  the interaction on top; `a * b` is the everyday shortcut people actually use;
  and `**n` earns its keep only at three or more variables, where it caps the
  expansion at degree n (pairwise for `**2`) so you are not stuck interpreting
  three-way and four-way interactions. The section-5 warm-up intuitions (cold rain feeling
  "doubly terrible," viral exposure biting harder when immunocompromised) are
  also the instructor's own spoken examples in this recording.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
