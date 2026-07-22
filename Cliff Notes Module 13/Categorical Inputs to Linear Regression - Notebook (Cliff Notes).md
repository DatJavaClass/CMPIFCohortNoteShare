# Categorical Inputs to Linear Regression (Lab Notebook)

**Module 13 Cliff Notes** | Source: lab notebook `categorical_inputs_to_linear_regression_start.ipynb`

---

## TL;DR

- Linear regression needs to **multiply each input by a coefficient**, but you cannot multiply a string like `"dog"` or `"setosa"`. The fix is to turn a categorical variable into **indicator (dummy) variables**: one 0/1 feature per category value that is **1 when the row has that value and 0 otherwise**.
- You do **not** make a dummy for every category. One value is picked as the **reference value** (it plays the role of "x = 0"). With `k` categories you get **k - 1 dummies plus the intercept**. `statsmodels` picks the reference **alphabetically first**.
- The notebook fits **`smf.ols(formula='petal_length ~ species', data=iris).fit()`** on the **Iris dataset** (150 rows, 50 per species) to predict **petal length from species** (a categorical predictor).
- **`model.params`** (Verified): `Intercept = 1.462`, `species[T.versicolor] = 2.798`, `species[T.virginica] = 4.090`. The missing species, **setosa**, is the reference (first alphabetically).
- **Coefficients for categories are read relative to the reference.** The intercept is the **average output at the reference** (setosa's mean petal length = 1.462). Each dummy coefficient is **that category's mean minus the reference mean**: `4.260 - 1.462 = 2.798` (versicolor), `5.552 - 1.462 = 4.090` (virginica), all Verified.
- All three p-values are astronomically small (versicolor `5.25e-69`, virginica `4.11e-91`), so every term is **significant at 0.05** (`model.pvalues < 0.05` returns all `True`). Significance is also **relative to the reference**.

> **The one takeaway:** a categorical predictor enters a regression as **k - 1 dummy variables** with one category held out as the **reference**; the **intercept is the reference category's average output**, and each **dummy coefficient is how far that category's average sits above or below the reference**.

---

## The problem: you cannot multiply a word

Linear regression predicts with the same familiar formula:

```text
ŷ = β₀ + β₁x
```

That requires multiplying the input `x` by a coefficient `β₁`. Fine for numbers, impossible for a categorical value like `"dog"`, `"cat"`, or `"fish"`. The solution is **feature extraction**: replace the category with **indicator variables** (also called **dummy variables**). For each possible value of the category, the dummy is **1 if the row equals that value and 0 otherwise**. Now every feature is numeric and can be multiplied by a coefficient.

## Why one category becomes the "reference"

You might expect three dummies for a three-value `pet` variable (`pet=dog`, `pet=cat`, `pet=fish`), but that is **not** what is done. Regression needs a **reference point** where every input is 0 and the prediction collapses to just the intercept `β₀`. Numeric inputs have a natural 0; categories do not, so **we choose one category to act as 0**. That chosen category is the **reference value**. When the variable equals the reference (and all other inputs are 0), the predicted output is exactly the intercept. This is why a `k`-category variable produces **k - 1 dummy variables** (the reference is absorbed into the intercept), which also avoids a redundant, perfectly collinear column.

## Load the data (Iris via seaborn)

The demo uses the **Iris flower dataset**, bundled with seaborn:

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
iris.info()
```

Verified: **150 rows, 5 columns**, no missing values. Four numeric measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, all `float64`) plus one **categorical** column `species` (dtype `object`). The `species` column has three values, **balanced at 50 rows each**: `setosa`, `versicolor`, `virginica`.

Goal: predict **`petal_length`** (the dependent variable / output) from **`species`** (the categorical predictor / independent variable).

## Visualize before fitting

**Violin plot** of petal length by species:

```python
sns.catplot(data=iris, x='species', y='petal_length', kind='violin', height=3)
plt.show()
```

A violin plot shows each species' distribution as a smoothed shape. Verified from the data, the three violins sit at very different heights and barely overlap: setosa is low and tight (petal length ~1.0 to 1.9, mean 1.462), versicolor is in the middle (~3.0 to 5.1, mean 4.260), and virginica is highest and widest (~4.5 to 6.9, mean 5.552).

**Point plot** of the average petal length per species:

```python
sns.catplot(data=iris, x='species', y='petal_length', kind='point', linestyle='none', height=3)
plt.show()
```

The point plot draws each species' **mean** with a **confidence-interval bar**; `linestyle='none'` drops the connecting line so you read the three means as separate dots. Verified: the three means (1.462, 4.260, 5.552) are far apart with **no overlapping confidence intervals**. This matters because **linear regression predicts the average output for a given input**, so clearly separated group means are exactly what a categorical model will capture.

## Fit the model and read `.params`

```python
import statsmodels.formula.api as smf

model = smf.ols(formula='petal_length ~ species', data=iris).fit()
model.params
```

Verified output:

```text
Intercept                1.462
species[T.versicolor]    2.798
species[T.virginica]     4.090
dtype: float64
```

Notice two things. First, the coefficient names are **`species[T.versicolor]`** and **`species[T.virginica]`**: the `T.` marks them as **treatment-coded dummy variables** (each is 1 for that species, 0 otherwise). Second, **setosa is missing**. That is not a bug: **setosa is the reference value**, chosen by statsmodels simply because it is **first alphabetically**. With a single categorical input, the reference category's average output is the **intercept** itself.

## Interpreting categorical coefficients (relative to the reference)

The rule to memorize: **coefficients for categorical values are relative to the reference category.** A dummy coefficient is the **average output for rows with that value minus the average output for reference rows**. Confirm it with a `groupby`:

```python
iris.groupby('species')['petal_length'].mean()
```

Verified output:

```text
species
setosa        1.462
versicolor    4.260
virginica     5.552
Name: petal_length, dtype: float64
```

Line these up against the coefficients (all Verified):

- **Intercept 1.462** = setosa's mean petal length (the reference category's average, the prediction when all inputs are "0").
- **species[T.versicolor] 2.798** = `4.260 - 1.462` = versicolor mean minus setosa mean.
- **species[T.virginica] 4.090** = `5.552 - 1.462` = virginica mean minus setosa mean.

So a **positive** dummy coefficient means that category has a **higher** average output than the reference; a **negative** one means **lower**. To get a category's actual predicted value, add its coefficient back to the intercept (for example versicolor: `1.462 + 2.798 = 4.260`, which equals its group mean). In other words, **the model's prediction for a species is just that species' training-set average petal length**.

## Significance is also relative to the reference

```python
model.pvalues
```

Verified output:

```text
Intercept                9.303052e-53
species[T.versicolor]    5.254587e-69
species[T.virginica]     4.106139e-91
dtype: float64
```

The `e` is scientific notation: `9.3e-53` means 9.3 x 10^-53, i.e. vanishingly small. Each dummy's p-value tests whether that category's average **differs significantly from the reference** (setosa), so significance, like the coefficients, is **relative to the reference**. A quick pass/fail against the 0.05 threshold:

```python
model.pvalues < 0.05
```

Verified output:

```text
Intercept                True
species[T.versicolor]    True
species[T.virginica]     True
dtype: bool
```

All `True`: every term is statistically significant. Both versicolor and virginica have petal lengths that differ significantly from setosa.

## Beyond the notebook (my completion of the trailing empty cell)

The source notebook ends with one **empty code cell** and no further instructions, so there is no assigned exercise; the lab is otherwise complete. As the natural next step (and to mirror how the Module 12 notebook was read), here is the **full fitted-model picture**, Verified by running the same model:

- **R-squared = 0.941** (Adj. 0.941): species alone explains about **94%** of the variance in petal length, unusually high because the three species barely overlap.
- **F-statistic = 1180**, **Prob(F) = 2.86e-91**: the model as a whole is overwhelmingly significant.
- `model.summary()` reports the same three coefficients with 95% confidence intervals that all exclude 0: Intercept `[1.342, 1.582]`, versicolor `[2.628, 2.968]`, virginica `[3.920, 4.260]`.
- **Prediction check:** `model.predict(pd.DataFrame({'species': ['setosa','versicolor','virginica']}))` returns `1.462, 4.260, 5.552`, which are exactly the three group means, confirming that a categorical regression predicts each category's average output.

---

## Quiz-ready facts

- A categorical predictor cannot be multiplied by a coefficient directly, so it is converted into **indicator / dummy variables**: one 0/1 feature per category value (**1** for that value, **0** otherwise).
- With `k` categories you create **k - 1 dummies**, not `k`. One value is the **reference value** (it plays the role of "x = 0" and is folded into the intercept). Dropping it also avoids perfect collinearity.
- `statsmodels` chooses the reference **alphabetically first** (here `setosa`). Dummy coefficients are named like **`species[T.versicolor]`**; the `T.` denotes treatment coding.
- Fit syntax is identical to the numeric case: **`smf.ols(formula='petal_length ~ species', data=iris).fit()`**; statsmodels detects that `species` is categorical (`object` dtype) and builds the dummies for you.
- **Intercept = the reference category's average output** (setosa mean petal length = 1.462, Verified).
- **Each dummy coefficient = that category's mean minus the reference mean**: versicolor `4.260 - 1.462 = 2.798`, virginica `5.552 - 1.462 = 4.090` (Verified). Positive means higher than reference, negative means lower.
- A category's predicted value = intercept + its dummy coefficient = that category's **group average** (e.g. versicolor `1.462 + 2.798 = 4.260`).
- Coefficients **and** their p-values are interpreted **relative to the reference category**.
- p-values (Verified): versicolor `5.25e-69`, virginica `4.11e-91`; `model.pvalues < 0.05` is all `True`, so every term is significant.
- Iris dataset: **150 rows, 50 per species**, three species (`setosa`, `versicolor`, `virginica`); the `point` plot shows their means with **non-overlapping confidence intervals**, and regression predicts those per-group averages.
- (Beyond the notebook) The model's **R-squared = 0.941** (Verified): species explains about 94% of petal-length variance.

---

> **See also:** the companion lecture note "Categorical Inputs to Linear Regression (Cliff Notes)" for the concepts behind this code, and the sibling Module 13 notebook notes "Interaction Features for Linear Regression - Notebook (Cliff Notes)", "Additive Features in Statsmodels - Notebook (Cliff Notes)", "Nonlinear Feature Transformations in Statsmodels - Notebook (Cliff Notes)", "Standardizing Variables for Linear Regression - Notebook (Cliff Notes)", and "Prediction Visualizations for Interaction Features - Notebook (Cliff Notes)" for how these dummy features combine with numeric predictors.

---

*Source: CMPIF2100 lab notebook (personal study use).*
