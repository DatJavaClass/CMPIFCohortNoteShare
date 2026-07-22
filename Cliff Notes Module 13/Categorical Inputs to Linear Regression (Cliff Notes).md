# Categorical Inputs to Linear Regression: dummy variables, reference categories, and reading the coefficients

**Module 13 Cliff Notes** | Source: lecture transcript "Categorical Inputs to Linear Regression"

---

## TL;DR

- Linear regression multiplies each input by a coefficient, so a **categorical variable** (like `pet` = dog/cat/fish) cannot be used directly: you cannot multiply a slope by the word "dog." The fix is to **extract features** from the category.
- The features you extract are **indicator variables** (also called **dummy variables**): one dummy per category value, equal to **1 if the row has that value, 0 otherwise**.
- You do NOT create one dummy per category. For a category with **K** levels you extract **K minus 1** dummies. The level left out is the **reference category** (reference value), the stand-in for "zero" that a categorical variable otherwise lacks.
- **Statsmodels handles all of this automatically.** Write the formula exactly like a numeric one, `"petal_length ~ species"`, and it detects that `species` is categorical, builds the dummies, and drops one level.
- Statsmodels picks the reference level as the **first one alphabetically** (here `setosa`). No coefficient is reported for the reference level, which trips people up at first ("where did my third category go?").
- A dummy's **coefficient is read relative to the reference**: it equals **mean(output for that category) minus mean(output for the reference category)**. With one categorical input, the **Intercept equals the reference category's mean output**.
- Each coefficient's **p-value** tests whether that category's average output is **significantly different from the reference**. Tiny p-values (shown in scientific notation, e.g. `5.25e-69`) mean a strongly significant difference.

> **One-line takeaway:** statsmodels turns a categorical input into K minus 1 dummy variables against a dropped reference level, and each coefficient is that level's mean output minus the reference level's mean output.

---

## The problem: you cannot multiply a slope by "dog"

The numeric case is easy. For a continuous input `x`, the estimate is just `β₁·x`: take the value, multiply by the coefficient, done. But real data is full of **categorical variables**: a `pet` column holding dog, cat, or fish; a `species` column; a city type. There is no sensible way to compute "slope times dog." Categorical inputs are less straightforward, but fully doable.

The trick is to **extract features** from the raw input. You keep the original column but derive extra numeric columns from it that CAN be multiplied and slotted into the linear regression formula.

## Indicator (dummy) variables

The extracted features are **indicator variables**, more often called **dummy variables**. The rule is simple: for a category value, the dummy is

```text
1  if the row's variable equals that value (dog, cat, fish, ...)
0  otherwise
```

So you might expect a separate coefficient for `pet == dog`, `pet == cat`, and `pet == fish`. Just 0/1 flags for "does this row have that value or not." That part is intuitive.

## Why you get K minus 1 dummies, not K (the reference category)

The catch that surprises almost everyone: for a category with three values you do NOT extract three dummies. You always extract **one fewer than the number of levels** (K minus 1). The level you leave out is the **reference category** (or reference value).

Why drop one? Linear regression needs a **reference point where the input is effectively zero**, a null value at which the predicted output is just the intercept. Numeric inputs give that for free (the value 0). A categorical variable has no natural zero, so you **elect one category to play the role of zero**. When a row is at the reference level, all of that variable's dummies are 0, and the model falls back to the intercept.

This is why, the first time you fit one of these, one category seems to vanish: "what is the coefficient for that level?" There isn't one. It is the reference, folded into the intercept.

> **Precise / exam note (why one MUST be dropped):** if you also kept a dummy for the reference level, the K dummies would always sum to 1 and be perfectly collinear with the intercept term (the "dummy variable trap"), leaving the coefficients unidentifiable. Dropping one level fixes the point of comparison and makes the fit solvable. The lecture's "reference point where the value is zero" and this collinearity explanation are two ways of saying the same thing.

## Running example: iris petal length by species

The lecture loads the classic **Iris** machine learning dataset, which ships with **seaborn**. It has **150 rows** describing flowers: `petal_length`, `petal_width`, `sepal_length`, `sepal_width`, and a categorical `species` column.

```python
import seaborn as sns
import statsmodels.formula.api as smf

iris = sns.load_dataset("iris")   # 150 rows x 5 columns
```

Pretend you are a biologist asking **how petal length varies by species**. In regression terms:

- **Dependent variable (output):** `petal_length`
- **Independent variable (input):** `species` (categorical)

> **Precise-code note:** the seaborn column is spelled `petal_length` with an underscore, and `species` is stored as a plain string (`object` dtype). The three levels are `setosa`, `versicolor`, `virginica`. (Verified.)

## EDA first: violin plot then point plot

Always visualize before modeling.

- A **violin plot** of petal length split by species shows the **shape of each distribution**. Setosa has short petals, versicolor is mid-range, virginica is longest, so there is real variation to model.
- A **point plot** shows each species' **average** petal length with confidence intervals. Linear regression always predicts the **average output**, so this is the quantity that matters. The three averages are clearly separated and the **confidence intervals do not overlap**, which foreshadows large, significant coefficients.

## Fitting the model: nothing new to type

The formula looks exactly like the numeric case. You do not flag the input as categorical; statsmodels figures it out.

```python
model  = smf.ols(formula="petal_length ~ species", data=iris)
result = model.fit()
result.params
```

Verified output:

```text
Intercept                1.462
species[T.versicolor]    2.798
species[T.virginica]     4.090
```

One input variable produced **two coefficients**, because statsmodels extracted the dummy features for you. The third level, **`setosa`, is the reference** and gets no coefficient. Statsmodels chose it because it is **first alphabetically** (setosa < versicolor < virginica). There are ways to change the reference if you want (see the note below).

> **Precise-code note (the `[T.` naming):** the term names read `species[T.versicolor]`. The `T.` marks a **Treatment contrast**, statsmodels' name for standard reference / dummy coding. `species[T.versicolor]` is exactly the 0/1 dummy "is this row versicolor?"

## Interpreting a categorical coefficient

For a numeric input, a coefficient is a slope: "+1 in x moves the output by the coefficient." That reading does not apply here. Instead:

**Categorical coefficients are relative to the reference category.** A coefficient is how much a level's average output **differs from the reference level's average output**. Precisely:

```text
coefficient(level) = mean(output | rows at that level)
                   - mean(output | rows at the reference level)
```

The lecture confirms this by hand with group means:

```python
iris.groupby("species")["petal_length"].mean()
```

Verified:

```text
setosa        1.462
versicolor    4.260
virginica     5.552
```

Check the arithmetic:

- **Intercept 1.462 = setosa's mean** (with a single categorical input, the Intercept IS the reference category's average output).
- **versicolor coef 2.798 = 4.260 minus 1.462** (versicolor mean minus setosa mean). Verified exactly.
- **virginica coef 4.090 = 5.552 minus 1.462.** Verified exactly.

So a large coefficient means that level's average output is far from the reference; a near-zero coefficient would mean "barely different from the reference, does not matter much."

> **Transcript garble / correction:** the audio says "4.26 minus the reference category, so the average value of the reference category, that's going to be 2.798." Read past it: **2.798 is the coefficient (the difference)**, not the reference category's mean. The reference (setosa) mean is **1.462**. The correct sentence is: versicolor mean 4.26 minus setosa mean 1.462 equals the coefficient 2.798.

## Significance: p-values against the reference

Each coefficient carries a **p-value** testing whether that level's average output is **statistically significantly different from the reference level**. In this dataset the p-values are minuscule, so being versicolor or virginica means a significantly different petal length than setosa.

> **Read scientific notation:** Python prints tiny p-values with `e`, where `e-NN` means "times 10 to the power minus NN." For example `5.25e-69` is 5.25 x 10^-69, effectively zero. (The lecture mentions "10 to the negative 53"; that magnitude matches the **Intercept's** p-value here, `9.30e-53`, while the species coefficients are even tinier at `e-69` and `e-91`. All are wildly significant.)

Quick true/false significance check at the usual 0.05 threshold, applied across the whole pandas Series at once:

```python
result.pvalues < 0.05
```

```text
Intercept                True
species[T.versicolor]    True
species[T.virginica]     True
```

## Optional: C() and changing the reference (companion-lab territory)

The lecture keeps the formula bare (`species`), but statsmodels also accepts an explicit **`C()`** wrapper to force a column to be treated as categorical, and a `Treatment(reference=...)` argument to choose the dropped level.

```python
smf.ols("petal_length ~ C(species)", data=iris)                              # same result as bare `species`
smf.ols("petal_length ~ C(species, Treatment(reference='virginica'))", data=iris)  # virginica becomes the reference
```

Verified: `C(species)` gives coefficients identical to the bare form; setting `Treatment(reference='virginica')` drops virginica instead and reports coefficients for setosa and versicolor.

---

## Quiz-ready facts

- **Why categoricals need special handling:** you cannot multiply a coefficient by a category label; you must **extract numeric features** from the column first.
- **Dummy / indicator variable:** equals **1 if the row has that category value, 0 otherwise**.
- **K minus 1 rule:** a category with **K** levels yields **K minus 1** dummies. The dropped level is the **reference category (reference value)**.
- **Why drop one:** the categorical variable has no natural zero; the reference plays the role of "input is zero," at which the prediction is just the intercept. (Keeping all K dummies would be perfectly collinear with the intercept, the dummy variable trap.)
- **Statsmodels is automatic:** write `"output ~ catvar"` like any formula; it detects the categorical column, builds the dummies, and drops one level. Optionally wrap with **`C(catvar)`** to force categorical treatment.
- **Reference choice:** statsmodels drops the **first level alphabetically** by default (here `setosa`); use `C(col, Treatment(reference='...'))` to change it.
- **Term naming:** coefficients appear as `col[T.level]`; `T.` denotes the Treatment (reference) contrast, i.e. the 0/1 dummy for that level.
- **Coefficient meaning:** `coef(level) = mean(output at level) - mean(output at reference)`. It is a **difference from the reference**, not a slope.
- **Intercept with one categorical input = reference category's mean output.** (Iris: Intercept 1.462 = setosa mean.)
- **Verified iris fit** (`petal_length ~ species`): Intercept **1.462**, versicolor **2.798**, virginica **4.090**; group means setosa **1.462**, versicolor **4.260**, virginica **5.552**; 2.798 = 4.260 - 1.462 and 4.090 = 5.552 - 1.462.
- **P-values** test each level's difference **from the reference**; scientific notation `e-NN` means x 10^-NN (tiny = highly significant). Test at 0.05 with `result.pvalues < 0.05`.
- **Regression predicts averages:** every coefficient and the intercept describe **average** output values, which is why the point plot of group means is the right EDA.

---

> **See also:** "Categorical Inputs to Linear Regression - Notebook (Cliff Notes)" (the companion lab that runs this iris code end to end), "Additive Features in Statsmodels - Notebook (Cliff Notes)" and "Interaction Features for Linear Regression - Notebook (Cliff Notes)" (combining this categorical input with other predictors, the "multiple inputs" the lecture defers), and "Standardizing Variables for Linear Regression - Notebook (Cliff Notes)" (preparing mixed numeric inputs alongside categoricals).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
