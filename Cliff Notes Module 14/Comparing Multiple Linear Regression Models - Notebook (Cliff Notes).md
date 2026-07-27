# Comparing Multiple Linear Regression Models (Lecture Notebook)

**Module 14 Cliff Notes** | Source: lecture notebook `Module-14_Comparing_Multiple_Linear_Regression_Models.ipynb`

---

## TL;DR

- Modeling is **iterative**: you fit **several** models on the same data, score each one, and compare. Same technique (linear regression), different **features** (raw inputs, combinations of inputs, or transformations of inputs).
- The workflow is **one repeating block per model**: name it, write the formula, `fit()`, pull **R-squared**, compute **RMSE** from the predictions, append a small dict to a running list. Build a DataFrame from that list at the end.
- Always include an **intercept-only baseline** (`y ~ 1`). It ignores the input entirely and predicts the **mean of `y`** for every row, so its **R-squared is 0 by construction** and its **RMSE is the population standard deviation of `y`**. Every later model has to beat it or it learned nothing.
- Five models on 100 seeded points of `y = sin(x) + noise` (Verified, `np.random.seed(2100)`), as R-squared / RMSE: **intercept `y ~ 1`** 0.000 / 0.761, **linear `y ~ x`** 0.268 / 0.651, **quadratic `y ~ x + np.power(x, 2)`** 0.325 / 0.625, **cubic** (adds `np.power(x, 3)`) 0.766 / 0.368, **sine `y ~ np.sin(x)`** 0.783 / 0.354.
- **Higher R-squared, lower RMSE = better fit** to the data being scored. The **sine** model wins (it matches how the data was generated) and the **cubic** is a close runner-up; the sine model gets there with only **2** fitted coefficients against the cubic's **4**.
- Formula gotcha: to square a feature use **`np.power(x, 2)`**, never `x**2`. In the R-style formula mini-language `**` means **interaction degree**, so `y ~ x + x**2` silently fits **plain `y ~ x`** (Verified: params are just `Intercept` and `x`).
- Everything here is scored on the **training data**, which is exactly the weakness the rest of the module fixes with cross-validation.

> **The one takeaway:** fit many models, score them all the same way into one table (baseline first), and pick the winner on R-squared and RMSE, but remember these are **training-data** scores, so they reward complexity and cannot tell you how the model behaves on data it has never seen.

---

## Why compare at all

Data scientists rarely fit just one model. Predictive modeling is iterative, and it loops back to EDA and even data cleaning when problems surface. Even inside a single technique (linear regression) you can build very different models by changing the **features**: the inputs you hand the model, which may be raw input variables, combinations of them, or transformations of them.

So the question stops being **"is my model good"** and becomes **"which of these models is best, and is any of them better than doing nothing"**. Answering it needs three things: more than one candidate, one shared scoring recipe, and a baseline to beat.

## The data: a seeded sine wave with noise

```python
np.random.seed(2100)
data = pd.DataFrame({'x': np.linspace(0, 7, 100)})
noise = np.random.normal(0, 0.4, size=len(data))
data['y'] = np.sin(data['x']) + noise
sns.relplot(data=data, x='x', y='y')
```

Verified: **100 rows**, `x` evenly spaced from **0 to 7** (step about 0.0707), `y` = `sin(x)` plus Gaussian noise with requested sigma **0.4**. Seven radians is about **1.11 full sine periods** (Verified), so the underlying wave rises to a peak near `x = 1.57`, crosses zero near `x = 3.14`, bottoms out near `x = 4.71`, and is climbing again at `x = 7`; the noisy scatter follows that path loosely. The key structural fact is that the curve has exactly **two turning points**, which is why a cubic (two turning points at most) can imitate it so well later on.

Because the seed is set, every number below is reproducible, with one floating-point-dust exception flagged in the Model 0 section.

> **Why generate the data?** The true relationship is known, so you can watch each model get closer to a truth you already have. The **sine** model is the "right answer" planted on purpose.

## The scoring recipe, repeated once per model

Every model cell in the notebook is the same short block with a different formula (the `rmse` import happens once, in the Model 0 cell, and the notebook `print`s both metrics):

```python
from statsmodels.tools.eval_measures import rmse   # once, in the Model 0 cell

model_name = 'linear'
model_formula = 'y ~ x'

model = smf.ols(model_formula, data=data).fit()
model_rsquared = model.rsquared

preds = model.predict(data['x'])
model_rmse = rmse(data['y'], preds)
```

Two metrics, two different sources:

- **R-squared** comes free off the fitted model as **`model.rsquared`**. It is the share of the variance in `y` the model explains, unitless, and on training data it runs 0 to 1.
- **RMSE** is computed by hand from predictions with **`rmse(actual, predicted)`** out of `statsmodels.tools.eval_measures`. It is `sqrt(mean((actual - predicted) ** 2))` (Verified from the function source), so it is a **typical error size in the units of `y`**, and **lower is better**.

`model.predict(data['x'])` returns a Series of fitted values, one per row. Verified: for these models it equals both `model.predict(data)` and `model.fittedvalues`, so the Series-versus-DataFrame detail does not change any number here.

## Model 0: the intercept-only baseline

```python
model_formula = 'y ~ 1'
```

The formula `y ~ 1` estimates **an intercept and no slope**, so it ignores `x` completely and predicts the same number for every row. Verified: that number is **0.036733**, which is exactly `data['y'].mean()`. The notebook asks the reader to guess what `preds` holds; the answer is **100 identical copies of the mean of `y`**.

Verified (fresh run):

```text
R squared: 2.220446049250313e-16     (that is 0, to floating-point dust)
RMSE: 0.7607764568100656
```

Two facts worth memorizing:

- **R-squared is 0 by construction.** R-squared measures improvement over predicting the mean, and this model *is* predicting the mean, so there is nothing to improve on. The tiny `2.22e-16` is rounding noise, not a real value.
- **The baseline RMSE equals the population standard deviation of `y`** (Verified: `data['y'].std(ddof=0)` = 0.7607764568100657, an exact match). Any later model whose RMSE is not clearly below **0.761** has learned nothing.

> **Reproducibility note:** the notebook's saved output shows `-2.220446049250313e-16` and the rounded table shows `-0.000`, while a fresh run on numpy 2.3.5 / statsmodels 0.14.5 gives the same magnitude with a **positive** sign (`0.000`). Same value, different floating-point summation order. Read both as **zero**.

## Models 1 to 3: straight line, quadratic, cubic

```python
'y ~ x'                                        # linear
'y ~ x + np.power(x, 2)'                       # quadratic
'y ~ x + np.power(x, 2) + np.power(x, 3)'      # cubic
```

Verified scores: linear **R-squared 0.268128, RMSE 0.650841**; quadratic **0.324706 / 0.625178**; cubic **0.766181 / 0.367872**.

The straight line can only tilt, so on a wave it captures the overall downward drift (Verified slope **-0.193**) and little else. The quadratic gets **one** turning point, enough to curve but not enough to trace both the hump and the trough, hence the modest gain. The cubic gets **two** turning points, exactly matching the shape of the data, and the score jumps hard.

**The `x**2` trap.** The notebook flags it and it is worth the ink: in a statsmodels formula, `**` does not mean exponentiation. It is inherited from R's formula language, where `(a + b)**2` means "all interactions up to degree 2". Verified consequence:

```python
list(smf.ols('y ~ x + x**2', data=data).fit().params.index)
# -> ['Intercept', 'x']          the squared term silently vanishes
```

`x**2` expands to the interaction of `x` with itself, which is just `x`, so the term collapses into the existing one and you get a **plain linear model with no error message**. Use **`np.power(x, 2)`** to actually square a feature. (`I(x**2)`, the formula language's "treat this as literal arithmetic" escape, also works and fits an identical model, Verified, but the notebook teaches `np.power` and that is what to write on an exam.)

> **Why `np.` works inside a quoted formula:** the formula string is not parsed by statsmodels alone; the pieces inside it are **evaluated in the namespace where you called `smf.ols`**. That is why the notebook's very first cell importing `numpy as np` is load-bearing. Verified: run the identical `'y ~ x + np.power(x, 2)'` in a session where numpy was imported under a different alias and it raises `PatsyError: Error evaluating factor: NameError: name 'np' is not defined`.

## Model 4: the sine model

```python
model_formula = 'y ~ np.sin(x)'
```

This applies `np.sin` to the input and regresses `y` on the transformed feature, so it has just **two coefficients**: an intercept and a slope on `sin(x)`. Verified fit: **Intercept -0.001026, np.sin(x) 0.991702**, which is a near-perfect recovery of the true generating rule (`y = 0 + 1.0 * sin(x) + noise`).

Verified scores: **R-squared 0.782898, RMSE 0.354477**, the best of the five.

That RMSE lands somewhere meaningful. The realized standard deviation of the noise actually drawn was **0.354522** (Verified), and the sine model's training RMSE is **0.354477**, essentially the same number. The model has captured the signal, and what it leaves behind is basically just the noise, which is the best an **honest** model can do here.

Note the hedge. Training RMSE can be driven *below* that level, but only by fitting the noise itself: a degree-8 polynomial on this same data reaches **R-squared 0.798768, RMSE 0.341276** (Verified), beating the sine model on both training metrics while being a worse description of the process that actually generated the data (which we know for certain is a pure sine). A training-data leaderboard cannot tell those two apart, which is the whole problem.

> **Source correction (math typo):** the notebook heads this section with `y = beta_0 + sin(beta_1)`, which would put the coefficient *inside* the sine and drop `x` entirely. The model actually fitted is `y = beta_0 + beta_1 * sin(x)`, the coefficient multiplying the transformed input, and the printed parameter name `np.sin(x)` confirms it (Verified). Exam safety: if a quiz keys to the notebook's printed formula, answer its way, but the fitted model is the multiplied form.

## Collecting the results: the list-of-dicts pattern

The notebook's storage idiom is worth copying wholesale (the dict variable is renamed here for clarity, see the source quirk below):

```python
model_performance_lines = []            # once, before the first model

model_eval = {'name': model_name,
              'formula': model_formula,
              'r_squared': model_rsquared,
              'rmse': model_rmse,
             }
model_performance_lines.append(model_eval)      # once, after each model

model_performance = pd.DataFrame(model_performance_lines)   # once, at the end
```

**Build a list of dictionaries, then make the DataFrame at the end.** Growing a DataFrame row by row is the wrong instinct: it is slow, and `DataFrame.append` was removed outright in pandas 2.0 (Verified: `hasattr(pd.DataFrame, 'append')` is `False` on pandas 2.3.3). Appending dicts to a plain list has neither problem, and `pd.DataFrame(list_of_dicts)` turns the dict keys into columns in one call. Verified: `len(model_performance_lines)` grows 1, 2, 3, 4, 5 as the models are added, which is the notebook's running sanity check.

## Reading the comparison table

```python
model_performance.round(3)
```

Verified:

```text
        name                                  formula  r_squared   rmse
0  intercept                                    y ~ 1      0.000  0.761
1     linear                                    y ~ x      0.268  0.651
2  quadratic                   y ~ x + np.power(x, 2)      0.325  0.625
3      cubic  y ~ x + np.power(x, 2) + np.power(x, 3)      0.766  0.368
4       sine                            y ~ np.sin(x)      0.783  0.354
```

(The notebook's own saved run prints `-0.000` in the intercept row; same value, see the reproducibility note above.)

`.round(3)` is purely cosmetic. The raw table prints the R-squared column in scientific notation (`2.681277e-01`) because the near-zero baseline value forces pandas into that format for the whole column. Verified: drop the intercept row and the same column prints as plain decimals. Rounding fixes the display without touching any fitted model.

Conclusions the notebook draws, all Verified:

- **Sine wins** on both metrics (highest R-squared, lowest RMSE), which is expected since the data was generated from a sine wave.
- **Cubic is a strong second** because a degree-3 polynomial has two turning points, enough to mimic the single hump and trough the data shows.
- The sine model wins with **2 coefficients** against the cubic's **4** (Verified), so it is both more accurate and simpler, which is the ideal outcome.

## The catch: R-squared and RMSE cannot disagree here

On **one fixed set of rows**, these two metrics are the same information twice:

```text
R-squared = 1 - RMSE^2 / var(y)
```

Verified to full precision for all five models (`var(y)` is the population variance, `ddof=0`). Because `var(y)` is the same for every model, RMSE going down **forces** R-squared to go up. So a table like this can never show one metric preferring model A while the other prefers model B. They will always rank identically. Report both anyway (R-squared is unitless and comparable across problems, RMSE is in the units of `y` and tells you the typical error size), but do not treat their agreement as independent confirmation.

The real limitation is different, and the notebook names it in its introduction: **every score above is measured on the same data the model was trained on.** Adding features to a nested model can only push training R-squared up, never down, so this comparison is structurally biased toward complexity.

An invented extension (not in the notebook) makes it visible. Continuing the polynomial ladder past cubic on this same data gives, Verified:

```text
degree 3:  R-squared 0.766181   adjusted 0.758874
degree 4:  R-squared 0.794517   adjusted 0.785865
degree 5:  R-squared 0.796927   adjusted 0.786126
degree 6:  R-squared 0.797385   adjusted 0.784313
degree 8:  R-squared 0.798768   adjusted 0.781077
```

Raw R-squared **never stops climbing** as terms are added, while **adjusted R-squared** (which penalizes parameter count) peaks at degree 5 and then falls. Adjusted R-squared is only a rough penalty, not a real test of generalization, so that divergence is a symptom rather than the cure. The actual cure, and where the module goes next, is scoring models on data they were never fitted to.

> **Source quirk (harmless):** every collection cell assigns its dict to the variable `model_intercept_eval`, even the ones for the linear, quadratic, cubic, and sine models. It is copy-paste leftover from Model 0 and changes nothing, since the dict is appended immediately, but the name is misleading if you skim.

---

## Quiz-ready facts

- **Features** are the inputs to a model: raw input variables, combinations of them, or transformations of them. Different features on the same technique give different models.
- The comparison loop: **name, formula, `smf.ols(...).fit()`, `model.rsquared`, `rmse(actual, preds)`, append a dict**, then `pd.DataFrame(list_of_dicts)` at the end.
- **`model.rsquared`** is an attribute of the fitted model. **RMSE** must be computed, via `from statsmodels.tools.eval_measures import rmse` then `rmse(data['y'], preds)`.
- **RMSE = sqrt(mean(squared errors))**, in the units of `y`, **lower is better**. **R-squared** is unitless variance explained, **higher is better**.
- **`y ~ 1`** is the intercept-only baseline: no slope, predicts the **mean of `y`** for every row (Verified 0.036733 here). Its **R-squared is 0** and its **RMSE equals the population SD of `y`** (Verified 0.760776).
- Verified five-model table (100 seeded rows, `np.random.seed(2100)`): intercept **0.000 / 0.761**, linear **0.268 / 0.651**, quadratic **0.325 / 0.625**, cubic **0.766 / 0.368**, sine **0.783 / 0.354**.
- **Sine wins, cubic is second.** The sine model needs only **2** coefficients versus the cubic's **4**, and its fitted slope on `sin(x)` is **0.9917** against a true value of 1.
- **Square a feature with `np.power(x, 2)`, not `x**2`.** In the R-derived formula language `**` sets interaction degree, so `y ~ x + x**2` silently fits `y ~ x` (Verified) with no warning.
- On one fixed dataset, **R-squared = 1 - RMSE^2 / var(y)** (Verified), so the two metrics always rank models the same way. Their agreement is arithmetic, not evidence.
- These are **training-data** scores. Adding features to a nested model can only raise training R-squared, so this table rewards complexity; the fix is evaluating on unseen data.
- `.round(3)` on the results DataFrame just removes scientific notation for readability.
- The notebook's Model 4 heading reads `y = beta_0 + sin(beta_1)`; the model actually fitted is **`y = beta_0 + beta_1 * sin(x)`**.

---

> **See also:** "Comparing Models with Cross-Validation (Cliff Notes)" and "Comparing Models with Cross-Validation - Notebook (Cliff Notes)" are the direct sequel to this note: the same comparison table, but scored on data the models never saw instead of on the training rows, which is the fix for the bias called out above. "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" and "Linear Regression Residuals and R-Squared - Notebook (Cliff Notes)" unpack the two metrics this note leans on end to end. "Introduction to Cross-Validation (Cliff Notes)" explains the fold machinery that makes the sequel work.

---

*Source: CMPIF2100 lecture notebook (personal study use).*
