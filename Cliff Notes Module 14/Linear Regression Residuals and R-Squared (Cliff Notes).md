# Linear Regression Residuals and R-Squared: measuring how well a model fits its training data

**Module 14 Cliff Notes** | Source: lecture transcript "Linear Regression Residuals and R-Squared"

---

## TL;DR

- Models are **always imperfect predictors** of real-world data, so we never expect 100 percent accuracy. Sensor noise, weather, and individual human decisions all inject randomness a model cannot see.
- We still need the model to fit **reasonably well**, otherwise we cannot trust the associations we read out of its coefficients. Evaluating that fit is called measuring the **goodness of fit**, or (the instructor's usual word) the **performance** of the model.
- A **residual** is the error for **one** data point: **observed minus predicted**. statsmodels stores predictions in **`.fittedvalues`** and residuals in **`.resid`**, and `.resid` is exactly `observed - fitted` (Verified).
- Running example: the familiar **food truck** dataset, **97 rows**, two columns, `population` (city size) and `profit`. Model: **`profit ~ population`**, ordinary least squares (Verified fit: Intercept **-3.8958**, population **1.1930**).
- Two ways to see the errors: overlay predictions on the observed scatter (residuals are the **vertical gaps**), or draw a **predicted vs observed** plot with a **45-degree line** showing where perfect predictions would land.
- **R-squared** is the lecture's first evaluation metric: as taught, it is the **correlation between observed and predicted values, squared**. Read it off **`model.rsquared`**. Verified here: **0.7020**, and `profits.profit.corr(profits.predictions) ** 2` reproduces it to floating-point precision (Verified).
- **1 = perfect, 0 = no correlation.** This model's 0.70 is "not too bad" per the instructor.
- Careful: the squared-correlation definition is only guaranteed to equal R-squared for an **OLS model with an intercept, scored on its own training data**. The general formula is **1 - SS_res / SS_tot**, which **can go negative** on new data. Module 14 heads straight into cross-validation, so this distinction matters soon.

> **The one takeaway:** a residual is one point's error (observed minus predicted, `.resid`), and R-squared summarizes all of them into a single 0-to-1 goodness-of-fit score (`model.rsquared`), which for this training-data fit equals the squared correlation between what happened and what the model said would happen.

---

## Why bother evaluating at all

The lecture opens with a reality check: **models are always imperfect predictors of real-world data, and we will never get 100 percent accuracy.** Two reasons given:

- **Sensor and environment noise.** If you are gathering data from sensors, "what's the wind speed that day" is genuinely random from the model's point of view.
- **Human behavior.** For the food truck, daily profit depends on weather and on individual decisions to eat there versus somewhere else that day. None of that is in the data.

But "imperfect" is not a free pass. We want the model to **match the real-world observations reasonably well**, because otherwise **we cannot trust the associations we learn from the model**. A badly fitting model's coefficients are not a description of anything. Hence the process this lecture starts: **evaluating how well the model matches the data**, also called **goodness of fit** or model **performance**.

The plan is two steps: first **residuals** (how far off are we for *one* data point), then a first **evaluation metric** answering the quantitative question, "how far off **in general and on average** is this thing?" There are many such metrics. This lecture covers **r-squared**, a common one for linear regression.

> **Training-data caveat, stated up front:** everything here is measured on the **training data**, the same rows the model fit its parameters on. The lecture flags this explicitly ("later we'll do this with data that the model hasn't seen before"). That later material is cross-validation.

## The setup: food truck profit

The dataset is the food truck data used in prior modules: **97 rows**, two columns, the **population** of the city and the **profit** of the food truck. The instructor is candid that he does not know the units ("I'm not sure, you know, a month or thousands of dollars or who knows what it is, tens of thousands"), so do not assume dollars, months, or anything else; the course does not specify.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

data_url = 'https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt'
profits = pd.read_csv(data_url, header=None, names=['population', 'profit'])
model = smf.ols(formula='profit ~ population', data=profits).fit()
```

(The file is a bare two-column CSV with **no header row**, so `header=None` plus explicit `names` supplies the column labels.)

Verified structure: **97 non-null rows, both columns `float64`**, first row `population 6.1101`, `profit 17.5920`. Verified fitted parameters:

```text
Intercept    -3.895781
population    1.193034
```

## Predictions: `.fittedvalues`

To see the errors, first get **what the model would have estimated for every input value of population in the training set**. This is so important that statsmodels already has it, on the fitted model object:

```python
model.fittedvalues
profits['predictions'] = model.fittedvalues
```

Verified: 97 predictions, the first being **3.393774**, while the actual profit for that row is **17.592**. As the lecture notes, that is **pretty far off**, and it is a simple model.

A quick look at just the predictions, with the figure-level `sns.relplot`, shows what a fitted line looks like:

```python
sns.relplot(data=profits, x='population', y='predictions')
```

The result is a **perfectly straight line**, because no nonlinear transformation was applied to the input: the model learned one slope and one intercept, so every population value maps onto that line. (Verified: the correlation between `population` and `predictions` is exactly **1.0**.)

## Seeing the errors: overlay the two scatters

To compare predictions with reality on the same axes, the lecture switches from the figure-level `sns.relplot` to **axes-level** functions, which can share one `Axes` object:

```python
fig, ax = plt.subplots(figsize=(4,3))
sns.scatterplot(data=profits, x='population', y='profit', ax=ax)        # actual (blue)
sns.scatterplot(data=profits, x='population', y='predictions', ax=ax)   # predicted (orange)
plt.show()
```

Seaborn handles the two colors automatically. The observed profits are "all over the place," **especially at lower populations**, while the predictions sit on their straight line. For any given population, imagine a **little vertical line** from the blue point down or up to the orange one. **That vertical distance is the residual**, the model's error for that data point.

## Residuals: `.resid`

Residuals matter enough that statsmodels hands them to you directly:

```python
model.resid
```

The first one is roughly **14 units off** (Verified: **14.198226**). And it is exactly what you would compute by hand:

```python
profits.profit - profits.predictions
```

Verified: `.resid` and `profit - predictions` are **identical element for element**. So the definition is simply **residual = observed minus predicted**, and a **positive** residual means the actual profit was **higher** than the model said (which is the case for data point 0).

Verified spread on this fit: residuals run from **-5.854** to **+14.198**, and they **sum to zero** (Verified: total 3.95e-14, mean 4.07e-16, both floating-point zero).

> **Why that zero matters:** for OLS **with an intercept**, the residuals always sum to zero, because the intercept absorbs any constant offset. So the *average residual* is a useless performance metric: it is zero for every such model, good or bad. Any real metric has to remove the sign, either by squaring (r-squared here, RMSE next) or by taking absolute values.

## The predicted vs observed plot

To set up r-squared, the lecture visualizes the errors a second way, in a **predicted vs observed figure**. The input variable is dropped entirely: put the **actual data on the x-axis** and the **predictions on the y-axis**.

```python
sns.relplot(data=profits, x='profit', y='predictions', height=3)
plt.show()
```

If the predictions were perfect, every point would sit at `predicted == observed`, which is a **45-degree line**. Draw that reference line by plotting `profit` against **itself**:

```python
fig, ax = plt.subplots(figsize=(4,4))
sns.scatterplot(data=profits, x='profit', y='predictions', ax=ax)
sns.lineplot(data=profits, x='profit', y='profit', color='red', ax=ax)   # perfect-prediction line
plt.show()
```

The red line is where perfect predictions would land ("exact same prediction, exact same actual value"). How far the cloud strays from it is how wrong the model is, and **r-squared = 1 means the cloud lies exactly on that 45-degree line**.

> **Do not over-read the picture (Verified):** a real cloud does not just scatter *around* the 45-degree line, it is systematically **flatter** than it. Fitting a line through this predicted-versus-observed cloud gives a slope of **0.702**, exactly the r-squared. That is a general property of OLS predictions on training data: predictions have less spread than the observations, so extremes get pulled toward the mean. Judge the fit by the **spread around** the cloud's trend, not by expecting the points to straddle the red line at both ends.

## R-squared

The lecture's definition: **r-squared is the correlation between the observed and the predicted values, squared.**

- **Perfect correlation gives an r-squared of 1**, with every point lined up on the 45-degree line.
- **No correlation** (dots all over the plot) gives an r-squared near **0**.
- **Why squared?** "We don't really want negative values. We don't really care if it's like above or below or whatever. We're just kind of looking for an average metric here." Squaring removes the sign so the metric is a single non-negative summary.

statsmodels stores it:

```python
model.rsquared
```

Verified: **0.7020315537841397**, so about **0.70**. Per the instructor, that is **"not too bad."**

You can compute it yourself, which is the whole point of the definition:

```python
profits.profit.corr(profits.predictions) ** 2
```

Verified: correlation **0.8378732325263406**, squared **0.7020315537841393**, matching `model.rsquared` to within 4.4e-16 (pure floating-point noise). It is also, as the lecture says, "one of the first things you see" in the model summary:

```python
model.summary()
```

Verified: `R-squared: 0.702` sits at the **top right of the summary's header block**, on the very first data line (opposite `Dep. Variable`). The rest of that header for this fit: **Adj. R-squared 0.699, F-statistic 223.8, Prob(F-statistic) 1.02e-26, No. Observations 97, Df Model 1**.

> **Sibling note:** the recomputation by correlation and the `model.summary()` call were **live-coded during the lecture**; the distributed start notebook stops at `model.rsquared` (Verified: 20 cells, last one `model.rsquared`). The companion notebook note covers the saved cells and their figures.

> **Simple-regression shortcut (Verified):** with a single input, the correlation between observed and predicted is the same magnitude as the correlation between observed and the **input itself**. Here `corr(profit, population)` is also **0.837873**, so its square is also **0.7020**. That equality is specific to one-predictor models; with several inputs, only the observed-versus-predicted correlation works.

## Correcting the course, carefully

**The course's wording:** "r squared is actually the correlation between the observed and predicted values ... it is squared."

**Where that holds:** exactly here. For an **OLS model with an intercept**, scored on the **same data it was fit on**, squared correlation and r-squared are provably the same number, which is why the hand computation above matched.

**Where it breaks:** the general definition of r-squared is

```text
R^2 = 1 - SS_res / SS_tot          SS_res = sum (observed - predicted)^2
                                   SS_tot = sum (observed - mean(observed))^2
```

which measures how much better the model does than always guessing the mean. That version is **not** a squared correlation, and unlike a squared correlation it **can be negative**. A tiny invented example makes the gap obvious (Verified):

```python
obs  = pd.Series([10.0, 20.0, 30.0, 40.0])
pred = pd.Series([110.0, 120.0, 130.0, 140.0])   # perfectly correlated, but 100 too high
```

```text
corr             1.0
corr ** 2        1.0        <- "perfect" by the squared-correlation reading
1 - SSres/SStot -79.0       <- the real R^2 (sklearn's r2_score agrees)
```

Flip the predictions to run backwards (`[40, 30, 20, 10]`) and squaring hides the sign entirely: correlation **-1.0**, squared **1.0**, true R-squared **-3.0** (Verified). So "we don't really care if it's like above or below or whatever" is fine as intuition for *this* fit and dangerous as a general rule: a systematically biased or backwards model can look perfect under squared correlation.

**Exam safety:** if a quiz keys to the course's wording, answer **"the squared correlation between observed and predicted values, 1 is perfect and 0 is no correlation."** That is what was taught and it is correct for everything demonstrated in this lecture. Keep `1 - SS_res / SS_tot` in your pocket for the cross-validation material, where models get scored on data they were **not** fit on and **negative r-squared is a real outcome** (it means the model is worse than predicting the mean).

> **One more nuance on the zero end (Verified):** the lecture's no-correlation case, where "it's going to have an r squared of zero," is the right intuition but not an exact floor. Fitting `y ~ x` on 97 rows of pure independent noise, 200 times (seeded with `default_rng(2100)` so this is reproducible), gave r-squared values with **mean 0.0096, median 0.0042, max 0.107, and never negative**. In-sample r-squared for OLS with an intercept is trapped in **[0, 1]** and drifts slightly above zero even for a worthless predictor, averaging about `1 / (n - 1)` per junk predictor (here 1/96 = 0.0104, matching the observed mean). Adding features can **never decrease** training r-squared, which is exactly why adjusted r-squared and cross-validation exist.

---

## Quiz-ready facts

- **Models are always imperfect**; real-world randomness (weather, sensors, individual human choices) guarantees it. Evaluating the mismatch is **goodness of fit**, also called model **performance**.
- **Residual = observed minus predicted**, for a **single** data point. Positive residual means the model **under**-predicted.
- **`model.fittedvalues`** = predictions on the training rows. **`model.resid`** = residuals. Verified identical to `data.y - data.predictions`.
- On the population-versus-profit scatter, residuals are the **vertical distances** between the observed points and the fitted line.
- Predictions plot as a **straight line** against the input when no nonlinear transformation is applied.
- A **predicted vs observed** plot puts observed on x and predicted on y; **perfect predictions form a 45-degree line**, drawn with `sns.lineplot(x='profit', y='profit', color='red')`.
- **Figure-level** seaborn (`relplot`) draws one layer and owns its own figure; to put two columns on one plot, switch to **axes-level** functions (`scatterplot`, `lineplot`) sharing an `ax`.
- **R-squared as taught:** the **correlation between observed and predicted, squared**. Squared so the metric has **no negative values** and does not care about direction.
- **1 = perfect predictions, 0 = no correlation.** Read it from **`model.rsquared`**, or recompute as `y.corr(predictions) ** 2`, or read it off the **top right of the `model.summary()` header block**.
- **Verified food truck fit** (`profit ~ population`, 97 rows): Intercept **-3.8958**, slope **1.1930**, **r-squared 0.7020** (summary shows 0.702, Adj. 0.699, F 223.8, Prob(F) 1.02e-26). First prediction **3.3938** vs observed **17.592**, residual **14.1982**.
- **Residuals of an OLS fit with an intercept sum to zero** (Verified: 3.95e-14). So mean error is a dead metric; you must square or take absolute values.
- **General r-squared is `1 - SS_res / SS_tot`** and **can be negative** on unseen data (worse than predicting the mean). The squared-correlation identity is a **training-data, intercept-included, OLS** special case (Verified counterexample: perfectly correlated but shifted predictions give squared correlation 1.0 and true r-squared -79.0).
- **In-sample r-squared never decreases when you add features**, and a useless predictor still buys about `1 / (n - 1)` of it on average (Verified: mean 0.0096 across 200 pure-noise fits at n = 97, against 1/96 = 0.0104 in theory). This is the motivation for adjusted r-squared and cross-validation.

---

> **See also:** "Linear Regression Residuals and R-Squared - Notebook (Cliff Notes)" (the companion notebook this lecture narrates, with the full cell-by-cell outputs and figures), "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" (the next metric, which keeps the residuals in the units of the target instead of collapsing them to a correlation), "Introduction to Cross-Validation (Cliff Notes)" (why a training-data score like this one flatters the model, and what to do about it), "Comparing Models with Cross-Validation (Cliff Notes)" (scoring competing models on data they have not seen), and "Comparing Multiple Linear Regression Models - Notebook (Cliff Notes)" (r-squared and adjusted r-squared used side by side to choose between fits).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
