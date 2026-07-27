# Module 14 - Cliff Jumper Notes

*One continuous lesson stitched from the ten Module 14 cliff notes: four
lecture notes (Linear Regression Residuals and R-Squared; Introduction to
Cross-Validation; Cross-Validation with Scikit-learn; Comparing Models with
Cross-Validation) and six notebook notes (their three companion labs, plus
Root Mean Squared Error, Comparing Multiple Linear Regression Models, and
Scikit-learn Pipeline Cross-Validation). The through-line: **Modules 12 and 13
taught you to fit ever richer linear regressions; Module 14 asks the question
that decides whether any of them deserve to exist: how wrong is this model,
and how wrong will it be on data it has never seen?** The module builds the
answer in one straight line: the residual (one row's error), R-squared and
RMSE (all the residuals collapsed into one score), the discovery that
training-data scores reward complexity whether or not it generalizes,
cross-validation as the honest replacement, scikit-learn as the machinery
that runs it, the one-standard-error rule for choosing among models, and the
pipeline for keeping preprocessing honest inside the folds.*

> All code targets **statsmodels** (formula interface, `smf`), **patsy**
> (`dmatrices`), **scikit-learn** (`KFold`, `cross_val_score`,
> `LinearRegression`, `make_pipeline`, `StandardScaler`), **pandas**,
> **numpy**, and **seaborn**. Four datasets appear: the **food truck** data
> (97 rows, profit vs city population, loaded from a URL), the **apartment
> prices CSV** (`apt_real_estate.csv`, 150 rows, the same synthetic data
> Module 13 used, loaded live from its course GitHub URL), a **seeded
> synthetic sine wave** (100 rows, `np.random.seed(2100)`), and seaborn's
> built-in **penguins** (344 rows); smaller seeded simulations built by the
> cliff notes to test claims (a 60-row polynomial demo, pure-noise fits, a
> 60-by-400 noise matrix) are named where they are used. Every number below
> was reproduced by execution (python 3.13.9, statsmodels 0.14.5, patsy
> 1.0.1, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.5, seaborn 0.13.2)
> during the cliff-note verification passes. Everything the module itself computes is **seeded**
> (`random_state=9` for every KFold, seed 2100 for the sine data), so exact
> digits are exact; the one unseeded thing anywhere is seaborn's bootstrap
> error bar, reported as behavior, not digits.

---

## 1. The map: from fitting models to judging them

Module 12 fit a line and read its coefficients. Module 13 made the inputs
rich: multiple predictors, dummies, interactions, transforms. Both modules
quietly assumed the model was worth having. Module 14 stops assuming.

The module is one long argument with seven steps:

1. **A residual** is one row's error: observed minus predicted. It is the atom
   every evaluation is built from.
2. **R-squared** and **RMSE** collapse all the residuals into a single score
   (unitless share-of-variance for the first, error-in-the-target's-units for
   the second).
3. Score several models this way on the data they trained on and a trap
   springs: **training scores can only improve as you add features**, so the
   leaderboard rewards complexity whether or not it generalizes.
4. **Cross-validation** is the fix: split the rows into K folds, train K
   models, score each on the fold it never saw, and average.
5. **scikit-learn** automates it, once patsy's `dmatrices` converts the
   formula into a numeric design matrix.
6. With honest scores plus their fold-to-fold spread, the
   **one-standard-error rule** picks the simplest model that ties the best
   one.
7. A **pipeline** extends the honesty to preprocessing, so a scaler estimated
   from data is re-estimated inside every training fold.

Everything else is detail on those seven moves.

## 2. Residuals: the atom of error

The lecture opens with a reality check: **models are always imperfect
predictors of real-world data.** Sensor noise, weather, individual human
decisions; none of it is in the data, so 100 percent accuracy is off the
table. But "imperfect" is not a free pass. If the model does not match the
observations reasonably well, its coefficients describe nothing and the
associations read from them cannot be trusted. Measuring that match is
called **goodness of fit**, or just the model's **performance**.

The running example is the food truck data from earlier modules: **97 rows**,
`population` and `profit` (the instructor is openly unsure of the units, so
do not assume dollars), fit with the familiar formula interface:

```python
import pandas as pd
import statsmodels.formula.api as smf

data_url = 'https://raw.githubusercontent.com/girishkuniyal/Predict-Profit-for-food-truck/refs/heads/master/ex1data1.txt'
profits = pd.read_csv(data_url, header=None, names=['population', 'profit'])
model = smf.ols(formula='profit ~ population', data=profits).fit()
```

(The file has no header row, hence `header=None` plus explicit `names`.)
Verified fit: **Intercept -3.895781, population 1.193034**.

The fitted model object already carries both halves of the error story:

- **`model.fittedvalues`**: the prediction for every training row. Row 0 has
  population 6.1101, so the model predicts
  `-3.895781 + 1.193034 * 6.1101 = 3.3938` while the actual profit is
  **17.592**. Pretty far off, and it is a simple model.
- **`model.resid`**: the residuals, and it is exactly
  `profits.profit - model.fittedvalues`, element for element (Verified:
  maximum difference 0.0). **Residual = observed minus predicted**, so a
  positive residual means the model guessed **too low**. Row 0's residual is
  **+14.198**, the largest in the dataset; the worst over-prediction is row
  94 at **-5.854**.

Overlay the two scatters (observed profit and predicted profit against
population) and the residuals become visible: each residual is the **little
vertical line** from a blue observed point to the orange predicted point.
The predictions themselves form a perfectly straight line, because no
nonlinear transform was applied. One plotting detail matters enough that the
course says it twice: **figure-level** seaborn functions (`relplot`) own
their whole figure and cannot be layered, so overlaying two scatters means
switching to **axes-level** functions (`scatterplot`, `lineplot`) sharing
one `ax`.

**The second view drops the input entirely.** A **predicted vs observed
plot** puts the actual values on x and the predictions on y. Perfect
predictions would land every point on the **45-degree line y = x**, which
the course draws in red by plotting `profit` against itself:

```python
fig, ax = plt.subplots(figsize=(4,4))
sns.scatterplot(data=profits, x='profit', y='predictions', ax=ax)
sns.lineplot(data=profits, x='profit', y='profit', color='red', ax=ax)
```

> **Do not over-read that picture (Verified).** A real cloud is not just
> scattered around the 45-degree line; it is systematically **flatter** than
> it. OLS predictions always have less spread than the observations
> (Verified here: `sd(predictions) / sd(profit) = 0.837873`, which is
> exactly `sqrt(R-squared)`), so extremes get pulled toward the mean. Judge
> the fit by the spread around the cloud's own trend, not by expecting
> points to straddle the red line at both ends.

**Why you cannot just average the residuals.** They sum to zero. Verified:
**3.95e-14**, floating-point zero, from 39 positive residuals and 58
negative ones that cancel exactly. That is a mathematical guarantee of OLS
whenever the model has an intercept, not evidence of quality, so the mean
residual is a dead metric for every such model, good or bad. Every real
metric first removes the sign, by **squaring** (R-squared, RMSE) or by
taking **absolute values** (MAE). That one fact generates the entire next
two sections.

## 3. R-squared: one number for the whole cloud

The course's definition: **R-squared is the correlation between the observed
and predicted values, squared.** 1 means perfect predictions (the cloud sits
on the 45-degree line), 0 means no correlation. It is stored on the fitted
model:

```python
model.rsquared                              # 0.7020315537841397  (Verified)
profits.profit.corr(profits.predictions)**2 # same number  (Verified)
```

Verified: the correlation is **0.837873**, its square **0.702032**, matching
`model.rsquared` to floating-point precision. It is also the first thing in
the top-right of the `model.summary()` header (this fit: **R-squared 0.702,
Adj. 0.699, F 223.8, Prob(F) 1.02e-26, n 97**). The instructor's verdict on
0.70: "not too bad."

> **Exam note (the definition's fine print).** The squared-correlation
> definition is guaranteed to equal R-squared only for an **OLS model with
> an intercept, scored on its own training data**, which is everything this
> course demonstrates. The general definition is
> **`R-squared = 1 - SS_res / SS_tot`**: how much better the model does than
> always predicting the mean. That version **can go negative**, and squared
> correlation cannot see the failure. Two Verified counterexamples from the
> cliff notes, each on its own toy predictor. On an invented 4-row set,
> `obs = [10, 20, 30, 40]` against `pred = [110, 120, 130, 140]` (perfectly
> correlated, 100 too high), squared correlation stays **1.0** while the
> true R-squared is **-79.0** (sklearn's `r2_score` agrees). On the food
> truck data, the deliberately backwards predictor
> `20 - 1.193034 * population` keeps squared correlation at **0.702** while
> actually scoring **-2.758**. If a quiz keys to the course's wording,
> answer its way (the
> squared correlation, 0 to 1); keep `1 - SS_res/SS_tot` in your pocket for
> the cross-validation sections below, where models get scored on data they
> were not fit on and **negative R-squared is a real outcome** (it means
> "worse than predicting the mean").

Two smaller Verified nuances worth carrying. First, statsmodels naming: the
**residual** sum of squares lives in `model.ssr` (868.5324 here) and the
**explained** sum of squares in `model.ess` (2046.3146); reading `ssr` as
"sum of squares regression" is a classic mix-up, and there is no
`model.sse`. Second, the zero end of the scale is an intuition, not a floor:
fitting `y ~ x` on pure noise (200 seeded runs at n = 97) gave in-sample
R-squared values averaging **0.0096**, never negative, close to the
theoretical `1/(n-1) = 0.0104` a junk predictor buys on average. In-sample
R-squared for OLS with an intercept is trapped in [0, 1], and **adding
features can never decrease it**. Remember that sentence; it detonates in
section 5.

## 4. RMSE: the error in the target's own units

R-squared is unitless. The second metric answers a more human question:
**"on average, how far off are our predictions?"** The RMSE notebook works
on the apartment data (the same 150-row synthetic CSV Module 13 used), with
the interaction model carried over:

```python
model = smf.ols('price_usd ~ size_m2 * city_center', data=data).fit()
```

Verified fit: R-squared **0.878**, coefficients Intercept **31586.41**,
city_center[T.True] **92869.53**, size_m2 **2996.99**, interaction
**870.57**. Then the metric is built in three steps, each fixing the last
one's problem:

1. **Mean residual: broken.** Verified: `model.resid.mean()` is 1.98e-09,
   zero as always (section 2). The signs must go.
2. **MSE, the mean squared error.**
   `MSE = (1/n) * sum((y - yhat)**2)`. Squaring makes every error positive
   and, the deeper reason, **penalizes large errors much more**: the course's
   words, "to be a little off on a prediction is alright, but being really
   off carries a much bigger penalty." Verified illustration with identical
   MAE: errors `10,10,10,10` give MSE 100; errors `0,0,0,40` give MSE 400.
   Same total miss, one catastrophic miss scored four times worse.
   Verified here: **MSE = 1,245,392,839.44**. The problem: predicting
   dollars gives an MSE in **dollars squared**. Nobody has intuition for
   that.
3. **RMSE = sqrt(MSE)**, back in the outcome's units. Verified:
   **35,290.12**, so this model is off by roughly **\$35,000** on a typical
   condo price, about **11.7 percent** of the mean price (301,926.67).
   **Lower RMSE is better.**

Both metrics have a manual form and a built-in, and they agree exactly
(Verified):

```python
import numpy as np
from statsmodels.tools.eval_measures import mse, rmse

np.power(model.resid, 2).mean()          # manual MSE
mse(data['price_usd'], model.fittedvalues)
np.sqrt(np.power(model.resid, 2).mean()) # manual RMSE
rmse(data['price_usd'], model.fittedvalues)
```

The built-ins take **two arguments** (actuals, predictions); in-sample
predictions come from `model.fittedvalues`.

> **Exam note (three traps).** (1) **`model.mse_resid` is not this MSE.** It
> divides the squared-residual sum by the residual degrees of freedom
> (150 - 4 = 146 here), not by n: 1,279,513,191.20 versus 1,245,392,839.44
> (Verified). Its square root (35,770.28) is the residual standard error
> behind the summary table, not the RMSE. The course's MSE and RMSE are the
> **n-denominator** versions. (2) RMSE is not literally the average miss;
> that is the **MAE** (Verified here: 28,448.99). RMSE is always greater
> than or equal to MAE (ratio 1.24 on this fit), because squaring weights
> the big misses. "Off by about \$35k on average" is the course's intended
> gloss; just do not equate RMSE and MAE on a definition question. (3) If a
> notebook's saved output shows a bare number where yours prints
> `np.float64(...)`, that is NumPy 1.x vs 2.x scalar display, one number
> either way.

**R-squared and RMSE are the same information on a fixed dataset.** Verified
identity: `RMSE = sd(y) * sqrt(1 - R-squared)` (population sd), so on the
same rows the two metrics **always rank models identically**. Three fits on
the apartment data make it concrete (Verified): size-only R-squared 0.338 /
RMSE 82,357; additive 0.872 / 36,204; interaction 0.878 / 35,290. Report
both (one is unitless and portable, the other is in dollars and vivid), but
their agreement is arithmetic, not corroboration.

## 5. The trap: comparing models on their training scores

The comparison notebook puts the two metrics to work the way a data
scientist actually would: **fit several candidate models on the same data,
score each, tabulate, pick.** The data is built to make the answer known in
advance, **100 seeded points of a noisy sine wave** (`np.random.seed(2100)`,
x from 0 to 7, noise sigma 0.4), so the "right" model is planted on purpose.

The workflow is one repeating block per model (name, formula,
`smf.ols(...).fit()`, `model.rsquared`, `rmse(actuals, preds)`, append a
dict to a list), then one DataFrame at the end:

```python
model_performance = pd.DataFrame(model_performance_lines)
```

**Build a list of dicts, then the DataFrame.** Growing a DataFrame row by
row is the wrong instinct; `DataFrame.append` was removed outright in pandas
2.0 (Verified: gone in 2.3.3).

Verified five-model table:

```text
        name                                  formula  r_squared   rmse
0  intercept                                    y ~ 1      0.000  0.761
1     linear                                    y ~ x      0.268  0.651
2  quadratic                   y ~ x + np.power(x, 2)      0.325  0.625
3      cubic  y ~ x + np.power(x, 2) + np.power(x, 3)      0.766  0.368
4       sine                            y ~ np.sin(x)      0.783  0.354
```

Read it bottom-up, because every row teaches something:

- **`y ~ 1`, the intercept-only baseline, earns its slot in every comparison
  you ever run.** It ignores the input and predicts one number, the mean of
  `y` (Verified: 0.036733, one hundred copies). Its R-squared is **0 by
  construction** (R-squared measures improvement over predicting the mean,
  and this model *is* predicting the mean), and its **RMSE equals the
  population standard deviation of `y`** (Verified: 0.760776, exact match to
  `y.std(ddof=0)`). That gives you a free sanity threshold: any model that
  does not clearly beat the target's own spread has learned nothing. The
  same anchor reappears at 102,193 vs sd 101,571 on the apartment data and
  at 14.06 mm on the penguins.
- **Linear** can only tilt (Verified slope -0.193, the wave's overall
  drift). **Quadratic** buys one turning point; the wave has two, so the
  gain is modest. **Cubic** buys two turning points, the same count as the
  wave, and the score jumps.
- **Sine wins**, as designed, and recovers the truth almost exactly
  (Verified: Intercept -0.001026, slope on `np.sin(x)` 0.991702, against
  true values 0 and 1), with only **2 coefficients** against the cubic's 4.
  Its training RMSE, 0.354477, is essentially the realized noise sd,
  0.354522 (Verified): the model captured the signal and left behind the
  noise, the best an honest model can do.

> **Exam note (the `x**2` landmine, again).** Inside a formula string `**`
> does **not** square anything; the formula mini-language is inherited from
> R, where `**n` caps interaction degree. Verified:
> `smf.ols('y ~ x + x**2', ...)` fits params `['Intercept', 'x']`, the
> squared term silently vanishing, no error raised. Square features with
> **`np.power(x, 2)`** (what the course teaches) or `I(x**2)`. This is the
> same landmine Module 13's nonlinear-transform notes stepped on, now with
> a comparison table at stake. And the `np.` prefix works inside the quoted
> formula only because patsy evaluates it in your namespace, which makes
> the notebook's `import numpy as np` load-bearing (Verified: absent the
> import, `PatsyError ... name 'np' is not defined`).

**Now the trap.** Every score in that table is measured on the rows the
model trained on, and section 3 already said the quiet part: training
R-squared **cannot go down** when you add a term to a nested model.
Verified extension from the cliff notes: keep climbing the polynomial
ladder on this same data and raw R-squared rises monotonically (degree 5:
0.797, degree 8: **0.799**) while **adjusted R-squared peaks at degree 5
(0.786) and then falls**. Worse, the degree-8 polynomial's training RMSE,
**0.3413**, actually **beats the sine model's 0.3545**, even though we know
for certain the data is a pure sine plus noise. The training leaderboard
cannot tell "learned the signal" from "memorized the noise." Adjusted
R-squared is a symptom-detector, still computed on training data; the cure
is scoring on data the model never saw. That is the rest of the module.

## 6. Cross-validation: the idea

**Overfitting** is the failure mode the trap produces: keep adding variables
and interactions and the training score keeps climbing while the model
learns the **idiosyncrasies of this particular sample** instead of
generalizable patterns. The lecture's example: model respiratory virus
spread on 2020 to 2022 data and you have mostly modeled **whichever
COVID-19 variant dominated those seasons**, not respiratory viruses.

**Fix 1: hold out a test set.** Shuffle the rows, carve off 10 to 30
percent, never train on it, score on it once. Good, and the lecture says
so, but on small data it hurts twice: you lose those rows from training,
and the estimate rests on a handful of points (Verified: a 10 percent
holdout of a 60-row dataset is **6 rows**), a handful that "might be kind
of weird itself." One number, no sense of how much it would move under a
different shuffle.

**Fix 2: cross-validation.** Shuffle, split into **K folds** (typically 3,
5, or 10), then run K iterations: **train on the other K-1 folds, test on
the held-out fold.** You train **K separate models** and get **K scores**.
Verified mechanics at n=150, K=5: five iterations, each training on 120
rows and testing on 30; **every row is tested exactly once and used for
training K-1 = 4 times**; `cross_val_score` performs exactly 5 fits. There
is no magic underneath: a hand-rolled loop with `KFold.split` and a cloned
model reproduces `cross_val_score` exactly (Verified, and the pipeline
section repeats the trick).

**The estimate is the average of the K test-fold scores**, and because it is
an average, the whole Module 4 toolkit applies: standard error of the mean,
a t-based confidence interval (df = K-1), model comparison. Verified demo
from the cliff notes: fold RMSEs `[0.554, 0.528, 0.501, 0.756, 0.442]`,
mean **0.556**, sem 0.053, t*(df=4) = 2.776, 95 percent CI **[0.408,
0.704]**.

Why the average is worth K times the work (Verified, 200 seeds on 60 rows):
a single 20 percent holdout returned R-squared anywhere from **0.459 to
0.965** (sd 0.063); the 5-fold average stayed within **0.829 to 0.915**
(sd 0.015), about **4.3x tighter**. Draw one unlucky split and you would
report 0.46 for a model that is really around 0.89.

And the overfitting exhibit, completed. Switch to the intro lecture note's
own ladder, a different and smaller seeded demo (60 rows,
`default_rng(2100)`, degrees 1, 2, 3, 5, 9, 15, folds at
`random_state=0`): **degree 15 hits training R-squared 0.943, the best of
all models, while its cross-validated R-squared collapses to 0.271**; the
honest winner of that ladder is degree 5 (CV 0.901). Training score says
"most complex model ever built,
please"; cross-validation says otherwise. That single contrast is the
module's reason to exist.

> **Exam note (shuffle first, and mean it).** The lecture mentions the
> shuffle in passing; treat it as mandatory. If the rows arrive sorted or
> grouped, contiguous folds test each model on a region it never trained
> on. Verified on x-sorted demo data: unshuffled 5-fold mean R-squared
> **-2.155** (meaningless), shuffled **0.899**. The same effect reappears
> with real data in section 9 (penguins arrive in species blocks:
> unshuffled 7.48 mm vs shuffled 6.54 mm). This is exactly why the course
> builds an explicit `KFold(shuffle=True, random_state=...)` object instead
> of passing `cv=5`, which does not shuffle.

Fine print the lecture glosses, kept short (all Verified in the intro
cliff note): "K equal folds" is exact only when K divides n (otherwise the
first `n % K` folds take one extra row); each held-out fold is conventionally
a **validation** set, with "test set" reserved for a final untouched
holdout, though this course says test fold and a quiz should be answered
the course's way; and the fold-based confidence interval is an
approximation, because the K training sets overlap heavily (at n=150, K=5,
any two share 60 percent of the data), a known limitation with no unbiased
fix. Useful rule of thumb, not a guarantee.

## 7. Cross-validation in scikit-learn: the design-matrix bridge

statsmodels owns the formula interface, coefficients, and p-values, but it
"doesn't really have much functionality for cross-validation." scikit-learn
automates cross-validation completely, but **accepts no formulas**: arrays
in, arrays out. The bridge is **patsy**, the same engine statsmodels
already uses to parse formulas:

```python
from patsy import dmatrices

y_feature, x_features = dmatrices('price_usd ~ size_m2 * city_center', data=data)
```

`dmatrices` returns the **outcome first, inputs second**. The feature array
is a **design matrix**: one column per fitted coefficient, with the
formula's conveniences materialized as plain numbers. Verified, shape
(150, 4):

```text
Intercept  city_center[T.True]  size_m2  size_m2:city_center[T.True]
        1                    0     75.9                          0.0
        1                    1     84.3                         84.3
        1                    0    100.2                          0.0
```

The `Intercept` column is hard-coded 1s; the dummy column is 1 only for
city-center rows (`False` is the reference); and the interaction column is
**literally the product** of the other two, so it equals `size_m2` in the
city center and 0.0 outside it. An interaction is not an instruction
scikit-learn must understand; it is just another numeric column. Sanity
check worth remembering (Verified): fitting scikit-learn's
`LinearRegression(fit_intercept=False)` on this matrix reproduces the
statsmodels coefficients **exactly** (31586.41 / 92869.53 / 2996.99 /
870.57). Same model, different steering wheel; `dmatrices` is also
byte-identical to what `smf.ols` builds internally as `model.exog`
(Verified).

Cross-validation is then three objects:

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

kf = KFold(n_splits=5, shuffle=True, random_state=9)
clf = LinearRegression(fit_intercept=False)
scores = -cross_val_score(clf, x_features, y_feature.ravel(), cv=kf,
                          scoring='neg_root_mean_squared_error')
```

Every argument is deliberate:

- **`KFold(n_splits=5, shuffle=True, random_state=9)`** describes the split
  (no data is passed at construction). Shuffling is the whole reason to
  build the object; `random_state=9` freezes it, and these two settings are
  the **module's constants**, reused in every remaining notebook, which
  also means every model is scored on **identical folds** (a paired, fair
  comparison). The shortcut `cv=5` is that same split **unshuffled**
  (Verified: mean RMSE 36,113.69 vs 35,464.73 shuffled).
- **`fit_intercept=False`** because the design matrix already carries the
  intercept column; letting scikit-learn add its own leaves the scores
  unchanged here but zeroes out `coef_[0]` and hides the real intercept in
  `intercept_` (Verified). `False` keeps columns and coefficients aligned.
- **`.ravel()`** flattens patsy's (150, 1) outcome to the 1-D shape
  scikit-learn documents for a single target. (Verified: `LinearRegression`
  tolerates the 2-D version silently; other estimators warn. Ravel anyway.)
- **`cross_val_score` clones the estimator** for each fold, so `clf` itself
  is never fitted (Verified: no `coef_` after the call). You get scores,
  never a trained model.

**The minus sign is a convention, not a bug.** Every scikit-learn scorer is
"higher is better," so error metrics ship negated:
`'neg_root_mean_squared_error'`. Negate the call to read RMSE. Verified
folds on the apartment interaction model: **31275.61, 39953.92, 40810.29,
29354.24, 35929.57**, mean **35,464.73** (about 11.7 percent of the
\$301,927 mean price). Swap in `scoring='r2'`, which needs **no** negation
(higher R-squared is already better): folds **0.9102, 0.8133, 0.8421,
0.9045, 0.8747**, mean **0.8690**, sitting a touch below the in-sample
0.8785, which is exactly the optimism cross-validation exists to strip
out.

> **Exam note (the course's loose words here).** The lecture calls the model
> "a linear regression classifier" and names the variable `clf`;
> `LinearRegression` is a **regressor** (continuous output), and
> `cross_val_score`'s parameter is actually `estimator`. The lecture reads
> the CV R-squared as "0.86 ... pretty good correlation": the value is
> 0.8690 (rounds to 0.87), R-squared is variance explained rather than a
> correlation, and a **cross-validated** R-squared is not floored at zero
> (Verified: a noise-only model scores around -7 to -12 per fold on this
> data). Answer a keyed quiz the course's way; know the distinctions.

Two Verified nuances from the cliff notes worth one sentence each. The
seed is a flattering draw: across `random_state` 0 to 29, mean RMSE ranged
35,465 to 36,733, and **seed 9 gave the lowest of all 30**, so quote the
course's number as the course's number and do not read meaning into its
last digits. And `scores.mean()` (the mean of fold RMSEs, the standard
report) is not the pooled RMSE over all held-out predictions (Verified:
35,464.73 vs 35,756.30 here; with equal-sized folds the mean of folds is
always the smaller of the two), so never mix the two statistics when
comparing models.

## 8. Choosing among models: the one-standard-error rule

Now the payoff. Even restricted to linear regression on **two** input
variables, the candidate space is real: keep or drop variables, add
interactions, transform features, combine all three. The comparison lecture
lines up **seven formulas** as models 0 through 6 and wraps the repeated
work in one user-defined function, `eval_cv(formula, model_name, data)`:
patsy expands the formula, `KFold(5, shuffle=True, random_state=9)` splits,
`LinearRegression(fit_intercept=False)` fits, `-cross_val_score(...)`
scores, and the function returns a tidy five-row DataFrame (name, formula,
coefficient count, fold id, RMSE). A `for` loop with **`enumerate`** (index
doubles as the model name) runs all seven, and `pd.concat` stacks the
results into one **35-row** frame. Verified means and coefficient counts:

```text
model  formula                                        coefs   mean RMSE    1 SE
  0    price_usd ~ 1                                    1     102,193     2,283
  1    price_usd ~ size_m2                              2      83,688     2,454
  2    price_usd ~ city_center                          2      73,214     4,722
  3    price_usd ~ size_m2 + city_center                3      36,225     2,152
  4    price_usd ~ size_m2 * city_center                4      35,465     2,278
  5    price_usd ~ np.power(size_m2, 2) + city_center   3      39,691     3,482
  6    price_usd ~ np.power(size_m2, 2) * city_center   4      38,534     2,893
```

Reading it the way the lecture does:

- **Model 0 is terrible, and that is the point.** Its 102,193 is essentially
  the standard deviation of `price_usd` itself (101,571, Verified), the
  section-5 anchor again: predict the mean, inherit the target's spread.
  On its first fold it predicts **303,658.33** flat, that fold's training
  mean (Verified).
- **Either variable alone helps; both together transform.** Model 2 to
  model 3 roughly halves the error, so size and location carry largely
  different information.
- **Models 5 and 6 are worse than 3 and 4** because `np.power(size_m2, 2)`
  **replaces** the linear term rather than adding to it (Verified from the
  design-matrix columns), and the data wants the line.
- **Model 4 is the best raw score**, and the lecture is candid about why:
  the apartment data is synthetic and "was actually generated based on"
  the size-by-location interaction.

The plot is a seaborn point plot of the 35 fold scores, mean plus error
bar per model, with the bar deliberately **narrow**:

```python
sns.catplot(data=results_df, x='model_name', y='rmse', kind='point',
            linestyle='none', errorbar=('ci', 68), height=4)
```

> **Exam note (68, not 95).** The between-model convention is a **one
> standard error** interval, about 68 percent, instead of the usual two
> standard errors, about 95 percent. Narrow bars make it harder for a
> mediocre model to look tied with the best. Two precision points the
> course blurs (Verified): "one standard deviation" and "one standard
> error" are different quantities (the relevant one is the SEM, the fold
> sd over sqrt(5)); and seaborn's `errorbar=('ci', 68)` is a **bootstrap**
> interval, not the literal 1-SE bar (`errorbar=('se', 1)` is; half-widths
> 1,866 vs 2,152 on model 3). The bootstrap is also the module's one
> unseeded computation, so its whisker ends wobble slightly between runs
> (behavior, Verified over 25 redraws: model 4's band stayed within about
> 33,200 to 37,700). None of it changes the selection. Answer a quiz with
> the course's framing: one SE instead of two, 68 instead of 95.

**The rule, and the pick.** Find the best mean, add its one standard
error, and choose the **simplest** model under that threshold. Verified
arithmetic: 35,464.73 + 2,278.50 = **37,743.22**; under it sit model 4
(35,465, 4 coefficients) and model 3 (36,225, 3 coefficients); models 5
and 6 have overlapping bars but means above the threshold, so they are
out. **The pick is model 3, `price_usd ~ size_m2 + city_center`.** The
interaction buys 760 USD of mean RMSE, well inside the roughly 2,200 USD
of fold-to-fold noise, and a difference smaller than the noise is not a
difference you pay complexity for. (The lecture applies exactly this
reasoning out loud; the standard name for it, the **one-standard-error
rule**, is the cliff notes' addition. Robustness, Verified across 20
seeds: model 4 had the best raw mean 20 of 20 times, and the rule picked
model 3 in 19 of 20.)

**Sit with what just happened.** Module 13, on this same CSV, showed the
interaction is real: significant (p = 0.0064), built into the generating
process by the course's own admission. Module 14, on this same CSV, hands
the trophy to the additive model anyway. Both are right. Significance asks
"is this effect distinguishable from zero in-sample?" The one-SE rule asks
"does paying for it buy out-of-sample error beyond the noise floor?" A
real effect can still not be worth a coefficient. That tension, not any
single number, is the module's sharpest lesson.

Last step of the lecture, easy to get wrong: **after selection, refit the
chosen formula on all the data.** Cross-validation's five fitted models
are throwaway; its job was estimating performance and choosing. The
deployed model is one fresh fit of `price_usd ~ size_m2 + city_center` on
all 150 rows.

> **Exam note (a source bug, flagged in both companion notes).** The
> notebook's helper contains `num_features = x_features.shape[0]` with a
> comment saying it counts columns; `shape[0]` is **rows** (150 for every
> model). The variable is dead (never used again) and the results table
> correctly uses `shape[1]`, so nothing downstream is wrong. If asked how
> complexity is measured: **the number of columns, `shape[1]`, which
> counts coefficients including the intercept.**

## 9. Pipelines: keeping preprocessing honest inside the folds

One loose end. Module 13 standardized features with `StandardScaler`,
whose mean and standard deviation are **estimated from the data**. That
makes standardization **data-dependent preprocessing** (converting grams
to kilograms is not: nobody estimated the 1000). Estimate the scaler on
**all** the rows and then cross-validate, and the training folds have
absorbed the test fold's statistics; the simulation of unseen data is
contaminated, and performance may be overestimated, the course warns. The
honest recipe per fold: estimate the scaler on the training folds, train
on the training folds, transform the test fold **with the training-fold
statistics**, then score.

Doing that by hand five times is tedious, so scikit-learn chains the steps
into one estimator:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), LinearRegression())
scores = -cross_val_score(pipeline, x_features, y_feature.ravel(), cv=kf,
                          scoring='neg_root_mean_squared_error')
```

A pipeline's steps are auto-named from the class names
(`'standardscaler'`, `'linearregression'`, Verified); every step but the
last must transform, the last is the estimator, and the whole pipeline
**is** an estimator, which is exactly why it slots into
`cross_val_score`'s estimator argument where the bare model went. Passed
this way, **every step is re-fit inside each training fold
automatically**: five folds, five separately estimated scalers, each
transforming only its own test fold. A hand-written loop that does
scaler-fit / model-fit / transform / score per fold reproduces the
pipeline's five scores **exactly** (Verified, max difference 0.0). The
pipeline is bookkeeping, not magic.

The worked model predicts penguin flipper length:
`flipper_length_mm ~ bill_length_mm + body_mass_g` on seaborn's penguins.
Verified plumbing: `dmatrices` drops the 2 rows with NAs among the modeled
columns (344 to 342); the columns sit on wildly different scales (body
mass sd about 802 g vs bill length sd about 5.5 mm), which is the
motivation for scaling; patsy's Intercept column gets flattened to zeros
by the scaler and `LinearRegression` (default `fit_intercept=True` here)
supplies its own intercept, fitting a harmless 0.0 on the dead column.
Verified result with the module's constant splitter
(`KFold(5, shuffle=True, random_state=9)`, test folds 69/69/68/68/68):

```text
fold RMSEs:  7.1495  6.6674  6.4717  6.5710  5.8154
mean 6.5350 mm       (unshuffled cv=5: 7.4763, penguins arrive in species blocks)
```

About 6.5 mm of typical error against a target whose own standard
deviation is 14.06 mm, consistent with the mean CV R-squared of 0.773
(Verified). The scaler touches **only the inputs**, never `y`, so the RMSE
stays in millimetres.

> **Exam note (and an honest one).** If asked why `StandardScaler` goes
> inside the pipeline: **so the standardization is estimated on the
> training folds only and performance on unseen data is not
> overestimated.** That is the right habit and the right quiz answer. The
> cliff notes then verify a subtlety worth knowing: **for plain OLS the
> pipeline changes nothing on this data**, because least squares is
> unmoved by linear rescaling of inputs; pipeline, no scaler at all, and
> even a deliberately leaky whole-data scaler all return 6.5350 to within
> 1e-14 (Verified). Standardizing changes the **coefficients' units**, not
> OLS predictions (Verified: the standardized intercept is 200.9152,
> exactly the mean flipper length). Where fit-inside-the-fold genuinely
> bites is scale-sensitive and selection steps (all Verified on the same
> folds): KNN improves from 6.86 to 6.34 when scaled; Ridge(alpha=100)
> shifts from 6.53 to 7.10; and the severe case, **feature selection
> outside the loop**, manufactured R-squared 0.135 out of pure noise
> while the honest pipeline reported -0.636, with the leaky version
> scoring higher in 30 of 30 seeded runs. The rule costs nothing when it
> does not matter and saves you exactly when it does.

## One-paragraph recap

A residual is one row's error, observed minus predicted, and because OLS
residuals sum to zero the only usable metrics square them (R-squared, the
share of variance explained, equal on training data to the squared
observed-predicted correlation; RMSE, the error in the target's own units,
with the baseline `y ~ 1` scoring the target's own standard deviation) or
take their absolute value (MAE). Training scores can only improve as
features are added, so they reward complexity indiscriminately (a degree-8
polynomial out-scores the true sine model on the training set); the fix is
cross-validation: shuffle, split into K folds, train K models, score each
on its held-out fold, and average, which is both fairer and stabler than
any single split. scikit-learn runs it once patsy's `dmatrices` freezes the
formula into a design matrix: `KFold(n_splits=5, shuffle=True,
random_state=9)`, `LinearRegression(fit_intercept=False)`, and
`cross_val_score` with negated error scorers. With honest means and their
fold-to-fold standard errors, compare models on a point plot with
one-standard-error bars and pick the **simplest model within one standard
error of the best**, then refit it on all the data; on the apartment CSV
that means the additive model beats the genuinely-real interaction, because
a real effect can still not be worth a coefficient. Finally, any
preprocessing whose parameters are estimated from data belongs **inside a
pipeline**, so it is re-estimated on each training fold and the test fold
stays honestly unseen.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
