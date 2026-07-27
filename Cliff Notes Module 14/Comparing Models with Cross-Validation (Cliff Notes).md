# Comparing Models with Cross-Validation: pick the simplest model that ties the best one

**Module 14 Cliff Notes** | Source: lecture transcript "Comparing Models with Cross-Validation"

---

## TL;DR

- Cross-validation is not just a score, it is a **model selection tool**: build several candidate models, cross-validate each one, and compare their **test-fold** performance.
- Even inside plain linear regression there are **many** possible models: different subsets of columns, interactions, nonlinear transformations, and combinations of all three. With four or five variables the list explodes.
- Two goals pull against each other: we want models that **generalize** to unseen data, and we want **simpler** models. A very complex model can fit the training data beautifully and still do worse on test data (overfitting), and simpler models give more generalizable, more explainable patterns.
- The lecture's recipe: wrap the repeated steps in **one user-defined function** (formula in, cross-validation scores out), loop it over a list of formulas, stack the results into one DataFrame, then plot.
- Always include a **`price_usd ~ 1` intercept-only model** as the floor. It ignores the inputs entirely and predicts one flat number, so any model worth keeping must beat it.
- Read the results from a **point plot of mean RMSE with a 68% (one standard error) interval**, not 95%. That narrower interval is the between-model comparison convention.
- The winner on raw score was **model 4** (`size_m2 * city_center`, mean RMSE 35,465, Verified), but the lecture picks **model 3** (`size_m2 + city_center`, mean RMSE 36,225, 3 coefficients, Verified), because model 3 sits **inside** model 4's interval and is simpler.
- Once a model is chosen, it is fine to **refit it on all the data**. Cross-validation's job was estimating performance and choosing, not producing the final fit.

> **The one takeaway:** cross-validate every candidate formula, plot mean test-fold error with a one-standard-error interval, and choose the **simplest** model whose score is within that interval of the best score, then retrain that model on the full dataset.

---

## Why you need to compare models at all

The lecture opens by pointing out how large the search space is even when "we are just restricting ourselves to just linear regression models like we are in this course." Given a dataset with four or five columns you can:

- keep some variables and remove others,
- add **interactions** between many different variables (`a * b`, where one variable's slope depends on another),
- apply **nonlinear transformations** (squaring a feature, for instance),
- or mix interactions and nonlinear transformations together.

Each of those is a different formula, and therefore a different model. "So how do we compare? How do we know which one to choose?" That is the whole question this lecture answers, and cross-validation is the tool.

## The trade-off: performs well vs. stays simple

Two wants, stated plainly in the lecture:

1. **Generalization.** We want models that fit well on **test data**, data they have not seen before, not just on the rows used for training.
2. **Simplicity.** We also prefer simpler models with fewer features.

These fight each other. A complex model with lots of interactions "might fit your data really well, your training data especially, really well, but not perform as well on test data." That is **overfitting**: the model memorizes noise that does not repeat in new data. Simpler models tend to capture the more generalizable patterns and associations, and they are easier to explain.

So the goal is not the highest score, full stop. It is the **simplest model that performs about as well as the best one**.

> **Quick refresher (self-contained):** k-fold cross-validation splits the data into k equal parts, then trains k times. Each round holds out one fold as the test set and trains on the other k minus 1. You end up with k independent test scores instead of one lucky or unlucky split. This lecture uses **k = 5**, so **five scores per model**.

## The candidate models

The running example is the apartment or condo real estate data: **150 rows** (Verified; the lecture says "about 150"), holding the size of the apartment in square meters, whether or not it is in the city center, and the price. Seven formulas are lined up as models 0 through 6 (Verified: `len(formulas)` is 7):

```python
formulas = ['price_usd ~ 1',                                   # 0: intercept only
            'price_usd ~ size_m2',                             # 1: size alone
            'price_usd ~ city_center',                         # 2: location alone
            'price_usd ~ size_m2 + city_center',               # 3: additive
            'price_usd ~ size_m2 * city_center',               # 4: additive + interaction
            'price_usd ~ np.power(size_m2, 2) + city_center',  # 5: squared size, additive
            'price_usd ~ np.power(size_m2, 2) * city_center']  # 6: squared size, interaction
```

**Why the intercept-only model earns a slot.** In statsmodels, `price_usd ~ 1` means "just the intercept." The lecture calls this "a really good one to include" because "it's not even looking at the test data coming in, it's just making a flat prediction over all test data based on the distribution of that output variable." It is the do-nothing baseline. You want a model that "actually learns something from the actual data that's coming in." Verified: on the first fold it predicts the single value **303,658.33** for every test row, which is exactly the mean price of that fold's training rows.

> **Why `np.power(size_m2, 2)` and not `size_m2**2`?** Because the formula mini-language is inherited from R, and inside a formula `**` sets the **maximum interaction degree** rather than raising anything to a power. Verified: `price_usd ~ size_m2**2` produces the design matrix columns `['Intercept', 'size_m2']`, with no squared term at all. Wrapping the arithmetic in a Python call (`np.power(...)`, or `I(size_m2**2)`) is what actually squares the column.

Note what models 5 and 6 really are: the squared size **replaces** linear size instead of joining it. Model 5 therefore carries the same number of coefficients as model 3 (Verified: 3 each) while fitting a worse-shaped curve, and it duly scores worse.

And keep the scale in mind: seven candidates came out of **two** input variables. In the lecture's words, "you can imagine this is just for two variables, so you can imagine so many more with a bigger data set of real-world data with many variables."

## Wrap the repetition in a function

"Whenever you're doing things, you know, a set of steps repeatedly in Python, you should be thinking about making one of those user-defined functions." The lecture's `eval_cv` takes a **formula string**, a **model name**, and the **dataset**, and returns a small DataFrame of RMSE values, one row per test fold. (The lecture notes you could return R-squared or another metric instead; in this code that means changing the `scoring` string.) Its docstring documents each parameter and the return value, which the lecture calls out as good practice.

Inside, four steps do the work:

```python
y_feature, x_features = dmatrices(formula, data=data)   # statsmodels/patsy formula -> arrays
kf = KFold(n_splits=5, shuffle=True, random_state=9)    # 5-fold splitter
clf = LinearRegression(fit_intercept=False)             # patsy already added an Intercept column
scores = -cross_val_score(clf, x_features, y_feature.ravel(), cv=kf,
                          scoring='neg_root_mean_squared_error')
```

The pattern is a **handoff**: patsy's `dmatrices` turns the human-friendly formula into the numeric design matrices scikit-learn wants, and scikit-learn does the cross-validation. The returned DataFrame carries the model name, the formula text, the coefficient count, the fold id (0 through 4), and the RMSE, so everything needed for the later plot travels with the scores.

> **Two details worth knowing.** (1) scikit-learn's `scoring` values are always **higher is better**, so root mean squared error ships as `neg_root_mean_squared_error`; the leading minus sign in `-cross_val_score(...)` flips the negatives back into ordinary positive RMSE. (2) `fit_intercept=False` is set because the patsy design matrix **already contains** an `Intercept` column of ones, so letting scikit-learn add its own would double up. (Verified: on this data the RMSEs come out identical either way, but `False` is the correct setting for a design matrix that already has its own intercept.)

**Complexity counter.** The function also records how many features the model uses, described in the lecture as "just the number of columns in the feature array or the design matrix." That count is the simplicity yardstick for the final decision.

> **Correction, flagged.** The notebook's own `num_features` line reads `x_features.shape[0]`, which is the number of **rows**, not columns (Verified: it returns 150 for every model). It contradicts the comment sitting next to it and the spoken description. Nothing downstream breaks, because the value actually saved into the results table uses `shape[1]` and is correct. **Exam safety:** if a quiz asks how the function measures complexity, answer the course's way, the **number of columns (coefficients) in the design matrix**; `shape[1]` is the axis that gives it.

Test it on one formula before looping, which is generally good practice:

```python
eval_cv(formulas[0], 0, data)
```

Verified output:

```text
   model_name  model_formula  num_coefs  fold_id           rmse
0           0  price_usd ~ 1          1        0  104715.328722
1           0  price_usd ~ 1          1        1   96785.749927
2           0  price_usd ~ 1          1        2  102733.480694
3           0  price_usd ~ 1          1        3   97636.407429
4           0  price_usd ~ 1          1        4  109093.823343
```

## Loop it, then stack the results

A `for` loop over the formula list, using **`enumerate`**, which the lecture flags as possibly new: it hands you both the item and its index, "kind of an alternative to range." The index doubles as the model name, so model 0 is the first formula and model 6 the last. (The lecture's spoken list of indices trails off early; with seven formulas `enumerate` runs 0 through 6.)

```python
results_list = []
for i, formula in enumerate(formulas):
    results_list.append(eval_cv(formula, i, data))

results_df = pd.concat(results_list)
```

Each list item is a five-row DataFrame; a list comprehension would work too, but the loop is clearer. Verified: 7 items in the list, and the concatenated `results_df` is **35 rows by 5 columns** (7 models times 5 folds). The lecture's aside is worth taking: `reset_index()` would be good practice here, because `concat` stacks seven copies of the index 0 through 4.

## Visualize and choose

The comparison plot is a seaborn **point plot** with model name on the x-axis and RMSE on the y-axis, points not connected by lines, and error bars set to a **68% interval instead of the usual 95%**:

```python
sns.catplot(data=results_df, x='model_name', y='rmse', kind='point',
            linestyle='none', errorbar=('ci', 68), height=4)
```

Keeping all five fold scores (instead of averaging them away immediately) is what makes the interval possible: the dot is the average across the test folds, the whisker shows the spread of that average.

> **Why 68% and not 95%.** The lecture: "The convention for confidence intervals for between models is actually just one standard deviation, or one standard error, instead of two standard errors. So instead of 95%, we're going to do 68%." A 95% interval is roughly two standard errors wide on each side; one standard error corresponds to about 68%. Narrower bars make it harder for a mediocre model to look tied with the best one.
>
> **Two precision notes, flagged.** First, "one standard deviation" and "one standard error" are not the same quantity; the relevant one is the **standard error of the mean** (the fold-to-fold standard deviation divided by the square root of 5). Second, `errorbar=('ci', 68)` asks seaborn for a **bootstrapped** 68% interval, not a literal one-standard-error bar; `errorbar=('se', 1)` is that (Verified: it returns exactly mean plus or minus the standard error). The two land in the same place on this data, but the bootstrap is resampled on every draw, so those whisker ends wobble slightly run to run. **Exam safety:** the course's framing (one standard error instead of two standard errors, so 68% instead of 95%) is what a quiz would key on.

Verified mean RMSE across the five test folds, with the analytic one-standard-error band (lower is better for RMSE; for R-squared and many other metrics higher is better):

```text
model  formula                                        coefs   mean RMSE   mean +/- 1 SE
  0    price_usd ~ 1                                    1     102,193      99,910 - 104,476
  1    price_usd ~ size_m2                              2      83,688      81,234 -  86,142
  2    price_usd ~ city_center                          2      73,214      68,492 -  77,936
  3    price_usd ~ size_m2 + city_center                3      36,225      34,073 -  38,377
  4    price_usd ~ size_m2 * city_center                4      35,465      33,186 -  37,743
  5    price_usd ~ np.power(size_m2, 2) + city_center   3      39,691      36,209 -  43,174
  6    price_usd ~ np.power(size_m2, 2) * city_center   4      38,534      35,641 -  41,427
```

(The bootstrap is not seeded, so the drawn whiskers move a little each time. Verified over 25 redraws: model 4's band ran from a lower end of 33,174 to 33,558 up to an upper end of 37,372 to 37,711, and model 3's from 33,886 to 34,418 up to 38,032 to 38,564, both hugging the analytic bands above.)

## Reading the plot

Walking it the way the lecture does:

- **Model 0 is terrible, and that is the point.** Mean RMSE about 102,193, which is essentially the standard deviation of `price_usd` itself (101,571, Verified). Predicting the mean and nothing else leaves you with an error the size of the target's own spread. Every other model improves on it, which is the first thing you want to see.
- **Models 1 and 2 make progress** (83,688 and 73,214) but are still far off; one variable alone is not enough.
- **Models 3, 4, 5, and 6 are all "really good"**, clustered in the 35k to 40k range.
- **Model 4 is the best on raw score** (35,465), the size-by-city-center interaction. The lecture says this "should be expected. The data was actually generated based on that," so the dataset is synthetic and the interaction was built into it.
- **Models 5 and 6 do not do quite as well** (39,691 and 38,534) and are dropped. Verified: both means sit **above** model 4's one-standard-error upper bound of 37,743, so the tie rule below does not rescue them.
- **Model 3 is the pick.** Its mean, 36,225, is inside model 4's interval (Verified: 36,225 < 37,743; also verified inside seaborn's bootstrapped band on 25 of 25 redraws). It has **3 coefficients** instead of 4, and the lecture calls it "nice and simple." Since there is no meaningful performance difference, the simpler model wins.

> **The rule in one sentence:** find the best mean score, then choose the **simplest** model whose mean falls within one standard error of it. (Added, not the lecture's wording: this is the standard machine learning heuristic called the **one standard error rule**. The lecture applies exactly this reasoning out loud without naming it.)

## After you choose

Last point of the lecture, and an easy one to get wrong: once cross-validation has selected a model, "if you are using this model in the real world, it's okay to just train it on all of your data." Cross-validation exists to **estimate performance on new data and to select models**, not to produce the deployed fit. The five fitted models from the folds are throwaway; the chosen formula gets refit once on everything.

---

## Quiz-ready facts

- Cross-validation compares models by their performance on **test data / unseen data**, which is the basis for model selection.
- The candidate space is large even for linear regression: variable subsets, interactions, nonlinear transformations, and mixtures of them.
- Two competing goals: **generalize well** (avoid overfitting) and **stay simple** (fewer features, more explainable, more generalizable patterns).
- Overfitting: a complex model can fit **training** data very well and still perform worse on **test** data.
- **`price_usd ~ 1`** in statsmodels means **intercept only**. It ignores the inputs and predicts one flat number (the training-fold mean), so it is the baseline every real model must beat. Verified: mean CV RMSE 102,193, about the same as `price_usd`'s own standard deviation (101,571).
- The lecture's helper takes **(formula, model name, dataset)** and returns **RMSE per test fold**; you could swap the metric to R-squared by changing the scoring string.
- Pipeline inside: **patsy `dmatrices`** extracts the design matrices from the statsmodels formula, **`KFold(n_splits=5)`** defines the splits, **`LinearRegression`** is the estimator, **`cross_val_score`** runs it. **5 splits produce 5 scores.**
- scikit-learn scorers are **higher is better**, so RMSE is requested as **`neg_root_mean_squared_error`** and negated back to positive.
- Model complexity is measured by the **number of columns (coefficients) in the design matrix**, i.e. `shape[1]`.
- **`enumerate`** yields both the item and its index while looping, an alternative to `range`. Here the index becomes the model name, 0 through 6.
- `pd.concat` on the seven five-row frames gives one **35-row** results DataFrame; `reset_index()` afterward is good practice.
- Compare models with a **point plot** of mean RMSE, error bars at a **68% (one standard error) interval**, not 95% (about two standard errors). Narrower is the between-model convention.
- **RMSE: lower is better.** R-squared and many other metrics: higher is better.
- **Verified mean RMSE** (5-fold, `shuffle=True, random_state=9`): model 0 = 102,193; 1 = 83,688; 2 = 73,214; 3 = 36,225; 4 = 35,465; 5 = 39,691; 6 = 38,534. Coefficient counts: 1, 2, 2, 3, 4, 3, 4.
- **The selection rule:** best mean score is model 4 (interaction), but **model 3 (additive, 3 coefficients) is chosen** because it falls inside model 4's one-standard-error interval and is simpler.
- The apartment data is **synthetic**: the lecture says it "was actually generated based on" the interaction, which is why model 4 tops the raw scoring.
- In a formula, `size_m2**2` does **not** square anything (`**` sets the maximum interaction degree, an R inheritance); use **`np.power(size_m2, 2)`** or `I(size_m2**2)`.
- **After selection, refit the chosen model on all the data.** Cross-validation is for estimating performance and selecting, not for producing the final deployed model.

---

> **See also:** "Comparing Models with Cross-Validation - Notebook (Cliff Notes)" (the companion notebook this lecture narrates, with the full function body, the complete results table, and the point plot), "Cross-Validation with Scikit-learn (Cliff Notes)" (where the `dmatrices` plus `KFold` plus `cross_val_score` handoff is built up step by step), "Introduction to Cross-Validation (Cliff Notes)" (why held-out folds beat a single train/test split), "Root Mean Squared Error (RMSE) - Notebook (Cliff Notes)" (what the metric on the y-axis actually measures), and "Comparing Multiple Linear Regression Models - Notebook (Cliff Notes)" (the same select-the-simplest-model question worked through in a notebook).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
