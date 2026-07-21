# Visualizing Uncertainty on Linear Regression Predictions: confidence intervals vs prediction intervals

**Module 12 Cliff Notes** | Source: lecture transcript "Visualizing Uncertainty on Linear Regression Predictions"

---

## TL;DR

- A prediction is only half the story: you also want to **show how sure you are**. This lecture draws **two uncertainty bands** on a regression line, a **confidence interval** and a **prediction interval**.
- **Confidence interval (CI)** = uncertainty about the **mean** output at each input value. Remember, linear regression predicts the **mean** outcome at a given x, so the CI is the natural band around that. It is **narrow** and it **shrinks** where the data is informative.
- **Prediction interval (PI)** = uncertainty about a **single new observation** at each input value. It is **always wider** than the CI because it must also absorb the **spread of the residuals** (how far real points scatter from the line), not just uncertainty in the mean.
- The workhorse is **`model.get_prediction(grid).summary_frame()`** (not plain `predict`). Its columns are **`mean`, `mean_se`, `mean_ci_lower`, `mean_ci_upper`, `obs_ci_lower`, `obs_ci_upper`**. The `mean_ci_*` pair is the CI; the `obs_ci_*` pair is the PI.
- Draw the bands with **`plt.fill_between`**: a **gray** ribbon from `mean_ci_lower` to `mean_ci_upper` (the CI) and a **wider orange** ribbon from `obs_ci_lower` to `obs_ci_upper` (the PI), with the fitted line and the training scatter on the same axes.

> **One-line takeaway:** the **confidence interval** bounds where the **average** outcome sits (narrow), while the **prediction interval** bounds where the **next individual** outcome could land (wide); `summary_frame()` hands you both.

---

## The setup: same model, a new way to predict

We reuse the **food truck** model from earlier: predict **profit** from **population** with one input, fit through the statsmodels formula API.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

model = smf.ols(formula="profit ~ population", data=df).fit()
```

The fit is instant on this tiny dataset. As before, the **population coefficient is positive with a very low p-value**, so we can be confident the slope is significantly different from zero. What is new is **how we make and visualize the predictions**.

## Step 1: build a visualization grid

Instead of predicting on the training rows, we build a smooth grid of **fake input values** that spans the range of the real population column, so the eventual line and ribbons are smooth:

```python
grid = pd.DataFrame({
    "population": np.linspace(df["population"].min() - 0.1,
                              df["population"].max() + 0.1,
                              101)
})
```

- `np.linspace(low, high, 101)` gives **101 evenly spaced points**. Asking for a round number **plus one** (100 + 1) gives you a clean start, a clean end, and round numbers in between.
- We nudge a little **below the min and above the max** just to capture a touch more range.
- The DataFrame has **one column named exactly like the model input** (`population`), which is what the model needs to predict.

## Step 2: get_prediction().summary_frame(), the key move

Here is the difference from before. Rather than `model.predict(grid)`, which returns only point predictions, we call **`get_prediction`** and then ask for its **`summary_frame`**, which crucially includes the **intervals**:

```python
pred = model.get_prediction(grid)
sf = pred.summary_frame()   # default level is 95% (alpha=0.05)
```

The naming is admittedly odd (the lecture jokes "don't ask me why they're named like that"), but `summary_frame` returns one row per grid point with these **six columns**:

| Column | What it is |
|---|---|
| **`mean`** | the **predicted output**, i.e. the mean outcome at that input (this is the fitted line) |
| **`mean_se`** | the **standard error** of that mean prediction |
| **`mean_ci_lower`** / **`mean_ci_upper`** | the **confidence interval** on the mean (the narrow band) |
| **`obs_ci_lower`** / **`obs_ci_upper`** | the **prediction interval** on an individual observation (the wide band) |

Naming caveat worth flagging: statsmodels labels the **prediction-interval** bounds **`obs_ci_lower` / `obs_ci_upper`** ("observation CI"). The lecture calls them "these weird columns" and notes "the observed ci lower is what statsmodels calls it." Do not let the `..._ci_...` in the name fool you: **`obs_ci_*` is the prediction interval, not a second confidence interval.**

To change the level, pass `alpha` (for example `summary_frame(alpha=0.10)` for 90%). Smaller alpha gives a wider band; the default `alpha=0.05` gives the standard 95%.

## Step 3: the confidence interval on the mean

The `mean_ci_lower` / `mean_ci_upper` band answers: **where does the true mean output lie at each input value?** By the central limit theorem, if you resampled repeatedly, the mean prediction would follow a Gaussian whose standard deviation is the **standard error** (`mean_se`). The interval is **roughly two standard errors** on each side of the mean:

> CI half-width is about **2 x standard error** (precisely a t-multiplier, about **1.96** for large samples; in the lecture's small-data fit it works out to almost exactly 2.0).

Two properties to remember:

- The band is **narrow** relative to the scatter of the raw points, because estimating a **mean** is far easier than pinning down an individual value.
- The band is **not the same width everywhere**. It is **tightest near the center of the data** and flares out toward the edges of the input range.

**Clarification on why the band tightens (source runs a little loose here).** The lecture explains the narrowing as "more data points -> smaller confidence interval," pointing to input values where "we have lots of different cities at that value." That intuition is fine as far as it goes (more observations do lower the standard error through the 1/n term), but the precise reason the ribbon pinches in one place and flares elsewhere is that the standard error of the mean prediction is **smallest at the mean of the input variable and grows with distance from it**. In the food truck data the cities cluster toward the lower populations, so the densest region also happens to sit near the input mean, which is why "more data" and "narrowest band" line up there. **If a quiz keys to the course's wording, answer that the confidence interval shrinks where there are more observations.** The fuller statement is that it is narrowest near the mean of x and widens toward the extremes.

## Step 4: the prediction interval on an individual observation

The CI on the mean does **not** tell you where a single real data point might fall, because it ignores the **spread of the residuals**. Recall the linear regression assumption: the **errors are drawn from a normal distribution** with some standard deviation, where an "error" is how far an observed point sits from the line. Your estimate of the *mean* does not capture that scatter.

The **prediction interval** (`obs_ci_lower` / `obs_ci_upper`) does. It bounds where an **individual observed outcome** could land at each input, so it must add the residual spread on top of the mean's uncertainty. Consequences:

- The PI is **always wider than the CI** (verified true at every grid point), often dramatically so.
- The PI stays **roughly constant in width** across the input range, because it is dominated by the residual standard deviation rather than by the mean's leverage term.
- **Some training points will still fall outside** the PI. That is expected: a 95% band is meant to contain most, not all, observations, and there are always some outliers.

**The temperature analogy (from the lecture).** Suppose you record the temperature on the same calendar day every year for 100 years:

- The **confidence interval** bounds where the **average** temperature for that day should sit across the 100 years.
- The **prediction interval** bounds where **any single year's measurement** should land (how much January 23 in 1995 might differ from January 23 in 2013), regardless of what the average is.

## Step 5: drawing both ribbons

Plot the fitted line, then lay the two bands underneath with **`fill_between`**, and drop the training scatter on top, all on one small (roughly 4 by 4) axes:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(4, 4))
x = grid["population"]

# fitted line: the mean prediction
ax.plot(x, sf["mean"], color="blue")

# wider PREDICTION interval ribbon (orange)
ax.fill_between(x, sf["obs_ci_lower"], sf["obs_ci_upper"],
                color="orange", edgecolor="orange", alpha=0.3)

# narrow CONFIDENCE interval ribbon (gray)
ax.fill_between(x, sf["mean_ci_lower"], sf["mean_ci_upper"],
                color="gray", alpha=0.3)

# training data on top
sns.scatterplot(data=df, x="population", y="profit", ax=ax)
ax.set_xlabel("population")
plt.show()
```

- `fill_between(x, lower, upper, ...)` paints the ribbon between the two bound arrays. `alpha` sets transparency so the overlap stays readable.
- Result: a thin **gray** band hugging the blue line (the CI on the mean) sitting inside a much broader **orange** band (the PI on individuals), with most, but not every, real point inside the orange.

---

## Quiz-ready facts

- **Linear regression predicts the mean outcome at a given x**, which is exactly what the **confidence interval** puts a band around.
- **Confidence interval (CI) = uncertainty on the MEAN** output. **Prediction interval (PI) = uncertainty on an INDIVIDUAL** observed output. **PI is always wider than CI.**
- Use **`model.get_prediction(grid).summary_frame()`**, not plain `predict`, to obtain the intervals.
- `summary_frame()` columns: **`mean`, `mean_se`, `mean_ci_lower`, `mean_ci_upper`, `obs_ci_lower`, `obs_ci_upper`**.
- **`mean_ci_*` = confidence interval** on the mean. **`obs_ci_*` = prediction interval** on an observation (statsmodels names it "obs_ci" but it is the PI, not a second CI).
- CI half-width is about **2 x standard error** (t-multiplier, near **1.96** for large samples). Default level is **95%** (`alpha=0.05`); smaller `alpha` = wider band.
- **CI is narrowest near the mean of the input** and widens toward the extremes; the course phrases this as "shrinks where there are more observations."
- **PI must include the residual spread** (errors are assumed normal), so it is wide and stays roughly constant across the input range; some points legitimately fall outside it.
- Draw both with **`plt.fill_between`**: gray for the CI, a wider orange for the PI, over the fitted line and the training scatter.

---

> **See also:** "Visualizing Uncertainty on Predictions - Notebook (Cliff Notes)" (the companion lab that runs this exact `get_prediction().summary_frame()` workflow), and "P-values and Confidence Intervals on Coefficient Estimates in Linear Regression (Cliff Notes)" (confidence intervals applied to the coefficients rather than the predictions).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
