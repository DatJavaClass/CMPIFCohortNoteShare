# Module 11 - Cliff Jumper Notes

*One continuous lesson stitched from the seven Module 11 cliff notes (Introduction
to Predictive Analytics; Predictive Models; Linear Regression Formula; Linear
Regression Coefficients; Linear Regression Example; Linear Regression Assumptions,
Normality of Errors; and the Linearity lab notebook). The through-line: **Module 11
is where the course turns from describing data to predicting from it, and it does
so in six moves.** First set the **goal**, learn a function from inputs to an
outcome (**predictive analytics**). Then pick up the **tool**, a **model**, a
simplified lens fit from data. Then meet this course's one model, the **linear
regression formula** (a glorified `y = mx + b`). Then learn to **read** its two
coefficients (**intercept** and **slope**). Then **run** it in code (predicting
river depth from rainfall). Finally **check** the assumptions it rests on
(**normality of errors**, plus the full four assumptions and what "**linear**"
really means). Module 10 got the continuous data ready to model; Module 11 builds
the first model.*

> All code here targets **numpy**, **pandas**, **matplotlib**, and **seaborn** with
> the standard aliases (`np`, `pd`, `plt`, `sns`). The datasets are **synthetic**,
> generated in the notebook with `np.random` so the relationships are known in
> advance. Described plot readings (a straight line, a wavy scatter, a parabola)
> are the observations the source notes report, not fresh reads of rendered images.
> All numbers below were reproduced by execution (numpy 2.4.6, pandas 3.0.3); the
> `np.random.seed` legacy draws are stable across versions, so the generated values
> match the notebook exactly.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 1. The map: predictive analytics and its first model

Everything before this module **described** and **visualized** data. Module 11
pivots to **prediction**: learning a relationship well enough to estimate an
outcome you have not seen yet. The whole module is one arc from "what am I trying to
do" down to "how do I know the model is trustworthy," in six moves:

| The move | The idea | The concept | Section |
|---|---|---|---|
| **Goal** | learn a function from inputs to an outcome | **predictive analytics** | 2 |
| **Tool** | a simplified lens fit from data | **the model** | 3 |
| **Model** | this course's one model, `ŷ = β₀ + β₁x + ε` | **linear regression formula** | 4 |
| **Read** | what the two numbers mean | **intercept and slope** | 5 |
| **Run** | predict and plot in code | **the river-depth example** | 6 |
| **Check** | what the model assumes | **the four assumptions** | 7 |

Sections 2 and 3 set up the vocabulary (X, Y, F, model, regression); sections 4
through 6 are linear regression itself, from formula to code; section 7 is the
fine print that tells you when linear regression is even allowed. Walk them in
order.

---

## 2. The goal: predictive analytics

**Predictive analytics** is the analysis of data with the goal of **learning a
relationship** between an **outcome** and one or more **inputs**:

- The **outcome** is denoted **Y**. It goes by many names: output, **response**,
  **target**, or **dependent** variable.
- The **inputs** are denoted **X** (capital, because there can be many of them, like
  a matrix), also called **independent** variables.

The job is to learn a **function F** that maps the inputs to the outcome, learned
**from patterns in the data**, not written by hand:

```text
Y = F(X)
```

Because the computer learns that association, this is also called **predictive
modeling** and **supervised learning**. The variables themselves are just **columns
in a data table**, and a big part of the real work (all of the earlier course's
merging, cleaning, and pandas wrangling) is getting data into an **analytics-ready**
shape first.

**Two goals, straight from Module 1's definition of data science.** Predictive
analytics lets you both:

1. **Describe** relationships between an outcome and its inputs (how much does a
   fresh coat of paint actually change a house's selling price?).
2. **Predict** the outcome for cases **not yet observed** (the sale price of a home
   about to hit the market).

**When it is worth doing.** Predictive analytics fits **meaningful** phenomena that
sit between two useless extremes:

- **Not deterministic.** A grocery bill is just quantity times price; there is no
  variation to learn, so use a calculator. But **how much** of each product sells by
  day, season, or shelf position **is** learnable, because human behavior is neither
  fully determined nor fully random.
- **Not completely random.** Predicting an athlete's jersey number from their name
  is hopeless (no logical link). Predicting **which team wins** from player and team
  stats is both possible and meaningful.

> **The caveat that outranks all the machinery: correlation is not causation.** The
> relationships predictive analytics finds are **correlations**, and they need a
> **logical reason** to be meaningful. The textbook trap is the strong measured
> relationship between **PhDs awarded in US civil engineering** and **US per-capita
> mozzarella consumption**: both merely rose over time, so they track a hidden third
> factor (time), not each other. A found relationship is not a real one until there
> is a reason for it.

---

## 3. The tool: models, and the two kinds

To do any of that, you build a **model** (a **predictive** or **statistical**
model): a **simplified little world** that isolates a few variables and relates them
through a **mathematical equation**. You **fit** the model (find its **parameters**)
from a dataset of observations, then judge it by its **fit** to the data.

**Models are lenses.** They bring a few relationships into focus and **blur out** the
rest of reality's complexity. The lecturer's image is a **diorama of New York City**:
it cannot capture every brick, but it still shows the useful pattern. So the goal is
never perfect prediction, only **partial** prediction, which is why **error is always
present**.

**The process has two steps:**

1. **Choose the model type** (a family of functions; your EDA can inform the choice).
   This course uses exactly one: **linear regression**.
2. **Learn the parameters** (in linear regression, the **coefficients**)
   **automatically from data**, via a learning algorithm that **minimizes the
   prediction error**. The optimization details are out of scope; that it is
   happening is not.

**Two kinds of predictive model, decided by the outcome's data type.** Just as a
variable's type drives which plot you pick, the **outcome's** type drives the model:

| Kind | Predicts | Outcome type | Example |
|---|---|---|---|
| **Regression** | a **continuous** outcome | numeric | selling price, river depth |
| **Classification** | a **categorical** outcome | category | dog / cat / bird, yes / no |

Predictive modeling is **supervised learning** (the root of **machine learning**),
and this course **focuses on regression**.

---

## 4. The model: the linear regression formula

**Linear regression** is this course's model. Because it is **regression**, the
output **Y is continuous**. It is really a **glorified `y = mx + b`**, the
slope-of-a-line equation. For a **single input**:

```text
ŷ = β₀ + β₁x + ε
```

Break it into parts:

- **ŷ (Y-hat):** the model's **estimate / prediction** of the output, not the true
  observed value (that is what the hat means).
- **β₀ and β₁:** the **coefficients (parameters)**, the **heart of the model**,
  **estimated from data**. Having exactly **two** coefficients is what makes this the
  **single-input (simple)** case; more inputs are possible but not covered yet.
- **ε (epsilon):** the **error**, always present.

**Per data point.** With **N** rows (input-output pairs), the prediction for row n
drops the error term:

```text
ŷₙ = β₀ + β₁xₙ
```

Plug in that row's input `xₙ` (say 11 mm of rain that day) and out comes a
prediction.

> **A variant worth recognizing: linear regression predicts the *mean*.** You will
> see the left side written **μₙ** (mu, "mean") instead of ŷₙ:
> `μₙ = β₀ + β₁xₙ`. This is not a different model; it spells out that linear
> regression targets the **average** output at a given X, not any single
> observation. Even at the **same** X the outcome varies (the same rainfall can leave
> the river at slightly different depths), and the model aims at the average. Hold on
> to this, section 7's normality assumption is exactly a statement about how the
> individual observations scatter around that mean.

---

## 5. Reading the model: intercept and slope

The two coefficients each have a name and a job:

- **β₀ = the intercept** (not multiplied by any input).
- **β₁ = the slope** (multiplied by the input x).

**The slope β₁ sets the trend.** Holding the intercept at 0 and varying the slope:

- **Slope 0** = **no trend**: a flat line, the prediction never moves as x changes.
- **Larger slope = steeper climb.** At the same x, a higher slope gives a larger ŷ
  (slope 2 is more vertical than slope 0.5).
- **Slope 1** ⇒ +1 in ŷ per +1 in x. **Slope 2** ⇒ +2 in ŷ per +1 in x.
- **Negative slope** ⇒ ŷ **falls** as x rises, more steeply the more negative it is.

The general reading to memorize:

> A coefficient is **how much the outcome changes for a one-unit change in the input,
> holding all other inputs constant.**

**The intercept β₀ sets the starting point.** Because it is not multiplied by x:

> The **intercept** is the **predicted outcome when the input is 0** (the river depth
> at zero rainfall; the base salary at zero years of experience).

Visually, the **intercept anchors where the line crosses x = 0** and the **slope
tilts it**. Any intercept pairs with any slope: both positive gives a line crossing
above 0 and climbing; both negative gives a line crossing below 0 and dropping
further as x grows.

---

## 6. Running the model: river depth in code

A worked example: predict **river depth (mm)** from a single input, **rainfall**,
with coefficients treated as **already estimated from data** (later lessons estimate
them with packages). The demo uses **intercept β₀ = 4780** mm (about 4.78 m, the
assumed depth at zero rain) and **slope β₁ = 2.89** (each unit of rainfall adds 2.89
mm of predicted depth).

**Build a grid of inputs.** Use `np.linspace`, which (unlike `range`) **includes the
endpoint**:

```python
x = np.linspace(0, 25, 21)          # 21 points from 0 to 25, inclusive
vis_df = pd.DataFrame({'rainfall': x})
```

The count is **21**, not 20, on purpose: aiming for 20 intervals and adding one point
gives round numbers (0, 1.25, 2.5, ..., 25).

**Apply the formula to the whole column at once** (vectorized):

```python
intercept = 4780
slope = 2.89
vis_df['predicted_river_depth'] = intercept + slope * vis_df['rainfall']
```

For each rainfall value this multiplies by the slope and adds the intercept. At about
**10 mm** of rain the prediction is `4780 + 2.89 * 10 = 4808.9` mm (~4.81 m), matching
the lesson's reading.

**Plot the model line** with a figure-level seaborn line chart; it comes out
**straight**, because the model is linear:

```python
sns.relplot(data=vis_df, x='rainfall', y='predicted_river_depth', kind='line', height=4)
plt.show()
```

**Overlay observed vs predicted.** Pretend field measurements were taken at each
rainfall total; the real observations will **not** sit exactly on the line. To layer
two plots, switch from the figure-level call to seaborn **axes-level** calls sharing
one `ax` you create yourself:

```python
fig, ax = plt.subplots(figsize=(...))
sns.scatterplot(data=vis_df, x='rainfall', y='river_depth', ax=ax)          # observed
sns.lineplot(data=vis_df, x='rainfall', y='predicted_river_depth', ax=ax)   # predicted
plt.show()
```

The line acts like a **best-fit line**, but the points **scatter around** it. That
spread is the model's **error**, because river depth depends on far more than a single
day's rainfall.

> **Figure-level vs axes-level, in one line:** the lone model-line plot used a
> **figure-level** call (`relplot`); to draw two plots on the *same* axes you drop to
> **axes-level** calls (`scatterplot`, `lineplot`) and hand each the shared `ax` from
> `plt.subplots`. This is the same figure/axes distinction the visualization modules
> established, now doing real work.

---

## 7. Checking the model: the four assumptions

That leftover scatter in section 6 is exactly what the assumptions are about. Linear
regression rests on **four**:

1. **Linearity of the mean.** The mean output is a **linear combination of
   predictors**, `μₙ = β₀ + β₁xₙ`. Individual points scatter, but the **average**
   follows a straight line.
2. **Normality of errors.** `εₙ ~ Normal(0, σ)` (detailed just below).
3. **Independence.** Datapoints are independent and their **errors are uncorrelated**;
   one observation tells you nothing about another. This is **violated in time-series**
   data, where current values depend on past ones.
4. **Constant variance (homoscedasticity).** The **variance of Y is constant across
   all x**, so the scatter around the line is a **uniform band**.

The two lessons dig into the first two.

### 7a. Normality of errors, up close

An **error** (also called a **residual**) is the **difference between the predicted
output and the observed output**. The assumption:

```text
ε ~ Normal(0, σ)
```

- The **tilde `~`** reads **"is distributed as."** The errors are **Gaussian**, with
  **mean 0** and **standard deviation σ**.
- **σ is a single estimated number** for the whole dataset and **does not vary with
  x** (which is precisely the constant-variance assumption above).
- **Mean 0** means observations cluster **around** the prediction (an error of 0 is
  an exact hit). With `σ = 0.5` and ribbons at ±1σ, ±2σ, ±3σ, **most points fall
  within ~1σ** of the line and **few** land 2 to 3σ away. Picture a little Gaussian
  standing on its side at each x.

**Why more data helps.** The more observations at a given x, the surer you are of the
average output there, because the **standard error of the mean** shrinks with sample
size:

```text
standard error = σ / √n
```

Your data is always just a **sample** of the true **population**, so you can always
collect more. Under the **Central Limit Theorem**, repeated sampling makes your
**estimate of a coefficient (the slope)** itself **Gaussian**, so you can put a
**confidence interval** on it and state, descriptively, "there is a **positive**
(or **negative**) relationship here." That confidence interval is the **descriptive
insight** that predictive analytics buys you.

### 7b. What "linear" actually means (and feature transformation)

The linearity lab fixes the most common misconception. "Linear" does **not** mean "a
straight best-fit line in the raw data":

> **"Linear" means the mean output is linear in the coefficients (β), not in the
> input data.** You may transform the input any way you like, and as long as the
> **coefficient stays outside** the transformation, it is still a linear model.

So `y = β₀ + β₁ sin(x)` is **still linear**, because β₁ is outside the `sin`. The lab
demonstrates this on three synthetic datasets:

```python
# (a) genuinely linear: true intercept 1, slope 1
np.random.seed(42)
data = pd.DataFrame({'x': np.linspace(-2, 2, 50)})
data['y'] = 1 + data['x'] + np.random.normal(0, 0.5, size=50)
data['estimated_y'] = 1 + data['x']     # same line the data came from -> clean fit
```

- Verified `head()` (seed 42): the first `y` values are
  `-0.751643, -0.987499, -0.512890, 0.006413, -0.790546`, and `estimated_y` is
  `-1.0, -0.918367, -0.836735, -0.755102, -0.673469`. The red `estimated_y` line runs
  right through the cloud.

```python
# (b) sine wave -> transform x into sin(x)
data = pd.DataFrame({'x': np.linspace(0, 7, 100)})
data['y'] = np.sin(data['x']) + np.random.normal(0, 0.2, size=100)
data['x_transformed'] = np.sin(data['x'])   # y vs sin(x) is now a straight line
```

```python
# (c) parabola -> transform x into x**2 (polynomial regression)
np.random.seed(9)
data = pd.DataFrame({'x': np.linspace(-2, 2, 50)})
data['y'] = 3 * np.power(data['x'], 2) + np.random.normal(0, 0.5, size=50)
data['x_transformed'] = np.power(data['x'], 2)   # y vs x**2 is now linear
```

- Verified for (c): `x` runs -2 to 2, `y` runs roughly **-0.64 to 12.41**, and `x²`
  runs **0 to 4**. Setting `f = x²`, the model is just `y = β₀ + β₁f`, ordinary linear
  regression on a new feature. **Any** transformation can go into `f`.

**When it is truly nonlinear.** Only when a coefficient sits **inside a function or is
multiplied by another coefficient**:

```text
y = β₀ + e^(β₁ x)      # β₁ INSIDE the exponential  -> NOT linear, cannot use LR
y = β₀ + β₁ e^x        # β₁ OUTSIDE                  -> linear (transform x -> exp(x))
```

> **The payoff, and the bridge back to Module 10:** you do **not** need a straight
> line in the raw data to use linear regression. Curved data just needs the right
> transform (sin, log, exp, powers) first, exactly the kind of shape you would spot in
> the EDA and correlation work of the earlier modules. The "linear" constraint is on
> the **coefficients**, not on the picture.

---

## One-paragraph recap

Module 11 turns from describing data to **predicting** from it, in six moves.
**Predictive analytics** learns a function **F** from inputs **X** to an outcome
**Y** (from patterns in data; also called predictive modeling / supervised learning),
to both **describe** relationships and **predict** unseen cases, for phenomena that
are neither deterministic nor random, and always remembering that the relationships
found are **correlations, not causation**. The tool is a **model**, a simplified lens
**fit from data**, chosen by type (here **linear regression**) and then fit by
**learning its coefficients to minimize error**; models split into **regression**
(continuous outcome) and **classification** (categorical), and this course does
regression. The model is `ŷ = β₀ + β₁x + ε` (a glorified `y = mx + b`), where **ŷ** is
the estimate, **β₀/β₁** are coefficients learned from data, **ε** is error, and the
formula really predicts the **mean** output **μ** at a given X. Its coefficients read
as **intercept** (predicted outcome at input 0) and **slope** (change in outcome per
one-unit change in input, all else held constant). In code, predicting river depth
from rainfall (`intercept 4780`, `slope 2.89`) means building inputs with
`np.linspace(0, 25, 21)` (21 points, endpoint included), computing
`intercept + slope * rainfall` vectorized, drawing a straight **figure-level** line,
then overlaying observed points as an **axes-level** scatter that scatters **around**
the line (the error). Finally the model rests on **four assumptions**, linearity of
the mean, **normality of errors** (`ε ~ Normal(0, σ)`, mean 0, constant σ, most points
within ~1σ, `standard error = σ/√n`, and via the **Central Limit Theorem** a
confidence interval on the slope), independence (violated by time series), and
constant variance, where **"linear" means linear in the coefficients, not the input**,
so a sine or `x²` **feature transformation** keeps a curved relationship inside linear
regression, while a coefficient trapped inside a function (`y = β₀ + e^(β₁x)`) does
not.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
