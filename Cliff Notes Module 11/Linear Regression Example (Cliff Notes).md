# Linear Regression Example: predicting river depth in code

**Module 11 Cliff Notes** | Source: lecture transcript "Linear Regression Example" (a walk-through of a Jupyter notebook)

---

## TL;DR

- A worked linear-regression example: predict **river depth (mm)** from a single input, **rainfall**, using two **pre-supplied** coefficients (assume they were already estimated from data; later lessons estimate them with packages).
- The coefficients: **intercept β₀ = 4780** (river depth in mm at zero rainfall, about 4.78 m) and **slope β₁ = 2.89** (each +1 of rainfall adds 2.89 mm of predicted depth).
- Build a **visualization grid** of evenly spaced inputs with **`np.linspace(0, 25, 21)`** (21 points because linspace includes the endpoint; 20 nice round intervals), put it in a DataFrame, then compute predictions with the formula **`intercept + slope * rainfall`** applied to the whole column at once.
- **Plot the model line** with a seaborn **line chart** (a figure-level `relplot`/`lineplot` style call); it comes out as a **straight line** because the model is linear.
- **Overlay observed vs predicted:** put pretend **observed** depths in a DataFrame, then draw a **scatter** of the observations and the **line** of predictions on the **same axes** (both seaborn **axes-level** calls sharing one `ax`). The scatter spreads **around** the line, which is the model's **error**.

> **The whole point of the overlay:** the prediction line acts like a best-fit line, but the observed points do not sit exactly on it. That spread is the model's error, because real river depth depends on far more than a single day's rainfall.

---

## Setup: the model and its coefficients

In predictive analytics you must **choose what to predict**. Here the output is **river depth in millimeters**, predicted from the single input **rainfall** (if the forecast calls for so much rain, how high will the river get?). Many other variables really drive river depth, but the demo simplifies to **two coefficients**.

Recall the formula, where **ŷ** is the estimate of the **mean** output at input X:

```text
predicted_river_depth = β₀ + β₁ * rainfall
```

The coefficients are treated as **already estimated from data** (later lessons show packages that actually estimate them):

- **Intercept β₀ = 4780** mm (about **4.78 m**): the assumed river depth when **rainfall = 0** (that whole `β₁ * rainfall` term cancels), roughly how high the river normally sits.
- **Slope β₁ = 2.89**: for **every one-unit increase in rainfall**, the predicted depth rises **2.89 mm**.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

## Build a range of inputs (the visualization grid)

To see what the model does across a range, generate evenly spaced inputs. The lecturer uses **`np.linspace`** rather than `range`:

```python
x = np.linspace(0, 25, 21)   # 21 evenly spaced values from 0 to 25, inclusive
```

- `linspace(start, stop, num)` **includes the endpoint** (25 is included), which `range` would not.
- The count is **21**, not 20, on purpose: aiming for "20 intervals" and adding 1 gives **round numbers** (0, 1.25, 2.5, ... 25). A handy habit is to pick 20, 100, 10, and then add one.

Put it in a DataFrame (the lecturer calls it `vis_df`, a "visualization DataFrame") with one column for now:

```python
vis_df = pd.DataFrame({'rainfall': x})
```

> **Jupyter tip from the lesson:** for long output, **right-click the output area and choose "enable scrolling"** so you can scroll the output; hovering over the white space outside it still scrolls the notebook itself.

## Compute predictions with the formula

Define the coefficients and apply the formula to the **entire column at once** (vectorized), creating a new prediction column:

```python
intercept = 4780
slope = 2.89
vis_df['predicted_river_depth'] = intercept + slope * vis_df['rainfall']
```

For each rainfall value, this multiplies by the slope and adds the intercept, filling the column with predictions. For example, at about **10 mm** of rainfall the prediction is `4780 + 2.89 * 10 = 4808.9` mm (roughly **4.81 m**), which matches the lesson's "around 4.81 meters" reading.

## Plot the model line

Tables are fine, but plotting is clearer. Draw a **line chart** in seaborn (a figure-level relational plot with `kind='line'`), x = rainfall, y = predicted depth, with a set height:

```python
sns.relplot(data=vis_df, x='rainfall', y='predicted_river_depth', kind='line', height=4)
plt.show()
```

The result is a **straight line** rising steadily, which is exactly what "linear" regression produces. (What "linear" really means is covered in the linearity lesson.)

## Overlay observed vs predicted on one set of axes

Now pretend someone went out and **measured actual river depths** at each rainfall total. Real observations will **not** follow the model exactly, because of all the other factors. Store them, then bring the observed column into `vis_df`:

```python
# observed depths measured in the field (real world, not perfectly on the line)
observed_df = pd.DataFrame({'rainfall': x, 'river_depth': observed_values})
vis_df['river_depth'] = observed_df['river_depth']   # copy the observed column over
```

Plot **both** on the **same axes** using seaborn's **axes-level** functions. First make the figure and axes with matplotlib, then pass `ax=ax` to each call so they draw on the same plot:

```python
fig, ax = plt.subplots(figsize=(...))                                   # one shared axes
sns.scatterplot(data=vis_df, x='rainfall', y='river_depth', ax=ax)      # observed points
sns.lineplot(data=vis_df, x='rainfall', y='predicted_river_depth', ax=ax)  # model prediction
plt.show()
```

The **scatter** shows the observed depths and the **line** shows the model's predictions. The line behaves like a **best-fit line**, but the points scatter **around** it: there is **lots of variation and error**, because the model is not perfect. Those errors get examined in the normality-of-errors lesson.

> **Figure-level vs axes-level, in one line:** the single model-line plot used a **figure-level** call (`relplot`), but to layer two plots together you switch to **axes-level** calls (`scatterplot`, `lineplot`) and hand them a shared `ax` you created with `plt.subplots`.

---

## Quiz-ready facts

- The example predicts **river depth (mm)** from **rainfall** with pre-supplied coefficients (assumed already estimated from data).
- **Intercept β₀ = 4780** mm (~4.78 m) = predicted depth at **rainfall 0**; **slope β₁ = 2.89** = mm of depth added per one unit of rainfall.
- **`np.linspace(0, 25, 21)`** makes **21** evenly spaced points because linspace **includes the endpoint**; using 20 intervals + 1 point gives round numbers.
- Predictions are computed **vectorized**: `intercept + slope * column` fills a new column (e.g., 10 mm ⇒ `4780 + 2.89*10 = 4808.9` mm ≈ 4.81 m).
- The single model plot uses a **figure-level** seaborn **line** chart and comes out **straight** (linear).
- To overlay **observed** (scatter) and **predicted** (line), use seaborn **axes-level** calls with a **shared `ax`** from `plt.subplots`.
- The observed points **scatter around** the prediction line; that spread is the model's **error** (river depth depends on far more than one day's rain).

---

> **See also:** "Linear Regression Coefficients (Cliff Notes)" for what the intercept and slope mean, and "Linear Regression Assumptions - Normality of Errors (Cliff Notes)" for the error (the gap between these observed and predicted values).

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
