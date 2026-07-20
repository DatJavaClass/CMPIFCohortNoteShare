# Linear Regression Formula: a glorified y = mx + b

**Module 11 Cliff Notes** | Source: lecture transcript "Linear Regression Formula"

---

## TL;DR

- **Linear regression** is the specific predictive model this course teaches. Because it is **regression**, the output **Y is continuous**.
- It is basically a **glorified `y = mx + b`** (the slope-of-a-line equation). For a single input the formula is:

  **ŷ = β₀ + β₁x + ε**

- **ŷ** (Y-hat) is the model's **estimated / predicted** output, not the true output. **β₀** and **β₁** are the **parameters (coefficients)**, the heart of the model, **learned from data**. **ε** (epsilon) is the **error**, always present.
- This is **simple linear regression**: exactly **one input variable** X. You can have more than one input, but not yet in this lesson.
- Per data point **n** (out of **N** input-output pairs / rows), the prediction is **ŷₙ = β₀ + β₁xₙ** (ignoring error): plug in that row's input `xₙ` and out comes a prediction.
- A common **variant** writes the left side as **μ** (mu, "mean"), because linear regression predicts the **average** output Y at a given input X, not any single observation.

> **The lecturer's tactic for scary equations:** break the formula into parts, understand each part on its own, then reassemble it. Unfamiliar Greek letters? Paste them into a search engine to get the name.

---

## From the general model to linear regression

Predictive models in general look like "output Y is some **function** of inputs X, plus some **error**." Linear regression makes that function specific. Quick quiz from the lecturer: because it is **regression**, what is the data type of the output Y? **Continuous.**

What we are really looking at is a **glorified version of `y = mx + b`**, the slope-of-a-line equation from middle- or high-school math. Pull out the graphing calculator.

## The formula, part by part

For linear regression with a **single input variable**:

```text
ŷ = β₀ + β₁x + ε
```

- **ŷ (Y-hat):** the little "hat" marks this as the model's **estimate / prediction** of the output, **not** the real observed output.
- **β₀ and β₁:** the **parameters** of the model, also called **coefficients**. These are the **heart of the model**, how it relates input to output, and they are the numbers **estimated from data** by a learning algorithm (covered elsewhere). Having **two** coefficients here is what makes this the **one-input** (simple) case.
- **β₁x:** the coefficient **β₁ multiplied by the single input X** (one continuous input variable).
- **ε (epsilon):** the **error** term. There is always some error; its structure gets detailed in the normality-of-errors lesson.

> **Why "simple":** this is the formula for linear regression with **just one variable of input**. More inputs are possible (multiple regression), but this lesson stays with one.

## Making it concrete: predicting river depth

Let the output **Y = the depth of a river** and the single input **X = how much it rained that day**. This is deliberately oversimplified (river depth also depends on yesterday's rain, the watershed size, soil, and more), but rainfall has a **logical association** with river depth, which is the point.

Our datasets are **collections of observations**, each with a value for every column. Split the columns into one **Y column to predict** and (here) one **X input column**, giving **N input-output pairs**, where **N is the number of rows**. Then, for each data point n:

```text
ŷₙ = β₀ + β₁xₙ        (ignoring the error term for now)
```

Plug in that row's input `xₙ` (say **11 mm of rain** that day), and out comes a prediction for the river's depth. (Actually estimating β₀ and β₁ from data, and seeing this in code, comes in later lessons.)

## A variant you will see: predicting the mean (μ)

There are several ways to write the linear regression formula. One common variant replaces ŷ with the Greek letter **μ (mu, "mean")**:

```text
μₙ = β₀ + β₁xₙ
```

This spells out something important: linear regression predicts the **average (mean) output Y** at a specific input value X, not any single observation. Even at the **same** value of X there is natural **variability** in the outcome (the same rainfall can leave the river at slightly different depths on different days), so the model targets the **average** output at that X.

---

## Quiz-ready facts

- **Linear regression** is a **regression** model, so its output **Y is continuous**.
- It is a "glorified **`y = mx + b`**."
- Single-input formula: **ŷ = β₀ + β₁x + ε**.
- **ŷ (Y-hat)** = **predicted/estimated** output (the hat means estimate, not the true value).
- **β₀, β₁** = **coefficients / parameters**, the heart of the model, **estimated from data**. **ε (epsilon)** = **error**, always present.
- **Two coefficients** ⇒ this is the **one-input (simple)** linear regression case.
- With **N** rows (input-output pairs), per point n: **ŷₙ = β₀ + β₁xₙ** (error term dropped).
- A variant uses **μ (mu, mean)** on the left, because linear regression predicts the **average** output at a given X, not a single observation (there is still variability at the same X).

---

> **See also:** "Predictive Models (Cliff Notes)" for the general model equation this specializes, and "Linear Regression Coefficients (Cliff Notes)" which explains what **β₀ (intercept)** and **β₁ (slope)** mean and how to read them.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
