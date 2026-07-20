# Linear Regression Assumptions - Normality of Errors

**Module 11 Cliff Notes** | Source: lecture transcript "Linear Regression Assumptions: Normality of Errors"

---

## TL;DR

- The **error** term ε in linear regression is the **difference between the predicted output and the observed output**. These errors are also called **residuals**.
- **The normality-of-errors assumption:** the errors are **normally (Gaussian) distributed** around a **mean of 0** with a **standard deviation σ**, written **ε ~ Normal(0, σ)** (the tilde `~` reads "is distributed as").
- **σ is a single estimated number** for the whole dataset. It does **not** change with x (a preview of the constant-variance assumption).
- **Mean 0** means observations sit **around** the prediction (an error of 0 is a perfect hit), and most data lands **within about one σ** of the regression line, with few points 2 to 3 σ away.
- **More observations at a given x** ⇒ **more confidence** in the predicted average, via the **standard error of the mean = σ / √n**. Under the **Central Limit Theorem**, the estimated **coefficient/slope** is itself Gaussian, so you can put a **confidence interval** on it and say (descriptively) whether the relationship is positive or negative.

> **Why errors exist at all:** you predict only from a few input variables, so random real-world factors are left out (Jim wandered down aisle four for the Nutella and forgot the cheese). Linear regression does not model those; it assumes they wash out as normal, zero-mean noise.

---

## What the errors are

The final term in the linear regression formula is **ε (epsilon)**, the **errors**. An error is the **difference between the predicted value of the output Y and the observed output Y**. Given a dataset you can compute these directly. They are also known as **residuals**.

Models are not perfect predictors, so errors are inevitable: you predict only from a **limited set of input variables** and cannot capture every random factor (the grocery-store shopper who detoured for Nutella and forgot the cheese). Linear regression makes a specific **assumption** about what these leftover random errors look like.

## The assumption: ε ~ Normal(0, σ)

The random errors are assumed to be **normally distributed** (a **normal / Gaussian distribution**) with a **mean of 0** and a **standard deviation of σ (sigma)**:

```text
ε ~ Normal(0, σ)
```

- The **tilde `~`** reads **"is distributed as."** So the line says the probability distribution of ε looks like a Gaussian with mean 0 and standard deviation σ.
- **σ is its own estimated number**, one value **across the whole dataset**. It **does not vary with x** (which anticipates the separate constant-variance / homoscedasticity assumption).

### The mean of 0

With a Gaussian, most data concentrates **around the mean**. A mean error of **0** means the observed values sit **right around the prediction**: an error of exactly 0 means the observation **equals** the prediction (which is what you hoped to predict). So the true observations are **not drawn completely at random**; they cluster around the prediction, with that Gaussian shape, at every value of x.

### The standard deviation σ

Take an example with **σ = 0.5** and a regression line drawn with **ribbons at ±1σ, ±2σ, and ±3σ**:

- **Most observed points fall within one σ** of the predicted output.
- **Few** points should land near **2, 3, or more σ** away.
- The lecturer's mental image: little **Gaussian distributions on their sides**, standing up and down all along the regression line, one at each x.

## More data means more confidence

The **more observations you have at each input value x**, the more sure you can be of the **average predicted output** there. One way to see it: the **standard error of the mean** is

```text
standard error = σ / √n
```

so **more data (larger n) shrinks the standard error** and tightens the estimate of the average outcome. Keep in mind your data is always just a **sample** of the true underlying **population** distribution, so you can always collect more (more river readings at a given rainfall, more days of cheese sales at a given shelf spot).

Just as with estimating means earlier in the course, the **Central Limit Theorem** says that if you keep drawing samples, your **estimate of a coefficient (the slope)** follows a **Gaussian** distribution. You can therefore look at its **standard deviation (its standard error)** and build a **confidence interval** on the coefficient. With that interval you can say things like "I'm pretty sure there is a **positive** (or **negative**) relationship" between an input and the outcome. That is the **descriptive insight** predictive analytics gives you, and it gets demonstrated in code later.

---

## Quiz-ready facts

- **Error / residual** = **predicted output minus observed output** (the ε term); computable from a dataset.
- **Normality-of-errors assumption:** **ε ~ Normal(0, σ)**, errors are **Gaussian** with **mean 0** and **standard deviation σ**; the tilde `~` means **"is distributed as."**
- **σ is a single estimated number** for the whole dataset and **does not vary with x**.
- **Mean 0** ⇒ observations cluster **around** the prediction (error 0 = exact hit); **most data is within ~1σ** of the line, few points beyond 2 to 3σ.
- **Standard error of the mean = σ / √n**, so **more observations at an x ⇒ more confidence** in the predicted average.
- Data is always a **sample** of the **population**; under the **Central Limit Theorem** the estimated **coefficient/slope** is **Gaussian**, giving a **confidence interval** on it.
- Those confidence intervals let you state a **positive or negative** relationship, the **descriptive** payoff of predictive analytics.

---

> **See also:** "Linear Regression Formula (Cliff Notes)" (where the ε term comes from) and the notebook note "Linear Regression Assumptions - Linearity - Notebook (Cliff Notes)", which lays out all four assumptions of linear regression together.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
