# Linear Regression Coefficients: the intercept and the slope

**Module 11 Cliff Notes** | Source: lecture transcript "Linear Regression Coefficients"

---

## TL;DR

- The two coefficients in single-input linear regression (**ŷ = β₀ + β₁x**) each have a name: **β₀ is the intercept** and **β₁ is the slope** (the one multiplied by the input x).
- **Slope β₁** controls the **trend**. A slope of **0** means **no trend** (flat prediction). A **larger** slope gives a **steeper** rise, so at the same x a higher slope yields a larger ŷ. **Negative** slopes make ŷ **fall** as x rises.
- **General interpretation of a slope:** the change in the outcome for a **one-unit increase in the input**, holding all other inputs constant. Slope 1 ⇒ +1 in ŷ per +1 in x; slope 2 ⇒ +2 in ŷ per +1 in x.
- **Intercept β₀** is the **predicted outcome when the input is 0** (it is not multiplied by any input). It is the river depth at zero rainfall, or the base salary at zero years of experience.
- Visually, **β₀ sets where the line crosses x = 0** and **β₁ sets its tilt**. You can combine any intercept with any slope.

> **One-line reading of the coefficients:** the **intercept** is the starting value at x = 0, and the **slope** is how much the prediction moves for each one-unit step in x (holding everything else fixed).

---

## The two coefficients, named

Starting again from single-input linear regression:

```text
ŷ = β₀ + β₁x
```

Both β₀ and β₁ are **estimated from data** (that estimation is covered elsewhere). If you already knew their values and an input x, you could plug in and get a prediction. But what do they **mean**? This matters especially for **describing the relationship** between x and y. They have names:

- **β₀ = the intercept** (not multiplied by x).
- **β₁ = the slope** (multiplied by x).

## Reading the slope β₁

Picture how the predicted output changes as the input changes, for different slopes (here the intercept β₀ is held at 0):

- **Slope 0:** a flat black line. **No trend** between output and input. As x grows from 0 to 2 or beyond, the prediction **stays at 0**. The input does not matter.
- **Positive slopes get steeper:** at slope **0.5** the predicted outcome rises gently; at slope **2** the line is the **most vertical**, climbing sharply. So **at the same input x, a higher slope gives a larger ŷ**.
- **Slope 1:** a one-unit increase in x gives a **one-unit** increase in ŷ. **Slope 2:** a one-unit increase in x gives a **two-unit** increase in ŷ.
- **Negative slopes:** with slopes like -0.5, -1, -1.5, -2, the predicted outcome **decreases** as the input increases, and it drops **more sharply** the more negative the slope.

**General interpretation (memorize this one):**

> A coefficient is **how much the outcome changes for a one-unit change in the input, assuming all other input values are held constant.**

## Reading the intercept β₀

The intercept is **not multiplied by any input**, so:

> The **intercept** is the **predicted outcome when the input is 0.**

Concretely:

- **Rainfall example:** the intercept is the **depth of the river when there is no rainfall at all**.
- **Salary example:** predicting salary from years of experience, the intercept is the **base pay at zero years of experience** (the starting salary).

**Visually** (shown with slope 0, giving a flat line): an intercept of **0** puts the line at 0 when x = 0; a **higher** intercept lifts the whole flat line up, so at x = 0 the prediction is higher. The line stays flat because with slope 0 the input is multiplied by zero.

## Combining intercept and slope

You can pair any intercept with any slope:

- **Both positive** (for example slope 1, intercept 1): the line **crosses x = 0 at 1** (its intercept) and keeps **climbing** as x grows (its slope).
- **Both negative:** the line crosses the x = 0 line at a **negative** value and continues getting **more and more negative** as x increases.

So the **intercept anchors where the line sits at x = 0**, and the **slope tilts it** up or down from there.

---

## Quiz-ready facts

- **β₀ = intercept** (not multiplied by x); **β₁ = slope** (multiplied by x). Both are **estimated from data**.
- **Slope = 0** ⇒ **no trend**, flat prediction. **Larger** slope ⇒ **steeper** line ⇒ larger ŷ at the same x. **Negative** slope ⇒ ŷ **decreases** as x increases.
- **Slope 1** ⇒ +1 in ŷ per +1 in x; **slope 2** ⇒ +2 in ŷ per +1 in x.
- **General coefficient interpretation:** the change in the outcome per **one-unit change in the input**, with all other inputs **held constant**.
- **Intercept = predicted outcome when the input is 0** (river depth at no rain; base salary at zero years' experience).
- Visually, the **intercept** sets where the line crosses **x = 0** and the **slope** sets its **tilt**; any intercept can combine with any slope.

---

> **See also:** "Linear Regression Formula (Cliff Notes)" (where β₀ and β₁ come from in the equation), and "Linear Regression Example (Cliff Notes)" which plugs specific coefficient values into code.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
