# Interaction Features for Linear Regression: multiplied effects, main effects, and the three ways to write an interaction

**Module 13 Cliff Notes** | Source: recorded lab session, transcript "Lab Recording 2026-07-22 1308" (the spoken walkthrough of the interaction features lab)

---

## TL;DR

- An additive model just **adds** each feature's effect (coefficient times value, summed). An **interaction feature** instead **multiplies the variables together**, so the model can capture "when this feature has a value AND that feature has a value, something extra happens to the output."
- The core sentence of the session: **how one variable relates to the output can depend on the value of another variable.** When that is true, addition alone is not enough.
- Two warm-up intuitions: cold is uncomfortable and rain is uncomfortable, but cold rain together is "doubly terrible" (more than the sum); and the amount of respiratory virus someone is exposed to predicts sickness, but that relationship is much sharper if the person is immunocompromised.
- The lab predicts apartment **price** from **size in square meters** (continuous) and **city center or not** (Boolean), on a 150-row dataset pulled from a URL. The instructor notes in passing that this is **synthetic data**.
- You usually do NOT want the interaction term alone. You usually keep the **main effects** (each variable's stand-alone, additive contribution) AND add the interaction on top.
- Three formula spellings, **identical fitted parameters** all three times:
  - **`a + b + a:b`**: the colon is the interaction term written explicitly, listed after both main effects.
  - **`a * b`**: the everyday shortcut, keeps both main effects and adds their interaction.
  - **`(a + b)**2`**: same model again; the `**2` caps interactions at **two-way (pairwise)**, which is what makes it useful once you have three or more variables.
- Everything in the fitted model is **significant under 0.05**, including the interaction: each additional square meter adds **more** to the price in the city center than outside it.

> **One-line takeaway:** an interaction feature multiplies two inputs so one input's effect can depend on the other; write it as `a:b` next to the main effects, or just `a * b`, and cap the order with `(...)**n` when the variable count grows.

---

## The idea: adding effects vs multiplying them

Everything before this point in the module treated features **additively**: each feature has a coefficient, you take coefficient times value for this feature, add it to coefficient times value for that feature, and so on. The session opens by pointing out that regression models can also encode **interactions between features**: if this particular feature has a value and the other particular feature has a value, you get a unique extra effect on the output.

Instead of adding the variables' influence, **interaction features multiply the effect of each variable**. This happens a lot in the real world, which is why the session spends time on intuition before any code.

## Two warm-up examples

**Weather comfort.** Imagine collecting data on how comfortable people feel across weather factors, say rain and temperature. People are less comfortable when it is cold, and less comfortable when it is raining, and maybe those are just additive. But think about being outside at 40 degrees Fahrenheit in the rain (the instructor's hiking-and-biking example): wet AND cold at the same time is "kind of doubly terrible." That extra misery beyond the sum of the two effects is exactly what an interaction term captures: when raining is yes and temperature is cold, the output effect is even stronger.

**Respiratory virus exposure.** In a healthcare setting, the amount of virus someone is exposed to (a continuous variable) matters a lot for predicting the probability of getting sick. But if the person is **immunocompromised**, that relationship is sharper: they may get sick from a much smaller exposure. The slope of exposure against sickness probability **depends on** immune status. (The instructor flags he is not an epidemiologist and is riffing, but the modeling point stands.)

Both examples land on the same sentence: **how one variable relates to the output you are predicting can depend on another variable.** That dependency is an interaction.

## The lab dataset: apartment prices

The running example is real estate: predict an apartment's **price** from its **size** (continuous, in square meters) and whether it is **in the city center** (a Boolean). This is a good candidate for an interaction: bigger apartments cost more, sure, but an additional square meter in the city center plausibly adds **more** to the price than an additional square meter outside it. The relationship between size and price depends on location.

```python
import pandas as pd
import statsmodels.formula.api as smf

data_url = 'https://raw.githubusercontent.com/michaelmilleryoder/cmpinf2100/refs/heads/main/apt_real_estate.csv'
data = pd.read_csv(data_url)
```

The data is pulled in from a URL: **150 rows** (Verified), with `size_m2` (continuous size in square meters), `city_center` (Boolean), and `price_usd` (the target). A quick look at the head shows a mix of city-center and non-city-center rows across a range of sizes and prices. The instructor mentions offhand that **this is synthetic data**, which is worth knowing: the effects are clean because they were built in.

## Way 1: spell it out with a colon (`a:b`), and keep the main effects

The session's first admission: there are **many** ways of specifying interactions in the statsmodels formula interface ("there might even be more" than the three shown), partly because with many variables there are many possible interactions. With two variables it stays manageable.

An important point before the first fit: **you usually don't want just the interaction feature; you usually want the main effects too.** The main effects are simply what you learn in an additive relationship: how much variation the size of the apartment explains on its own, versus how much city center explains on its own. The interaction is added **on top of** those, so the explicit formula lists all three terms:

```python
model = smf.ols(formula='price_usd ~ size_m2 + city_center + size_m2:city_center', data=data).fit()
model.params
```

The **colon** (`size_m2:city_center`) is the interaction between the two variables, written explicitly. Verified parameters:

```text
Intercept                      31586.41
city_center[T.True]            92869.53
size_m2                         2996.99
size_m2:city_center[T.True]      870.57
```

## Reading the parameters

Taking the fitted parameters in the order the session walks them:

- **Intercept**: present as always.
- **`city_center[T.True]` (a main effect)**: `city_center` is categorical, and with Booleans it is really nice to interpret, because the **reference category is `False`**. So the `True` coefficient says being in the city center raises the price by a whole bunch (about $92.9k here).
- **`size_m2` (the other main effect)**: a positive relationship between size and price, which makes a whole lot of sense (about $2,997 per square meter).
- **`size_m2:city_center[T.True]` (the interaction)**: for every square meter you add, **if you are in the city center**, you get an additional price bump on top of the base slope (about $871 more per square meter). Positive coefficient, so what we suspected going in is what the data shows.

> **Reference-category refresher:** the `[T.True]` naming is the same Treatment (dummy) coding from the categorical inputs lecture. `False` is the dropped reference level, so both `city_center` terms are read **relative to a non-city-center apartment**. See the categorical cliff notes for the full story.

## Way 2: the star shortcut (`a * b`)

Because you so often want the main effects AND the interaction together, there is a shortcut people commonly use, `*`:

```python
model = smf.ols(formula='price_usd ~ size_m2 * city_center', data=data).fit()
model.params
```

This **keeps both main effects and adds the interaction term**. And, look, the parameters are **exactly the same** as Way 1 (Verified: identical output). `a * b` is just `a + b + a:b` spelled compactly.

## Way 3: parentheses and a power, `(a + b)**2`

The third way is where it gets, per the instructor, even more complex: put the variables in parentheses and raise to a power.

```python
model = smf.ols(formula='price_usd ~ (size_m2 + city_center)**2', data=data).fit()
model.params
```

Exact same model, exact same coefficients, exact same extracted features (Verified). In Python, `**2` reads like "to the power of two," which the instructor finds confusing and admits he does not really like. The reason the syntax looks alien is that **the formula interface is drawn from the R programming language**, so its operators follow R's formula conventions rather than Python's arithmetic.

So what is the point of `**2`? **Capping the interaction order.** With only two variables it changes nothing, but with three or more variables:

- Chaining stars (`a * b * c * ...`) gives **all possible interactions** among the variables, including three-way, four-way if you had four variables, and so on.
- `(a + b + c)**2` means "**only up to two-way (pairwise) interactions**": all main effects, all pairs, no `a:b:c`.
- Three-way and higher interactions get tricky to interpret, so often you want just the pairwise ones. You can cap at up to two or up to three with `**2` or `**3`.

## Significance check

Last stop of the session: the p-values.

```python
model.pvalues
```

**Everything is significant under 0.05** (Verified: all four terms, and `model.pvalues < 0.05` returns all `True`; the interaction's p-value is about 0.0064). Significance here means the coefficients can be interpreted with respect to this dataset, and the interesting one is again the **interaction effect**: each additional square meter adds more to the price when the apartment is in the city center than when it is not.

---

## Quiz-ready facts

- **Additive models add** each feature's contribution (coefficient times value, summed). **Interaction features multiply the variables**, capturing effects that appear only when both conditions hold together.
- **The key sentence:** how one variable relates to the output **can depend on another variable**. That dependency is what an interaction term models.
- **Session examples:** cold + rain being "doubly terrible" for comfort (beyond the additive sum); viral exposure predicting sickness with a **sharper slope** for immunocompromised people; apartment size adding **more** price per square meter in the city center.
- **Main effects** = each variable's stand-alone additive contribution. **You usually keep the main effects and add the interaction**, not the interaction alone.
- **Three equivalent spellings** (identical parameters, Verified): `a + b + a:b` (colon = explicit interaction term), `a * b` (shortcut: main effects plus interaction), `(a + b)**2` (interactions up to degree 2).
- **`**n` caps the interaction order**: with 3+ variables, `(a + b + c)**2` gives main effects plus pairwise interactions only, while chained `*` gives every order (3-way, 4-way, ...). Higher-order interactions are hard to interpret, which is why you cap.
- **The formula mini-language comes from R**, which is why `*`, `:`, and `**` do not mean what they mean in ordinary Python.
- **Boolean categorical inputs**: reference category is `False`, terms print as `[T.True]`, and coefficients are read relative to the `False` baseline.
- **Verified fit** (`price_usd ~ size_m2 * city_center`, 150 synthetic rows): Intercept 31586.41, city_center[T.True] 92869.53, size_m2 2996.99, interaction 870.57; **all p-values < 0.05** (interaction about 0.0064).
- **The apartment data is synthetic** (instructor's aside), which is why the built-in interaction shows up so cleanly.

---

> **See also:** "Interaction Features for Linear Regression - Notebook (Cliff Notes)" (the companion lab notebook these remarks narrate, with full verified outputs, dtypes, and the two-line slope algebra), "Prediction Visualizations for Interaction Features - Notebook (Cliff Notes)" (drawing the fanning non-parallel lines this model implies), "Additive Features in Statsmodels - Notebook (Cliff Notes)" (the plain `a + b` baseline and its parallel lines), "Categorical Inputs to Linear Regression (Cliff Notes)" (why `city_center[T.True]` is read against a `False` reference), and "Nonlinear Feature Transformations in Statsmodels - Notebook (Cliff Notes)" (the other place the R-style `**` operator bites, where `x**2` in a formula does NOT square anything).

---

*Source: University of Pittsburgh lab session transcript (personal study use).*
