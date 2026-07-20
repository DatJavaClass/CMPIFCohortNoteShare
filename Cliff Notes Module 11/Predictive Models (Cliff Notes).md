# Predictive Models: simplified lenses fit to data, and the two kinds

**Module 11 Cliff Notes** | Source: lecture transcript "Predictive Models"

---

## TL;DR

- A **model** (predictive model or statistical model) is a **simplified little world** that isolates a few variables and relates them through a **mathematical equation**.
- You **fit** a model (find its **parameters**) using a dataset of observations, so it captures the pattern between the inputs X and the outcome Y.
- Models are **lenses**: they bring some relationships into focus and blur out the rest of the real world's complexity. The goal is **partial** prediction, not perfect prediction, so there will always be **error**.
- The **process**: (1) the analyst picks the **type** of model (a family of functions; this course uses **linear regression**), and (2) the model's **parameters** (called **coefficients** in linear regression) are learned **automatically from data** by an algorithm that **minimizes prediction error**.
- Predictive modeling is **supervised learning**, and it comes in two kinds by outcome type: **regression** predicts a **continuous** outcome, **classification** predicts a **categorical** outcome. This course focuses on **regression**.

> **Why models have error:** they are necessarily simplifications of complex real-world systems. To perfectly predict grocery sales you would need every shopper's meal plans and aisle choices, which is impossible. A good model captures the useful pattern anyway.

---

## What a model is

To do predictive analytics (learn a relationship between inputs and an outcome, then describe or predict), you build a **model**: a **predictive model** or **statistical model** that can be written as a **mathematical equation**.

Models are **simplified little worlds** where we isolate a few variables and numerically relate them. A **statistical** model relates them through statistics and mathematics. You **fit** the model (find its **parameters**) using **statistical and computational methods** on a dataset of observations, finding the patterns in that data and expressing the relationship in the model. Then you can ask **how good** the model is by testing its **fit** to all the data points you have.

It will **not** be perfect. To fully predict which products sell in a grocery store, you would have to know the meal preferences of everyone walking in, why they picked this store, and which aisle they walk down first. Real-world complex systems cannot be predicted perfectly.

## Models as lenses

Think of models as **lenses**. They necessarily bring **some** relationships into focus and **blur out** the rest of the complexity. You can choose from **different types** of models (different lenses) that fit what you are looking at better. The goal is **not perfect prediction, but partial prediction**: how much of the variation we see (how much cheese the store sells) depends on things we can measure (shelf placement, day of week, season).

## The math, and why error is inevitable

Predictive models are written mathematically, with **X** for inputs and **Y** for the outcome. The goal is to learn a **function** you apply to X to get an estimated value for Y:

```text
Y ≈ F(X)
```

X can be a **list of inputs** (a **row** of attribute values for one data point). For the ancient-buildings example, a row might be `0.84 parts per trillion of carbon 14, material = marble`, and the model outputs a predicted **year**. Ideally the model does well not just for one row but for **all** of them.

Error is inevitable because **models are simplifications**. The lecturer's image: a **diorama of New York City** does not capture every brick on every building, of course not, that is impossible. It gets some things wrong and has some error, but it can **still be useful** by showing the patterns we can find.

## The process of building a predictive model

Two steps:

1. **Choose the type of model.** You, the analyst, pick the **type of function** from **families of functions** that can be learned between inputs and outputs. Your **exploratory data analysis (EDA)** can help inform the choice. In this course you do not have to agonize over this: we use **linear regression** as the one type of model.
2. **Learn the parameters.** Within each model type are **parameters** (in linear regression these are called **coefficients**) that define exactly how the inputs relate to the output. These are **useful for the descriptive purpose** of modeling, and they are learned **automatically** by a **learning algorithm** that estimates them from data by trying to **minimize the prediction error** (getting the most predictive accuracy possible). This course does not go into the details of that optimization, but you should know it is happening.

## Two kinds: regression vs classification

Predictive modeling is also called **supervised learning** (the "learning" is the machine estimating parameters from the relationships in data, which is where **machine learning** gets its name). Just as the **data type** of a variable drives which plot you use, the **data type of the outcome variable** decides the kind of predictive model:

| Kind | Predicts | Outcome data type | Example |
|---|---|---|---|
| **Regression** | a **continuous / numerical** outcome | numeric | selling price, river depth |
| **Classification** | a **categorical** outcome | category | dog / cat / bird, yes / no |

This course **focuses on regression**.

---

## Quiz-ready facts

- A **model** (predictive / statistical model) is a **simplified** representation that relates a few variables through a **mathematical equation**.
- **Fitting** a model = finding its **parameters** from a dataset of observations using statistical/computational methods.
- Models are **lenses**: focus some relationships, blur the rest. Goal is **partial**, not perfect, prediction, so **error is always present**.
- Error is inevitable because models **simplify** complex reality (the NYC diorama analogy).
- **Process:** (1) analyst **chooses the model type** (a family of functions; EDA can inform it; this course = **linear regression**), then (2) **parameters/coefficients** are **learned automatically from data** by an algorithm that **minimizes prediction error**.
- Predictive modeling = **supervised learning** (the root of **machine learning**).
- **Regression** predicts a **continuous** outcome; **classification** predicts a **categorical** outcome. The **outcome's data type** decides which. This course does **regression**.

---

> **See also:** the previous lesson "Introduction to Predictive Analytics (Cliff Notes)" (X, Y, and the function F), and the next lesson "Linear Regression Formula (Cliff Notes)" which makes the general model equation concrete.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
