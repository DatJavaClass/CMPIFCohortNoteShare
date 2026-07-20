# Introduction to Predictive Analytics: learning relationships to predict an outcome

**Module 11 Cliff Notes** | Source: lecture transcript "Introduction to Predictive Analytics"

---

## TL;DR

- **Predictive analytics** is analyzing data to **learn a relationship** between an **outcome** and one or more **inputs**, so you can both describe that relationship and predict the outcome for cases you have not yet seen.
- The **outcome** (also called output, response, target, or dependent variable) is denoted **Y**. The **inputs** (also called independent variables) are denoted **X** (capital because there can be many of them, like a matrix).
- The goal is to learn a **function F** that maps the inputs X to the outcome Y. That function is **learned from patterns in the data**, not specified by hand. This is also called **predictive modeling** and **supervised learning**.
- Variables are just **columns in a data table**. Getting messy data into an analytics-ready format (merging, cleaning, pandas work) is a big part of the real job.
- **When it helps:** meaningful phenomena that are **not completely straightforward** (deterministic) and **not completely random**. A grocery bill is deterministic (use a calculator); a player's jersey number from their name is random (no logical link).
- **Big warning:** the relationships predictive analytics finds are **correlations, not causation**. There must be a **logical reason** for a relationship, or you risk a spurious one (like PhDs awarded vs mozzarella consumption, both just rising over time).

> **The one-line definition:** predictive analytics learns a function from input variables (X) to an outcome variable (Y) using patterns in data, so you can describe how they relate and predict Y for new, unobserved cases.

---

## What predictive analytics is

Predictive analytics is the **analysis of data with the goal of learning a relationship** between a couple of things:

- The **outcome** we care about, denoted **Y**. It has many names depending on the field: output, **response variable**, **target variable**, or (in more experimental terms) **dependent variable**.
- The **inputs** doing the predicting, denoted **X**, also called **independent variables**. X is written capital because it can hold **one or many** variables at once (think of it like a matrix, though you do not need to worry about that here).

We learn a **function F** that goes from the inputs X to the outcome Y:

```text
Y = F(X)
```

That function F is **learned from data**, from the patterns in it, so we can estimate the relationship between the outcome and the inputs that predict it. Because the computer is learning that association F, this is also called **predictive modeling** and **supervised learning**.

> **Heads up, math is coming.** This module (and the ones after) uses more equations than the earlier course. The lecturer's advice: break each equation into parts, understand each part, then put it back together. If a Greek letter is unfamiliar, paste it into a search engine to get its name.

## Variables are columns, and getting there is work

These X and Y variables are literally **columns in a data table**. It often takes real work to get data into a modeling-ready format, which is exactly why so much of the earlier course was about **merging datasets, cleaning them up, and wrangling in pandas**. Those unglamorous skills are what put data into an analytics-ready shape.

**The ancient-buildings example** (used earlier in the course too): a table of historic buildings with a `year built`, the `material` recorded, and `parts per trillion of carbon 14` (useful for dating the material). As the analyst, **you choose**:

- The **Y** (output to predict): usually the variable you actually care about. Here, the **year built** (a paleontologist unearths something and wants to know how old it is).
- The **X** (inputs): often everything else in the table. Here, **carbon 14**, the **material**, the **location**, and other factors whose relationship with the outcome you want to learn.

## Why do it? Describe, then predict

Going back to Module 1's working definition, **data science** is collecting, processing, and manipulating data to **derive insight** and **make predictions**. The two goals map directly onto predictive analytics:

1. **Describe** relationships between an outcome you care about and the input variables. (For a house: how much does a fresh coat of paint really change the selling price? You can express that mathematically from data.)
2. **Predict** the outcome value for cases you have **not yet observed**. (Literally predict the sale price of a home about to go on the market from its location, square footage, number of rooms, and so on.)

## When to use it, and when not to

Predictive analytics is for **meaningful phenomena** with real variation. It sits between two extremes it is **not** good for:

- **Too straightforward (deterministic).** Your grocery bill is just the quantity of each item times its price. There is no interesting variation to learn, so just use a calculator. But predicting **how much of each product will sell** by day of the week, season, or shelf placement **is** a job for predictive analytics, because people's behavior is not deterministic (yet not random either), and those patterns can be learned.
- **Too random.** Predicting a professional athlete's jersey number from their name is hopeless; there is no logical link, so even a mountain of data would not help, and there would be no useful insight even if a pattern appeared. By contrast, predicting **which team wins** from player and team stats is both **possible** (winning depends on measurable things) and **meaningful** (people care), so it is a good application, even though luck and chance mean you can never predict it perfectly.

There is always some degree of randomness, luck, and chance. The point is that the problem should not be **completely** random.

## The big caveat: correlation, not causation

The relationships predictive analytics finds are like **correlations**. They are **not causation**. There has to be a **logical reason** for the relationship to be meaningful.

The classic spurious example: there is a strong measurable relationship between **PhDs awarded in US civil engineering programs** in a given year and **US per-capita consumption of mozzarella cheese**. That relationship means nothing. Both quantities simply **rose over time** for unrelated reasons, so they track a hidden third factor (time), not each other. Use predictive analytics for phenomena that are meaningful, not completely straightforward, not completely random, and that have a **logical reason** to be related rather than being driven by something you are not seeing.

---

## Quiz-ready facts

- **Predictive analytics** = analysis of data to **learn a relationship** between an outcome and its inputs, in order to **describe** the relationship and **predict** the outcome for **unobserved** cases.
- **Outcome / output / response / target / dependent variable = Y.** **Input / independent variable = X** (capital X because it can be many variables).
- The learned mapping is a **function F** from X to Y, learned **from patterns in data**.
- Predictive analytics is also called **predictive modeling** and **supervised learning**.
- Variables are **columns** in a data table; a lot of real work is getting data into an **analytics-ready** format (merging, cleaning, pandas).
- Two data-science goals it serves: **describe** relationships between attributes, and **predict** attribute values for cases not yet observed.
- Good targets are **not deterministic** (a grocery bill is; use a calculator) and **not completely random** (a jersey number from a name).
- The relationships found are **correlations, not causation**; a meaningful relationship needs a **logical reason** (the PhDs-vs-mozzarella correlation is spurious, both just rise over time).

---

> **See also:** the next lesson "Predictive Models (Cliff Notes)" (what a model is and how it is fit), then "Linear Regression Formula (Cliff Notes)" for the specific model this course teaches.

---

*Source: University of Pittsburgh lecture transcript (personal study use).*
