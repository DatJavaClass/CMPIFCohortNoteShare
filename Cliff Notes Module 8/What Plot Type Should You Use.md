# What Plot Type Should You Use

## The core answer: "it depends"
Plot choice depends on:
- **Purpose** — what you're trying to do with the data.
- **The data itself** (the focus of this class), specifically:
  1. The **data type** of the variables.
  2. The **number of unique values** per variable.
  3. Whether you're plotting **one variable** vs. **relationships between multiple variables**.

These come straight from essential EDA.

## Setup
```python
import pandas as pd
import seaborn as sns               # customary alias
```
- **Gapminder dataset** used for illustration (country info across years; from the *Pandas for Everyone* book on GitHub, tab-separated).
```python
df = pd.read_csv(url, sep='\t')
```
- ~1,700 rows, 6 columns, no missing values. Countries A–Z (Afghanistan to Zimbabwe), continents, years ~1957–2007, population/life expectancy/GDP.

## Factor 1: Variable data type
Two main types for this class:

| Type | Description | Pandas dtype |
|------|-------------|--------------|
| **Categorical** | Non-numeric, usually strings | `object` |
| **Continuous (numeric)** | Integers and floats; specific values, decimals, percentages | `int`, `float` |

- Quiz fact: a **string** column shows up as `object` in pandas.
- Check dtypes directly with `.dtypes` (faster than scanning `.info()`).

## Factor 2: Number of unique values
- **Continuous variables with few unique values can be treated as categorical.**
- Rule of thumb: if a numeric variable has **fewer than ~20 unique values**, consider treating each value as a category.
- Example: `year` has 12 unique values -> reasonable to treat as a category.
- Judgment call ("go with your gut"). A year column collected in just 2021/2022 is clearly categorical (no half-years); a wide continuous range leans numeric.

## Starter plots by data type

### Categorical -> bar chart of value counts
Plot the number of rows per unique value.
```python
df['continent'].value_counts().plot(kind='bar')
```
- Instantly shows most/least frequent categories (e.g., Africa has the most rows, Oceania the fewest).

### Continuous -> histogram
The corollary to value counts: shows where data is concentrated.
```python
df['lifeExp'].hist()
```
- Splits the data into **intervals (bins)**; each bar = count of rows in that range.
- Reveals distribution shape: e.g., most life-expectancy values cluster ~71–76, with a secondary peak ~41–46.

## Factor 3: Individual variables vs. relationships
- So far everything plots **one variable's distribution** = **marginal distributions / marginal plots** (ignore all other variables, focus on one).
- **Relationship plots** (covered later) show how **multiple variables vary together** at a glance.
- Whether you're showing one variable or a relationship strongly drives the plot type you pick.

## Quick decision summary
1. Categorical or continuous? (`.dtypes`)
2. How many unique values? (few numeric -> treat as categorical)
3. One variable or a relationship?
   - Categorical, one var -> **bar chart** (value counts)
   - Continuous, one var -> **histogram**
   - Multiple vars -> **relationship plots** (later)

---

*Authored and directed by **Victor Sverdlin (DatJavaClass)**: conceived, structured, formatted, fact-checked, and edited by Victor Sverdlin, with assistance by Claude.*
