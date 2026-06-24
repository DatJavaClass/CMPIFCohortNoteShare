# Module 8 Cliff Notes: Plotting Combinations of Categorical Variables

## Key Concepts

- Use different plot types based on variable types.
  - Categorical vs categorical: grouped count plots, heatmaps, facets
  - Continuous vs continuous: scatter plots, line plots
  - Categorical vs continuous: boxplots, violin plots, point plots

- For categorical relationships, count-based visualizations are ideal.
  - Count plots show the number of rows in each category.
  - Heatmaps encode counts as color intensity in a matrix of category combinations.

## Setup

- Import `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn`.
- Load sample data: `penguins = sns.load_dataset("penguins")`

## Identifying Categorical Variables

- Use `penguins.info()` or `penguins.dtypes`.
- In this dataset, `species`, `island`, and `sex` are categorical/object columns.

## Grouped Bar Charts / Count Plots

- Use `sns.catplot(..., kind="count")` or `sns.countplot(...)`.
- Example: group counts by `species` and color by `island`.
- Good for two categorical variables with relatively few unique values.

## Facets (Subplots)

- Use `col` or `row` in figure-level Seaborn functions (`catplot`, `relplot`, `displot`).
- Faceting lets you compare the same plot across subsets of a third categorical variable.
- Example: `sns.catplot(data=penguins, x="species", hue="island", col="sex", kind="count")`

## Heatmaps for Categorical Relationships

- Use `pd.crosstab()` to build a count matrix for two categorical variables.
- Example: `ct = pd.crosstab(penguins["species"], penguins["island"])`
- Plot with `sns.heatmap(ct)`.
- Add `annot=True` and `fmt="d"` to show integer counts inside cells.

## Practice Exercises Covered

1. Create a grouped count plot with `sex` on the x-axis and `species` as the hue.
2. Create an annotated heatmap of counts for `sex` vs `island`.

## Notes

- `sns.catplot` handles both grouping and faceting in one call.
- When a categorical variable has many levels, heatmaps often scale better than grouped bar charts.
- Use `fill_value=0` after `unstack()` when constructing count tables to avoid missing combinations.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
