<p align="center"><img width="651" height="701" alt="ChatGPT Image Jun 11, 2026, 12_49_40 PM" src="https://github.com/user-attachments/assets/99da5453-5993-4993-8aac-51c0a4b6510c" /></p>

# Cliff Jumper Notes

## What these are

A Cliff Jumper Note takes all the separate Cliff Notes from one module and
weaves them into a single, connected lesson. Instead of six short notes that
repeat the same ideas, you get one walkthrough that introduces each idea once
and builds on it. Where it helps, a small amount of extra explanation or
example is added, but only when it is correct.

There is one note per module (Modules 1 through 13; Module 7 is split into
parts, with Parts 1 and 2 here).

## How they are checked: "Who Watches the Watchmen"

These notes were drafted with AI help, so every one was put through a checking
system to catch mistakes before it went in here. It works in a diamond shape:

1. The note is written.
2. Two separate checkers read it independently and hunt for errors. One checks
   that every claim matches the source Cliff Notes. The other checks the
   technical side and actually runs every piece of Python and every number to
   confirm it is right.
3. A third checker reviews the first two. This is the "who watches the
   watchmen" step. It catches anything the first two missed and throws out any
   complaint they got wrong.

Anything the checkers flagged was fixed, and the fix was checked again. All
code and all numbers in these notes were run, not guessed.

Module 10 went through an expanded version of this: two
diamonds in sequence, one for accuracy and one for clarity, then a full-document
release check over the finished note before it went in.

## Confidence of Accuracy

How sure we are each note is free of factual or code error. All are high; they
differ mostly by how much of the note is runnable code (objectively checkable)
versus concept and advice (checked against the source).

| Module | Confidence of Accuracy | Basis |
|---|---|---|
| 1 | High (~95%) | Facts all traced to source, one code block run, one editorial overreach caught |
| 2 | Very High (~98%) | Smallest note, purely Python, every line and operator (including the `^` XOR bit math) run |
| 3 | High (~93%) | Technical claims run and corrected, quiz-safety hazard caught |
| 4 | High (~95%) | Every number confirmed by simulation, CI coverage tested at 200k trials, adversarial review |
| 5 | Very High (~96%) | About 50 code checks all passed, SEM verified bit for bit, two source errors caught and fixed |
| 6 | Very High (~97%) | Every code block and number run on pandas 3.0.3; two version-drift fixes (string dtype, Copy-on-Write) verified; a missing plot import and one overreaching claim caught |
| 7 (Part 1) | High (~94%) | Passed full Diamond QC (independent source and technical audits plus meta-review) |
| 7 (Part 2) | Very High (~96%) | Every number and code block run on pandas 2.3.3 (groupby/agg/sem, Titanic survival); full Diamond QC found no factual or execution errors; one source slip corrected ("fewest in 1st"→2nd), and the `.sem()` √n basis disambiguated to non-null count |
| 8 | Very High (~96%) | Every seaborn/pandas/matplotlib call run on seaborn 0.13.2 / matplotlib 3.10.6; numeric reads verified against the data (lifeExp bimodality with modes ~71 and ~46, gdpPercap and diamonds-price right-skew, `year` = 12 uniques, the IQR and 1.5×IQR whisker rule, KDE y-axis = density); `str`/`object` dtype note checked on pandas 3.0.3 and 2.3.3; full Diamond QC, where the meta-auditor caught and fixed a whisker-anatomy claim that contradicted the note's own outlier example |
| 9 | Very High (~96%) | Every seaborn/pandas call run on seaborn 0.13.2 / pandas 2.3.3 (relplot/displot/catplot defaults, point-plot default bootstrap CI and `common_norm=True`, `year` categorical-by-default, dropna 344→342); point-plot statistics (95% CI about 2×SEM via the Central Limit Theorem, 68% about 1×SEM) verified against source; full Diamond QC where the meta-auditor caught the shared blind spot both auditors missed, that the violin `inner="box"` median is a short white tick (not a dot) and the `inner="quartile"` lines are dashed and dark, not white, corrected to match the notebook |
| 10 | Very High (~97%) | Every seaborn/pandas/scikit-learn call run on seaborn 0.13.2 / pandas 2.3.3 / scikit-learn 1.7.2 (pairplot diagonal KDE and `diag_kws` `common_norm`, `.corr()` diagonal 1.0 and symmetry, wine `Total_phenols`~`Flavanoids` 0.86 and `~Alcalinity_of_ash` -0.32, `StandardScaler` init/fit/transform on penguins, output shape (344, 4)); the load-bearing statistics pinned down: `StandardScaler.scale_` is the population standard deviation (ddof=0), not the sample std, and standardized `describe()` reads 1.0014652 = sqrt(342/341) while the population std is exactly 1 (the quiz answer); the FIRST note built with the expanded Double Diamond QC (a truth diamond, then a clarity diamond, then a full-document release gate), whose meta-auditor caught a `dark` versus `darkgrid` conflation hazard, same gray background, differing only by the grid |
| 11 | Very High (~96%) | Every numpy/pandas call and number run on numpy 2.4.6 / pandas 3.0.3 (`linspace(0, 25, 21)` endpoint-inclusive 21-point grid, the river-depth `4780 + 2.89·rainfall` arithmetic, the seed-42 linear and seed-9 polynomial notebook draws reproduced bit for bit, the sine and x² feature-transform ranges, and `standard error = σ/√n` confirmed by a 200k-trial simulation); the load-bearing idea pinned down, that "linear" means linear in the coefficients (β), not the input, so a `sin(x)` or `x²` feature transformation keeps a curved relationship inside linear regression while a coefficient trapped inside a function (`y = β₀ + e^(β₁x)`) does not; full Diamond QC found zero factual or fidelity errors and confirmed every invented cross-section bridge, the only fix a malformed map-table header (a rendering defect, not a fact) |
| 12 | Very High (~97%) | Every statsmodels/pandas call and number run on statsmodels 0.14.5 / pandas 2.3.3 against the live-fetched 97-city food-truck data (the `smf.ols` fit reproduced to six decimals: intercept -3.8958, slope 1.1930, R-squared 0.702; `.bse`, `.conf_int()`, and `.pvalues` cross-checked against the summary table; the "about 2 x SE" rule pinned to the exact t multiplier 1.9853; the `get_prediction().summary_frame()` confidence and prediction intervals rebuilt by hand from `mean_se` and the residual variance `model.scale` and matching to six decimals; the prediction interval wider than the mean interval at all 101 grid points, and empirically containing 93 of 97 training points while the mean interval holds only 22, each keeping exactly its promise); the load-bearing correction pinned down, that a p-value is the probability of data at least this extreme computed assuming the true coefficient is zero, not "the chance the coefficient is zero" as the lecture and lab loosely say (kept, with an exam-safety note), plus the lecture's flipped CI-crossing-zero pairing corrected; full Diamond QC where both auditors independently isolated the same single numeric slip (a confidence-band edge width misstated as 3.16, actual 1.59, fixed and re-verified), and the meta-auditor caught the one nuance both missed, that the cities cluster at low populations with the input mean at the dense region's upper edge |
| 13 | Very High (~97%) | Every statsmodels/pandas/scikit-learn call and number run on statsmodels 0.14.5 / pandas 2.3.3 / scikit-learn 1.7.2 across six datasets (a seed-2100 additive synthetic, seaborn's penguins and iris, the live-fetched 150-apartment CSV, and two unseeded synthetic curves): the additive fit reproduced to six decimals (4.726926 / 1.967544 / -2.908475, R-squared 0.944, and the 909-row prediction grid whose parallel lines drop exactly 1.8119 per x2 step), the standardization invariants pinned down (slope p-values and R-squared 0.7884 identical raw vs standardized, standardized intercept exactly the outcome mean 200.915205, and the StandardScaler ddof=0 vs pandas ddof=1 subtlety that makes standardized columns read std ~1.0015), iris dummy coding verified to the digit (each coefficient = its group mean minus the reference mean, 2.798 = 4.260 - 1.462), the three interaction spellings (`a + b + a:b`, `a * b`, `(a + b)**2`) proven byte-identical with the in-center slope 2996.99 + 870.57 = 3867.57 confirmed on the 202-row fanning grid, and the patsy landmine proven by execution (`y ~ x**2` silently collapses to plain `y ~ x` with near-zero R-squared; `np.power(x, 2)` or `I(x**2)` is required); full Diamond QC where the execution auditor caught the one executable defect (the shown visualization code never rebound `model` to the apartment fit, fixed and re-run to the exact 221145 / 106511 / 623759 / 418498 corner predictions) and the meta-auditor caught the shared blind spot both auditors underrated, that the unseeded sine and parabola "behavior" bands were drawn tighter than the true sampling spread (a 5000-run Monte Carlo put the stated sine R-squared band at 15% coverage, contradicted by the note's own cited seed values on the same page), all bands widened to the true spread and re-verified |

## A note of caution

This is still a study aid, not a textbook. Use it to understand and review, but
check anything important against the actual course material, especially before
a quiz.

## Contributors

- **DatJavaClass (Victor S)**, author and director. Conceived these notes, established their format and structure, directed their creation, and fact-checked, edited, and quality-controlled every one, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.