<p align="center"><img width="651" height="701" alt="ChatGPT Image Jun 11, 2026, 12_49_40 PM" src="https://github.com/user-attachments/assets/99da5453-5993-4993-8aac-51c0a4b6510c" /></p>

# Cliff Jumper Notes

## What these are

A Cliff Jumper Note takes all the separate Cliff Notes from one module and
weaves them into a single, connected lesson. Instead of six short notes that
repeat the same ideas, you get one walkthrough that introduces each idea once
and builds on it. Where it helps, a small amount of extra explanation or
example is added, but only when it is correct.

There is one note per module (Modules 1 through 10; Module 7 is split into
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

The most recent note, Module 10, went through an expanded version of this: two
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

## A note of caution

This is still a study aid, not a textbook. Use it to understand and review, but
check anything important against the actual course material, especially before
a quiz.

## Contributors

- **DatJavaClass (Victor S)**, author and director. Conceived these notes, established their format and structure, directed their creation, and fact-checked, edited, and quality-controlled every one, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.