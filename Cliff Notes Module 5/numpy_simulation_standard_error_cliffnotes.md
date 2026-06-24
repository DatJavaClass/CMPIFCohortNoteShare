# NumPy: Simulating the Standard Error of the Mean

## Concept

The standard error of the mean (SEM) is the standard deviation of the means
of repeated samples drawn from a population. You can't sample infinitely, but
you can approximate the process by drawing many random samples and looking at
how their means spread. NumPy makes this far more efficient than looping over
Python lists.

## Setup

- Import NumPy.
- Build a random number generator: `rg = np.random.default_rng(seed)`.
  - `default_rng` is a function inside NumPy's `random` module, so the dot
    notation matters (`np.random.default_rng`, not an underscore).
  - Seed it (e.g. `2100`) for reproducible results.

## Building the simulation

Draw a sample of size 7 from a uniform distribution on [0, 1], and repeat that
5,000 times. Store the result as a 2D array: 5,000 rows (replications) x 7
columns (sample values).

```python
num_samples = 7
num_reps = 5000
random_array = rg.random((num_reps, num_samples))   # shape (5000, 7)
random_array.shape          # (5000, 7)
random_array[:5, :]         # peek at first 5 samples
```

`rg.random(...)` draws from a uniform [0, 1) distribution; the tuple argument
sets the output shape.

## Sample means -> standard error

Take the mean of each row (each sample). The mean is taken *across columns*,
which is `axis=1`.

```python
random_array.mean()         # WRONG: collapses the whole array to one number (~0.5)
random_array.size           # 35000 = 5000 * 7

sample_means = random_array.mean(axis=1)   # shape (5000,) -- one mean per sample
```

The SEM is the standard deviation of those sample means. Use `ddof=1` for the
unbiased estimator.

```python
sem = sample_means.std(ddof=1)   # ~0.1096 for n=7
```

## One-liner

NumPy lets you chain the whole thing: simulate, take row means, take the std.

```python
sem = rg.random((num_reps, num_samples)).mean(axis=1).std(ddof=1)
```

Same answer as the step-by-step version.

## Effect of sample size on SEM

Larger samples -> smaller standard error. Sweep sample size from 10 to 100 and
compute SEM at each:

```python
sample_sizes = 10 * np.arange(1, 11)   # 10, 20, 30, ... 100

sem_vs_size = [
    rg.random((num_reps, n)).mean(axis=1).std(ddof=1)
    for n in sample_sizes
]
```

Plotting `sem_vs_size` against `sample_sizes` gives a downward curve: as
sample size rises, the standard error falls (more data -> a more confident
estimate of the mean).

## Key takeaways

- SEM = std of repeated sample means; simulate it with many random draws.
- Organize draws as a (replications x sample_size) 2D array.
- `axis=1` = across columns = per-row (per-sample) operation.
- `ddof=1` gives the unbiased standard deviation.
- NumPy chains operations into a single efficient line.
- SEM shrinks as sample size grows.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2100 Faculty and Assistants.*
