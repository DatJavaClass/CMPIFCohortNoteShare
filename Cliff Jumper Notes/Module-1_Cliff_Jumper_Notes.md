# Module 1 — Cliff Jumper Notes

*One continuous lesson stitched from the Module 1 cliff notes: the course's
framing, what data and data science actually are, the pipeline that organizes
the whole program, and the single piece of Python this module hands you.*

---

## 1. What this course is really teaching

CMPINF-2100 (*Introduction to Data-Centric Computing*) sells three concrete
skills — programming for data analysis in Python, data visualization through
code, and making predictions from patterns in data — but the framing that
matters is the one underneath them: this is a **practice**, a way of
interacting with a computer and with data through programming, not a checklist
of tricks. Everything downstream in the MDS program builds on it.

It starts from scratch. No programming background is assumed, and a wide mix of
backgrounds is expected. If you've coded before, the early modules will feel
fast; if you haven't, front-load your time here and keep the Python intro
material within reach. The instructor, Michael Miller-Yoder, is himself a
case study in the course's own premise: a CS undergrad who picked up data
science not from formal stats coursework but by *doing* — writing Python in a
psychology lab to store brain images, then carrying the same toolkit into text
and online-community research. The transferable lesson is structural, not
biographical: **the same data science toolkit moves across wildly different
domains**, and the field is reachable from any adjacent technical role.

---

## 2. Data — and why it is never "raw truth"

**Data is structured information about the world** — observations and recorded
measurements. Inputs can be numeric outright, or derived from text, images, or
audio that get processed into structured form first.

The caveat is more important than the definition: data is *not* raw truth.
Somebody — or a sensor someone designed — did the observing, so data can be
incomplete, imperfect, and subjective. Keep that human/instrumentation layer in
view at all times; it's the thread that reappears at every later stage.

**The shape of data** is a table of observations (also called data points or
instances — the terms are interchangeable):

- each **row** is one observation,
- each **column** is one **attribute**,
- each **cell** is that attribute's **value** for that observation.

Take the Parthenon as a single row: `name → Parthenon`, `material → Marble`,
`year built → 438 BCE`. Even here the provenance caveat bites — does "year
built" mean construction started or finished, and how precise is any estimate
for ancient ruins? Interrogating what an attribute *actually means* is part of
the work, not a preliminary to it.

This row/column/cell picture is worth holding onto, because the definition of
data science is going to be expressed directly in its terms.

---

## 3. Data science — two goals, read straight off the table

**Data science is collecting, processing, and manipulating data toward two
goals**, and both map cleanly onto the table above:

- **Insight** — finding patterns and relationships *among attributes*, i.e.
  **across the columns**. Across a set of ancient buildings: which materials
  appear in which eras? Add a location column and you can relate region × era ×
  material.
- **Prediction** — estimating attribute values for observations *not yet seen*,
  i.e. **a new row**. A newly discovered ruin: given its material and location,
  predict its likely build year from the patterns learned in known ruins.

So data science is column-pattern-finding (insight) plus extrapolation to
unseen rows (prediction). That's the whole definition, and it's why the table
structure was worth nailing down first.

### Where it sits among its neighbors

The course places data science against four adjacent fields — useful for
knowing what data science *is* by contrast:

- **Artificial Intelligence** leans heavily on prediction-from-data, but its
  goal differs: AI builds systems to *accomplish tasks* (generate language,
  generate images), whereas data science treats insight or prediction from a
  dataset as the end product itself.
- **Machine learning** shares the fundamental technique — learn patterns from
  data, use them to predict — but is more computational. You won't build
  self-driving cars or a ChatGPT here; you *will* use the same core idea.
- **Statistics** is the closest relative: inferring structure and patterns from
  data, and used *inside* data science for pattern identification and
  significance testing. The difference is emphasis — stats trends more abstract
  and cares less about cleaning, organization, and computation. The "data
  science is just rebranded statistics" jab has some truth to it.
- **Database management / engineering** handles organizing and storing data;
  data science borrows from it without going as deep into storage architecture.

Mental model: **data science sits at the intersection** — the data-organization
side of database work joined to the computational/predictive side of ML.

---

## 4. The pipeline — the backbone for everything above

The two goals tell you *what* data science produces; the **data science
pipeline** tells you *how* you get there. It's the process used throughout the
course and the program, and it rests on **CRISP-DM** (Cross-Industry Standard
Process for Data Mining), a framework worth looking up because you'll meet it
again. The practical habit it teaches: when you're buried in technical detail,
"pop up a level" and name which step you're on, so purpose stays in view.

The course's six steps:

1. **Research question / business objective.** Always start here — you want to
   know *something*, and it has to be answerable with quantitative measures.
   (Early diabetes indicators? Which customers am I losing? Where will housing
   prices move? What goes viral?)
2. **Select and understand the data.** Which datasets can answer the question —
   and the provenance questions from Section 2 become operational here: *why*
   was this collected, *how*, how representative is it, and **what's missing?**
   The instructor's bot-dataset example is the warning: seed your collection
   from only political bots and you silently miss news bots entirely. Scrutinize
   the collection method; hunt for what isn't there.
3. **Prepare and clean the data** — famously ~**80%** of the job. Data arrives
   incomplete (handle missing values), cluttered with irrelevant fields (remove
   them), at the wrong granularity (aggregate cities into regions, sometimes
   *creating* new attributes), split across sources (combine and integrate), and
   mistyped (numbers stored as strings, inconsistent abbreviations — normalize
   it).
4. **Explore the data.** Surface basic trends, counts, and distributions —
   most/least frequent values, how many points fall across attribute values,
   summary statistics, trends bearing on the question. **Steps 3 and 4 are
   iterative, not sequential:** clean → explore → spot an outlier → return to
   cleaning. An outlier may be a mistake to drop *or* the real signal — judgment
   required.
5. **Statistical modeling**, usually **predictive modeling**, and structured
   exactly like the "prediction" goal from Section 3: one attribute is the
   **outcome** you want to predict, the others are candidate **predictors**, and
   you test which predictors actually associate with the outcome. (Outcome:
   developing diabetes; predictors: health-history attributes.)
6. **Communicate the results.** Evaluate and interpret the models, then tell
   the story to stakeholders who *don't* share your background: translate, convey
   the trends and predictions, and explain why they answer the original
   question. This closes the loop back to Step 1 — analysis that can't be
   explained to a stakeholder hasn't finished the pipeline.

Notice the spine running through it: the provenance awareness from Section 2
*is* Step 2, and the prediction goal from Section 3 *becomes* Step 5. The pipeline
isn't new material so much as the earlier ideas put in working order.

---

## 5. The one Python tool this module hands you: list comprehensions

Step 3 — preparation — is where you'll spend most of your time, and most of that
work is the same shape over and over: **take a list, do something to each
element, build a new list of the results.** Python's idiom for exactly that
pattern is the **list comprehension**, and that's why this otherwise-conceptual
module pauses to teach it.

Start with the long form, a plain for-loop:

```python
fruits = ["apple", "orange", "banana"]
salads = []                          # the empty list MUST come before the loop
for fruit in fruits:
    salad = fruit + " salad"
    salads.append(salad)
# salads == ['apple salad', 'orange salad', 'banana salad']
```

Every loop of this kind contains the same **four pieces**:

1. the **new list** you're building (`salads`),
2. the **iterating variable** — the name you give each element, which *you make
   up* (`fruit`),
3. the **old list** you read from (`fruits`),
4. the **action** that turns each element into its new value (`fruit + " salad"`).

Identify those four and you can mechanically collapse the loop into one line:

```python
salads = [fruit + " salad" for fruit in fruits]
# new_list = [   action      for   variable   in  old_list ]
```

Same output, one line. The brackets do two jobs the loop needed separate lines
for — they **create** the new list *and* **append** to it automatically — which
is why a comprehension needs no preceding `salads = []`.

Three traps worth internalizing from the loop version, because they explain *why*
the comprehension is built the way it is:

- The empty list must be initialized **before** the loop; inside, you'd wipe it
  every iteration. (The comprehension sidesteps this entirely — the brackets
  initialize.)
- The iterating variable must match between the `for` clause and the action, and
  **Python is case-sensitive** — `fruit` and `Fruit` are different names.
- Singular/plural naming is a deliberate signal: `salads` (plural) is the
  container, `salad` (singular) is the current item.

One conceptual distinction the lecture leaves implicit but is worth making
explicit: **a list comprehension is *syntax*, not a *method***. Methods like
`.append()`, `.upper()`, or `.sort()` are named functions you call with
parentheses on an object. Syntax — for-loops, `if` statements, comprehensions —
is grammar built into the language; you don't call it, you *write code in that
shape* and Python recognizes it. There's no "list-comprehension function" to
import or look up. And the same grammar generalizes: `{k: v for ...}` for dicts,
`{x for ...}` for sets, `(x for ...)` for generator expressions — different
brackets, identical idea.

Tie it back to the pipeline: when Step 3 asks you to normalize every value in a
column — say, lowercase a list of messy category labels — the comprehension *is*
the move:

```python
labels = ["Marble", "BRONZE", "limestone"]
clean = [label.lower() for label in labels]
# clean == ['marble', 'bronze', 'limestone']
```

That is the bridge from Module 1's concepts to the keyboard.

---

## 6. Practical: where to get help (logistics, not theory)

Separate from the conceptual material but worth keeping straight, the course
routes support by question type:

| Need | Channel |
|---|---|
| Peer collaboration, non-urgent public Q | **Slack** (`CMPINF2100` channel) |
| Stuck on an assignment/quiz, non-urgent | **Discussion Forums** (Coursera, by module) |
| Scheduled live help | **Office Hours** (Coursera "Live Events"; ~24h email reminder) |
| Urgent or private; guaranteed response | **Email** — `online-mds-support@pitt.edu` |

The one rule that actually matters: **only email guarantees a response.** Slack
and the forums are monitored but best-effort, and the teaching team does **not**
reply on Slack. When you email, put the course number **CMPINF 2100** (and the
module/assignment number) in the message — that's what creates the support
*ticket* and routes it correctly; mail to any other address may go unnoticed.
Academic-integrity rules apply on the public channels: don't post your answers
or large chunks of your own solution code.

---

## One-paragraph recap

Data is structured, human-mediated information shaped like a table; data science
reads patterns *across columns* (insight) and predicts *unseen rows*
(prediction); the CRISP-DM-based six-step pipeline turns those goals into a
repeatable process whose center of gravity is the ~80% spent cleaning; and the
list comprehension — `[action for var in old_list]` — is the Python idiom that
makes that per-element cleaning work fit on one line.
