# Module 3 — Cliff Jumper Notes

*One continuous lesson stitched from the Module 3 cliff notes (Tips for Python
Programming, The Processes of Programming, Getting Stuck on Programming
Problems). This module isn't about new syntax — it's about how to actually
program: the mindset to adopt, the repeatable process to follow, and what to do
when you're stuck.*

---

## 1. The shift that makes it feel hard at first

If you're new to programming, the difficulty early on isn't really about
Python — it's about a change in *how you operate a computer*. Your whole life
you've **pointed and clicked**: menus, drop-downs, a fixed set of choices laid
out for you. Programming throws that away. Now you **write the commands
yourself** and invent names for "imaginary objects" that live in a world you're
constructing as you go. There's no menu to pick from — you generate everything.

The lecture has a word for how that feels: **unmoored**, floating free. That
disorientation is *normal* and nearly universal, even among capable students.
Most people — including those brand new to this — find their footing within a
few weeks to a few months. The point of this module is to hand you **anchors**
so you're not adrift: a mindset and a process to lean on instead of expecting to
know how to start from nothing.

---

## 2. The mindset: get something that *works* first

Programming is a **creative problem-solving process**, not a lookup task. Two
attitudes will save you a lot of early frustration:

- **There is no single "best way."** Some approaches are better than others, but
  *don't optimize early.* Elegance and efficiency are later concerns.
- **Your first goal is just a solution that WORKS.** Get correct output, then —
  much later, and only if it matters — worry about making it better.

Holding "working beats elegant" in mind is what keeps a hard problem from
feeling impossible. Refinement is a luxury you earn after something runs.

---

## 3. The process: a four-step loop with anchors

The frustration of programming is much easier to manage when you have a defined
set of steps. Here's the loop:

> **Design → Write → Correct Errors → Test → (loop back to Design as needed)**

### Step 1 — Design (don't jump straight to code)

You can poke at code a little, but stepping back to plan first pays off:

- **(a) Think INPUT and OUTPUT first.** A program takes some input, manipulates
  it (variables, objects, saved values), and produces an output. Know those two
  ends before anything in the middle.
- **(b) Write down EXAMPLES.** For a specific example input, what *exactly*
  should the output be? Write it down — you'll reuse it when testing.
- **(c) BREAK THE PROBLEM INTO STEPS.** This is the essential move; it turns an
  intimidating problem into manageable pieces. **Key insight for when you feel
  stuck:** you almost always already know how to do at least the *first* step,
  and often the *last* one (right before output). Start from what you know.
  Building a list, for instance: first step — initialize the list; last step —
  add items with `append()`. The unknown middle shrinks once the ends are in.
- **(d) Write PSEUDOCODE.** Plain-language instructions for what each step does.
  It's the bridge between the design and the actual code.

### Step 2 — Write the code

Translate each planned step into Python. This gets easier with familiarity ("I've
done that before, I just need this function," "I can look that up"). The
single most important truth here:

> **Almost all programming is adapting examples.** You'll rarely find an exact
> match for what you need — *adapting* one is itself a core skill, not cheating.

For this course, your primary examples to adapt come from the **course
material** — the homeworks are largely adaptations of it. Supporting resources
(covered later) include **documentation**, which tells you what functions exist,
what **arguments** they take, and what **methods** are available. (Terms in play:
functions, arguments, methods.)

### Step 3 — Correct errors

You *will* have errors — everyone does, including experienced programmers;
running into someone who never hits errors is basically impossible. The course
splits them into two kinds:

- **Syntax error** — the code is malformed, so it won't run. Classic causes are
  a typo or a structural mistake like a missing colon or an unmatched bracket:

  ```python
  for n in [1, 2, 3]      # SyntaxError: expected ':'  -> never runs at all
      print(n)
  ```

- **Logic error** — the code *runs fine* but produces the *wrong* result. These
  hide until you test:

  ```python
  def average(a, b):
      return a + b / 2     # bug: needs (a + b) / 2
  average(4, 6)            # returns 7.0, not the intended 5.0
  ```

> **One precision worth having (the course is loose here):** the course lists
> "using an undefined variable" as a syntax error. In Python specifically,
> that's actually a **runtime** error — a `NameError`. A true syntax error (the
> missing colon above) is caught *before* the program runs; an undefined name
> isn't caught until execution reaches that line, so the program has already
> started. For a beginner both feel like "it broke immediately," but the
> distinction matters once you start reading error messages: `SyntaxError` =
> won't start; `NameError`/others = started, then failed.
>
> *(Exam note: this course still classifies "undefined variable" as a syntax
> error in its own materials. Know the real distinction for your own debugging,
> but if a quiz keys to the course's wording, answer the course's way.)*

### Step 4 — Test

Go back to the example input/output you wrote in design. Did the example input
produce the correct output? Often the answer is *no* — and that's how you catch
logic errors. Testing is what turns "it runs" into "it's right."

### The loop is iterative — and that's not failure

Fixing a logic error often sends you back to Step 1 (design). This is **not**
starting over:

- You've usually already made significant progress.
- Students who feel totally lost typically have *most* of the pieces and are
  missing just one small step.
- "Going back to design" usually means **tweaking** the steps you already have,
  not rebuilding from scratch.

---

## 4. Getting stuck — what to actually do

Getting stuck is **inevitable** — the instructor describes hours of debugging,
late nights, real frustration. The reframe that matters: those hours are **not
wasted**, because you're learning *how* to do something, so next time you're
less lost. Being stuck is part of learning, not evidence of failure.

Notice this connects straight back to Step 1(c): the remedy for "I'm totally
stuck" is almost always **decompose and start from the step you do know.** The
feeling of being lost usually means one small missing step, not a missing whole.

**The resource trap.** When stuck, the temptation is to grab an answer in a way
that skips the thinking:

- **Generative AI** — asking directly and pasting the exact answer back.
- **Posted online solutions** — common for intro material, often copyable nearly
  wholesale.

Both hand you output without making you learn. (The course isn't anti-AI
wholesale — it promises a later video on what *good* use of AI and other
resources looks like; the trap is the copy-paste shortcut, not the tool.) The
course is blunt about why that's a bad trade:

- You're here to **learn, not just to get a grade** — someone is investing in
  this degree so that you *actually know something* at the end.
- Struggling more early pays off later in real, retained skill.
- **Job-market reality:** data scientists and programmers are tested in
  interviews on code they can produce *themselves*, not what an AI produced for
  them. The minimum bar — even when you *do* use AI — is being able to
  **understand and correct** its output, not just paste it.

**A habit worth stealing (offered as one method, not the One True Way):** when
you work through a stuck point, jot a quick note with example code into a
personal reference document for later reuse. It both cements the lesson and
builds your own library of examples to *adapt* later — which, per Step 2, is
what programming mostly is anyway.

---

## One-paragraph recap

The early disorientation is the click-to-command shift, and it's normal — lean
on anchors. Aim for *working before elegant*. Follow the loop: **Design**
(input/output, concrete examples, decompose starting from the step you know,
pseudocode) → **Write** (adapt examples, mostly from the course material) →
**Correct Errors** (syntax = won't run; logic = runs but wrong) → **Test**
against your example I/O → loop back as needed. When stuck, decompose from what
you know, refuse the copy-paste shortcut (understand-and-correct is the floor),
and keep your own notes of what you figure out.
