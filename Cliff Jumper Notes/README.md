# Cliff Jumper Notes

## What these are

A Cliff Jumper Note takes all the separate Cliff Notes from one module and
weaves them into a single, connected lesson. Instead of six short notes that
repeat the same ideas, you get one walkthrough that introduces each idea once
and builds on it. Where it helps, a small amount of extra explanation or
example is added, but only when it is correct.

There is one note per module (Modules 1 through 5).

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

## A note of caution

This is still a study aid, not a textbook. Use it to understand and review, but
check anything important against the actual course material, especially before
a quiz.
