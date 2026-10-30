# Does the draft read as AI-written? (check of `paper/main.tex`, 2026-10-29)

**Update 2026-10-30:** the draft was rewritten against this list ("ticker" for "ours", field terms for our private
words, colons from 166 to about 30, one "X, not Y" left, each headline number stated once in the results, no announced
honesty), and against the NeurIPS paper checklist (compute, licences, error bars, proof, broader impact). The list
below is kept as the reference for later edits.

Short answer: it avoids the well-known tells (no "delve", "crucial", "landscape", "it is worth noting", no lists of
three adjectives, few dashes). It has a different set of tells that a careful reader would notice. Most come from how
the draft was produced: written fast from our plan and result notes, in one voice, with the same sentence shapes
everywhere. Use this list when you rewrite by hand.

## What gives it away

1. **The "X, not Y" sentence, 16 times.** "The pitch of this paper is native 4-bit weights, not memory." "This is a
   property of Newton–Schulz, not of the 4-bit leftover." "...comes from the two programs' arithmetic, not from
   rounding." One or two are fine. Sixteen is a pattern that language models are known for. Keep the two that matter
   most (memory is not the pitch; LoRA's learning is stored in 16 bits), and say the rest positively.
2. **Our private words.** "pass line", "arm", "as written", "the guard fired", "a likely reading, not measured",
   "pitch". These come from our plan, not from papers. Readers expect "pre-registered threshold" or "criterion",
   "method" or "baseline", "set in advance", "we expected". "Pitch" is sales language.
3. **"Ours" as a noun, about 50 times.** "Ours ends 3.7 points behind", "ours gains 20.4". Papers name the method
   ("the ticker") or say "our method". Pick one name and use it.
4. **Announcing honesty.** "Assumptions, stated honestly." "...reported whether or not they hold." "It held more
   strongly than written." "we report the ones that failed." Doing it builds trust; announcing it reads like a
   salesperson. Keep the pre-registration and the failed lines, and let the reader see them for themselves.
5. **Every section has the same shape.** Setup, table, figure, then a paragraph headed "What it shows." (3 times),
   each sentence about the same length, almost always with a colon (166 colons in 8,300 words). Real papers vary:
   some results get one sentence, some get a page.
6. **The same numbers, word for word, in the abstract, the contributions and the conclusion** (0.0013, 0.013, 0.080,
   3.7). Say each number once in full, in the results. Elsewhere, say it in words ("within a thousandth of a nat").
7. **Labels in place of transitions.** "Two facts follow." "Three things happen:" "A likely reading, not measured:"
   This style is clear, but it is the style of notes. In your own voice, link the ideas with "because", "so", "but".
8. **Uniformly plain, flat tone.** The draft was written to "grade-12 maths, plain words" rules for our own reports.
   That suits a report for you; a paper can use the field's usual terms (master weights, round-to-nearest, stochastic
   rounding) without explaining each one.

## What is fine and should stay

- The structure (mechanism, method, experiments, failed lines, limitations) and every number.
- The pre-registration and the change-log appendix. They are rare in this field and are the paper's strongest
  signal of care.
- The "Use of AI tools" statement. Disclosing is now normal and protects you; hiding it is what damages a reputation
  if it comes out later.

## How to rewrite by hand without losing anything

- Work section by section with the PDF beside you. Keep every table, figure and number; rewrite only the sentences.
- Read each paragraph aloud. Where you would not say it that way to Luis, change it.
- After each section, compare its numbers against `runs/paper/*_RESULT.md`. I can check this for you at the end.

No AI-text detector is reliable (they flag human writing too), so do not tune the text to one. Write for a reader in
the field.
