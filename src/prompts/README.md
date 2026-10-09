# prompts

Every prompt the benchmark sends, and nothing else — no HTTP, no scoring
logic, no orchestration. One file per task:

| File | Task prompt | Judge prompt | Rubric |
| --- | --- | --- | --- |
| [summarization.go](summarization.go) | summarize a call transcript | grade the summary against the transcript | faithfulness, coverage, conciseness |
| [extraction.go](extraction.go) | extract participants/action items/decisions/entities as JSON | grade the extraction against the transcript | groundedness, completeness |
| [correction.go](correction.go) | clean up disfluencies without changing meaning | grade the cleanup against the corrupted transcript the model was shown (reference-free) | faithfulness, improvement |
| [dictation.go](dictation.go) | apply the spoken dictation command grammar (comma/period/all caps/new paragraph/...) | *(none — see below)* | exact_match, word_error_rate, punctuation_accuracy, capitalization_accuracy, formatting_command_accuracy |

**Dictation's Judge is the one exception to the pattern below.** Its Task is
a normal fixed prompt, sent to every record exactly like the other three
tasks' — but its Judge is empty and its Rubric does not name an LLM rubric:
every dictation record is graded mechanically against a real answer key
(`expected.truth_text`), never an LLM judge call. See `dictationSpec`'s doc
comment in [dictation.go](dictation.go) and `../dictation.go`'s package doc
comment for the full picture; the rest of this page's "two kinds of prompt"
framing still applies to dictation's Task, just not its (nonexistent) Judge.

## Two kinds of prompt, never confused

Each task file defines a `Spec{Task, Judge, Rubric}`:

- **`Task`** is what a candidate model — one of everything the benchmark pulled
  from `GET /v1/models` — receives as its system prompt, to actually perform
  the task. This is the thing being measured.
- **`Judge`** is what a separate, fixed model (`-judge-model`) receives to
  *grade* the candidate's output. It runs against a different model than the
  one under test, on a different call, and is never itself benchmarked.

Mixing these up — e.g. accidentally sending the judge prompt to a candidate —
would silently make every quality score meaningless, so `Spec` keeps them as
two distinctly-named fields rather than a single prompt list, and every
caller in `../judge.go` and `../voicebench.go` reaches for one or the other by
name, never by position.

## Why every JUDGE prompt is reference-free

For summarization and extraction, the transcripts these prompts run against
(`../dataset.go`, sourced from `aai_transcripts.jsonl`) have no gold summary
and no annotated entity list — see the aai_transcripts adapter notes in
`../../jobs/summarization.py`. So a judge prompt is never "does this match the
reference" — it is always "does this hold up against the source transcript
it was produced from."

text_correction follows the same rule, though its data does carry an original
transcript (`text`) next to the corrupted `missed_text` the model is shown. That
original is not an accurate answer key, so nothing is scored against it: the
`Judge` prompt in `correction.go` sees only `missed_text` and the candidate's
output (see `correctionSpec`'s doc comment). Every `Judge` prompt in this
package receives only the input the candidate model was shown as grading
context.

## Where the rubrics come from

- **Summarization** follows SummEval's (Fabbri et al., 2021) evaluation
  dimensions, narrowed to the two that actually separate modern models
  (faithfulness, coverage) plus conciseness, and scored using G-Eval's
  (Liu et al., 2023) form-filling paradigm: state the criteria, ask for brief
  chain-of-thought per dimension, then a JSON score.
- **Extraction** follows the instruction-based structured-extraction pattern
  from the LLM-IE literature (e.g. ChatIE, Wei et al. 2023): the target schema
  is stated in full inside the prompt, and the model must emit empty
  arrays/`"unspecified"` rather than omitting keys, which is what lets
  `ValidateExtraction` (`../judge.go`) check structural validity mechanically,
  with no model call spent on it. The groundedness/completeness rubric is a
  reference-free stand-in for classic precision/recall.
- **Correction** follows the constrained-rewriting contract from OCR/ASR
  post-correction work (see the HIPE-OCRepair task framing already used in
  `../../jobs/text_correction.py`): restore intended wording, never paraphrase.
  The faithfulness/improvement rubric is reference-free: it judges whether the
  readability fixes are good and whether any content was changed, against the
  transcript the model was shown.

Full rationale for each rubric is in that task's own file — read the doc
comment above `summarizationSpec`, `extractionSpec`, or `correctionSpec`
before changing it.

## Adding a task

1. Add the task name as a new constant in `prompts.go` and append it to `All`.
2. Add `<task>.go` with a `<task>Spec = Spec{Task: ..., Judge: ..., Rubric: ...}`,
   documented like the three above; `prompts_test.go` checks every rubric against
   the JSON shape its prompt asks for.
3. Register it in `specs` in `prompts.go`.

Nothing outside this package should ever need to change — `../voicebench.go`
and `../judge.go` iterate `prompts.All` and look specs up by name.

This is the pattern for a task that fits the shared assumptions every step
above leans on: one fixed system prompt for every record, and a judge-model
quality score. Dictation shares the first assumption (its Task is a normal
fixed prompt) but not the second — see `dictationSpec`'s doc comment in
[dictation.go](dictation.go) — so it still needed its own loader, runner, and
scorer outside this package (`../dictation.go`), not just its own file in
here. If a future task also breaks the judging assumption, follow
dictation's shape instead of this one: register a `Spec` here (a real Task,
an empty Judge) so `-tasks` validation and `prompts.All` still work, but give
it its own file alongside `../dictation.go` for the parts that don't fit
`../voicebench.go`'s `runOne`/`runModel`.
