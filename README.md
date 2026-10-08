# benchmark

A Go CLI that benchmarks every model behind `llm_gateway` on four tasks —
summarization, information extraction, text correction, and dictation —
against real AssemblyAI call transcripts (or, for dictation, a synthetic
dictation dataset — see "Dictation" below), tracking latency and quality
broken down by audio-length size. Summarization and extraction are graded
reference-free (there's no gold summary or annotated entity list to compare
against); text correction has a real answer key (see "Data source"), which a
second LLM judge reads: it scores how similar the model's output is to that
truth in actual meaning, in addition to a reference-free LLM judge.
Dictation also has a real answer key, and — unlike every other task — is
graded ENTIRELY mechanically, with no LLM judge call at all (see
"Dictation").

```bash
export ASSEMBLY_AI_TOKEN=...   # set in your environment; there is deliberately no -api-key flag
go run .
```

Run these commands from the `src/` directory (`cd src`): the `normalized/`,
`results/`, `results_ttft/`, `errors/`, and `errors_ttft/` directories live
one level up (`../normalized` etc., at the repo root). Tests run with
`go test ./...` from the same directory.

Run with `-h` for the full flag list — every flag has a one-line description
there rather than duplicated here. The API key is read only from the
`ASSEMBLY_AI_TOKEN` environment variable (there is no `-api-key` flag, so the
key never appears in the process list or shell history). `GET /v1/models`
needs no key at all; running completions does.

## What it measures, and how

For every (model × task × transcript) combination:

- **Latency**: total wall-clock latency and tokens/second (completion tokens ÷
  total latency), each reported as min/max/mean/p50/p95 (`Stats` in
  `src`). Time-to-first-token is also tracked but is always identical
  to total latency: `RunCompletion` calls the gateway synchronously (a single
  request, a single response), not via streaming, so there is no earlier
  moment to measure a first token against — see `RunCompletion`'s doc comment
  in `gateway.go`. The field is kept rather than dropped so the report JSON schema stays
  stable; expect the TTFT and latency columns to
  read identically for every row.
- **Quality**: an LLM judge scores the output against the SOURCE TRANSCRIPT for
  every task — reference-free, because summarization and extraction have no
  gold summary or annotated entity list to compare against. See
  `src` for the full rationale and the papers each rubric is
  drawn from. **text_correction is the exception**: `aai_transcripts.jsonl`
  carries `missed_text`, a simulated corrupted transcript, alongside the
  known-true original in `text` (see "Data source" below). The candidate model
  corrects `missed_text` and never sees `text`. Its output is judged twice: by
  the same reference-free judge as the other tasks (graded against `missed_text`,
  so it never sees the answer key either — `faithfulness`, `improvement`), and
  by a second, reference-based judge call that is shown `text` next to the
  output and scores how similar the two are in actual **meaning**
  (`meaning_similarity`, 1-5: `RunReferenceJudge` in `judge.go`, the `Compare`
  prompt in `prompts/correction.go`). That one is a judge call rather than a
  word comparison because the truth is a raw transcript and the task prompt asks
  for things a raw transcript doesn't have: the model is told to drop fillers,
  false starts and stutters, which the truth keeps (about 3% of the words at
  3600s are "um"/"uh" and another 1.5% are immediate repeats), and to leave
  numbers as spoken, which the truth writes as numerals. Compared word for word,
  a model doing exactly what it was told is marked down — a simulated, perfectly
  obedient one scored worse than doing nothing at 3600s. The judge is told to
  look past those differences (fillers, stutters, punctuation, capitalization,
  "5" against "five") and to count the ones that change what was said: a
  sound-alike swap ("week" for "weak"), a wrong name or amount, text that is
  dropped, added, or loops. `truth_wer` and `input_wer` — the output's and the
  uncorrected input's word error rates against the truth — are recorded beside
  it, but only as context: they are not part of `quality_score`. They flag a
  grossly wrong output (truncated, or a runaway generation), which is captured
  under `errors/`, and they feed `-debug`. text_correction therefore makes two
  judge calls per sample. Reports written before this scoring carry `accuracy`
  and `edit_ratio` (or, briefly, `error_reduction`) instead; summarizing ignores
  those, so re-run text_correction to get `meaning_similarity` for those rows
  (see `correctionSpec`'s doc comment in `prompts/correction.go`).

Every statistic is broken down **by audio-length bucket** (`30s`, `120s`,
`900s`, `3600s`, ... — whatever's actually present in the loaded data; buckets
are read off each record, never hardcoded, so a new bucket the normalization
side adds later works with no change here). That's the whole point of
grouping by bucket rather than reporting one number per model: a model that's
fast and accurate on a 30-second call and falls over on an hour-long one is a
different tool than a report that just says "fast, accurate" would suggest.

## Debug mode (`-debug`)

`-debug` makes `text_correction` print, as each sample is scored, the texts the
score was given, stacked one word per column, with the scores and the judge's
own note on the meaning comparison — so you can read why a score is what it is:

```bash
go run . -debug -tasks text_correction -models some-model -buckets 30s -max-samples-per-bucket 3
```

```
text_correction debug  model=some-model  reasoning_effort=default  bucket=30s  record=…
scores:  faithfulness=5  improvement=4  input_wer=33.33  meaning_similarity=4  truth_wer=11.11
judge:   the output still says budjet where the reference says budget
TRUTH    9 words -- the original text, the answer key (never shown to the model)
INPUT    3 edits from the truth, word for word (33.33% WER) -- the corrupted text the model was shown
OUTPUT   1 edit from the truth, word for word (11.11% WER) -- the model's corrected text
legend   [word] differs from the truth word for word (the judge scores meaning, so fillers, stutters and number formatting show here without costing the model)
         · no word here   [·] a truth word that row is missing

(truth words 1-9)
TRUTH  we  agreed  to  ship    the  budget    review  on  ·     friday
INPUT  we  agreed  to  [shop]  the  [budjet]  review  on  [um]  friday
OUTPUT we  agreed  to  ship    the  [budjet]  review  on  ·     friday
```

The brackets are a word-for-word comparison, deliberately stricter than the
score: a filler, a stutter, or "5" against "five" is bracketed here but is
exactly what the meaning judge is told to look past, so a bracket is something
to look at, not a mark against the model. Set the rows beside `meaning_similarity`
and the `judge:` line to see whether the verdict is fair. Each row is aligned to
the truth by the same minimum-edit alignment behind `truth_wer` and `input_wer`,
over exactly the normalized words they compare, so the bracket count in a row
**is** that row's edit count; if the view and those counts ever disagree a
`WARNING` line says so. Words are shown as written; case and punctuation
differences are visible but are not differences (the word comparison ignores
them), so they are not bracketed. Rows wrap at 120 columns, each block says
which truth words it covers, and a run of 40 or more words that is identical in
all three rows is folded into `… N identical words …` (with 8 words of context
kept on each side) so a 3,600-second transcript's handful of differences aren't
buried. It prints to standard output, one whole block per sample even with
concurrent judging, and changes nothing that is measured or recorded; with a
task other than `text_correction` in `-tasks` it has nothing to print and says
so. The code is `correction_debug.go`; the alignment is `alignWords` in
`editdistance.go`.

## Dictation

Dictation is included in `-tasks` (`-tasks dictation`, or alongside the other
three) and flows through the exact same staging and canonical-merge pipeline, but
is loaded and scored differently under the hood — see the package doc comment at the top of `src` for the full
picture.

Every record gets the exact same system prompt —
`prompts.dictationSpec.Task` (`src`), describing the
spoken-command grammar being tested ("comma" -> ",", "all caps `<word>`"
uppercases it, "new paragraph" starts a new paragraph, and so on) — exactly
like the other three tasks' fixed task prompt. Its records come from
`normalized/dictation/*.jsonl` (e.g. `synthetic.jsonl`), never
`aai_transcripts.jsonl` — the schema and filename both differ. Each record also carries its own `input.instruction` field, but it
is deliberately never read (see `rawDictationRecord` in `dictation.go`): the
one shared prompt above is the only one ever sent, regardless of what that
field happens to say.

What DOES still keep dictation off the shared `runOne`/`runModel` pipeline
is grading: every record already carries a real answer key
(`expected.truth_text`), so grading is entirely mechanical — `ScoreDictation`
in `src` computes some or all of `exact_match`, `word_error_rate`,
`punctuation_accuracy`, `capitalization_accuracy`, and
`formatting_command_accuracy` (whichever names that record's own `metrics`
field asks for) directly against `truth_text`, in code, with no LLM call
spent on it at all — unlike every other task, which always calls `RunJudge`.

Because it's scored mechanically, dictation's five score names aren't a 1-5
Likert judge rubric — they're percentages (0-100, except `exact_match`,
which is a plain 0/1) already in the units you'd read out of a raw report.
They still fold into the same `quality_score`/`task_quality` breakdown every
other task's scores do (see "Per-task quality breakdown" below) —
`src`'s `normalize` just knows to invert `word_error_rate` (lower is
better) and scale the rest.

## Time-to-first-token

The standard benchmark's `RunCompletion` calls the gateway synchronously, so
its `TimeToFirstToken` is always identical to `TotalLatency` — see "What it
measures, and how" above. Every run therefore also includes a separate pass
that measures a real time-to-first-token via streaming (`RunCompletionStream`
in `src`), for the 30s and 120s audio-length buckets only — TTFT is a
real-time-adjacent responsiveness signal, not the full transcript-size sweep
the standard benchmark runs. It never judges (no LLM-judge call, no schema
check): a TTFT sample only measures how fast the first token came back,
never what the model said.

The two passes run sequentially, never concurrently: running them together
would send streaming AND synchronous requests to the same candidate models
at once, compounding against the gateway's per-account-per-model rate limit
(see `NewGateway`) for BOTH passes' numbers, not just one. The TTFT pass has
its own everything besides the shared model-discovery/dataset-loading setup,
so it can never change a single number the standard pass produces: its own
sample/cell/report types (`src`) and its own staging directories
(`results_ttft/` and `errors_ttft/`, mirroring `results/` and `errors/`,
never shared with them).

The TTFT reports are merged into the summary: a TTFT latency breakdown
(`ttft_n` through `ttft_max`) appears in every `summary.csv`/table/JSON row by
(model, bucket) — new columns alongside the existing ones, blank for a bucket
the TTFT pass never measured (anything other than 30s/120s).

## Canonical result set

`results/canonical/<version>/` (version folder name uses dashes, e.g. `0.1.2` →
`0-1-2`) is a single running result set that accumulates coverage across
however many separate runs it took to benchmark every model — unlike
`results/`'s fresh, uniquely-named subfolder per run, it's the same
`report-<judge>.json`/`ttft-report.json`/`summary.json`/`summary.csv` files
updated in place. Appending is keyed by `(model, reasoning effort, task,
bucket)`: new data for a combination entirely REPLACES old data for that same
combination (however many records the old run sampled for that bucket), and
every combination the new run didn't touch passes through byte-for-byte
unchanged — e.g. a run restricted to `-buckets 3600s` only replaces that
bucket's rows, leaving `30s`/`120s`/`900s`'s already-merged data alone. See
`src`'s package doc comment for why a naive recompute-on-every-append is
wrong.

Every run does the whole thing in one command: it runs the standard benchmark
AND the TTFT pass back-to-back (sequentially, never concurrently, for the same
rate-limit reason as above) for `-models` x `-tasks`, stages both under
`results/` and `results_ttft/`, prints a combined review summary so a human
can sanity-check the numbers, then either merges into canonical immediately
(`-approve-merge`, for scripted/CI use) or asks for interactive confirmation
on stdin first — the same plan-then-confirm shape as `terraform apply`.
Declining leaves both staged directories on disk untouched. A successful merge
re-runs the summary code over the canonical version directory afterward, so its
`summary.json`/`summary.csv` reflect the merge immediately.

```bash
go run . \
  -models new-model-1,new-model-2
```

## Per-task quality breakdown

Every `summary.csv`/table/JSON row also carries a `task_quality`
breakdown alongside its combined `quality_score` — one more 1-100 score per
task that contributed a judged sample to that (model, bucket) row, computed
the same way `quality_score` is but restricted to just that task's own
samples, so `summarization`'s score is never diluted by `text_correction`'s
or `information_extraction`'s. The row's `quality_score` is the equal-weight
mean of these per-task scores: every task counts the same regardless of how
many samples it ran, and a task with no judged samples (not run at all) is
left out of the mean rather than counted as 0. In the CSV this is one
`quality_<task>` column per task actually present in the data (discovered,
not hardcoded — see `taskColumns` in `src`), blank for a task
that contributed no judged sample to that row; in the table/JSON it's the
same breakdown under `TASK QUALITY`/`task_quality`. Filtering the one
`summary-*.csv` for a (model, bucket) shows both the combined score and
each task's own score side by side — no separate files. See `summarize`'s
doc comment in `summary.go`.

## Cost per quality

Every `summary.csv`/table/JSON row also carries `total_cost_usd` and
`cost_per_quality` alongside `quality_score`. `total_cost_usd` is the real USD
spent on the judged samples behind that score (each sample's actual
prompt/completion tokens times the candidate model's global per-token pricing,
captured from `GET /v1/models` when the model list is pulled — see
`ModelInfo.Cost` in `src`). `cost_per_quality` is the **mean** cost of one
judged sample divided by the score — `(total_cost_usd * 100 / judged samples) /
quality_score`, scaled to **cents**, i.e. cents per call per quality point — not the total over the
score: a total grows with how many samples a row holds, so a model run with a
smaller `-max-samples-per-bucket`, or one whose calls errored more often, would
look cheaper than an identical model with more samples. Lower is more
cost-efficient, and the ratio is comparable across models for the same
bucket/task since `quality_score` is always on the same 1-100 scale. Because a
single call costs a tiny fraction of a cent, the value is rounded to 4
*significant* digits (e.g. `5.046e-05`), not to a fixed number of decimal
places. `task_cost_usd`/`task_cost_per_quality` mirror `task_quality` above:
the same breakdown restricted to one task's own samples (that task's own mean
cost over its own score). A row with no judged samples carries neither, the
same "no data" convention as `quality_score` itself. See `summarize`'s doc
comment in `summary.go`.

## Use cases (`use_cases.json`)

Every time `RunSummary` refreshes the canonical version's `summary.json` (i.e.
after every benchmark run's merge — see "Canonical result set" above), it also
derives `use_cases.json` in the same directory: a mechanical, no-LLM-call
pass over that run's `combined` rows (see `src`) that names a short
list of use cases per audio-length bucket — e.g. "Budget Summarization",
"Real-Time Voice Agent" — each backed by the 1-3 models that best fit it in
that bucket, given `quality_score`, `cost_per_quality`, latency `p50`, and
`task_quality`. Every use case first gates on a minimum quality bar (a
"premium" or "acceptable" floor on either the combined `quality_score` or
one task's own `task_quality`) before cost, latency, or throughput ever rank
anyone — so "cheapest" or "fastest" never surfaces a model that's merely
cheap or fast because it's bad. Per-task use cases are generated for
whichever tasks actually have data in that bucket (dictation, for example,
never covers the 900s/3600s buckets); latency-flavored use cases
("Real-Time ...") are only generated for buckets at or under 120s, and the
long-form-batch one only for buckets at or over 900s.

"Near-Real-Time ..." sits next to each "Real-Time ..." use case: instead of
ranking by latency outright, it caps eligibility at `nearRealtimeLatencyMultiplier`
times the fastest acceptable-quality model's own p50 in that bucket (see
`fastestLatencyP50`) and ranks by quality within that budget — a step up in
latency for a real quality gain, rather than the fastest answer available
full stop. A use case with zero eligible models in a bucket is omitted
rather than written out empty. Only generated for the canonical version dir,
never a one-off staged-review directory — see `RunSummary`'s doc comment in
`src`.

Right after `use_cases.json` is written, `RunSummary` also renders it into
`use_cases.html` (`src`) — a standalone, dependency-free report
you can open directly in a browser: one card per use case, grouped by
audio-length bucket, with search/tag/bucket filtering in vanilla JS and no
build step. It's a pure re-render of `use_cases.json` (read back off disk,
not passed in memory), so the two files never drift apart, and the HTML can
be regenerated on its own from any existing `use_cases.json` by calling
`writeUseCasesHTML` — e.g. after hand-editing the page template.

`-refresh-canonical-summary` recomputes `-canonical-version`'s summary.json/
summary.csv/use_cases.json/use_cases.html from whatever it already has staged
and exits — no gateway, `ASSEMBLY_AI_TOKEN`, or benchmark run needed (see `RunRefreshCanonicalSummary`
in `src`, a thin wrapper around the same `refreshCanonicalSummary` a real
merge already calls). Useful after changing `use_cases.go` itself, when the
underlying report data hasn't changed but the derived files should:

```bash
go run . \
  -refresh-canonical-summary -canonical-version 0.1.3
```

## Data source

This section covers summarization/information_extraction/text_correction's
shared dataset; dictation's is entirely separate — see "Dictation" above.

Only `aai_transcripts.jsonl` files are read — never `unified.jsonl`,
`unified_reference_free.jsonl`, or any other dataset's normalized file sitting
in the same folder (`mederrbench.jsonl`, etc). Each task reads
`normalized/<task>/aai_transcripts.jsonl` at the repo root (`../normalized` from `src/`).
See `src`'s doc comment on `LoadTranscripts`.

`text_correction/aai_transcripts.jsonl` carries one extra field the other two
tasks' files don't: `missed_text`, a simulated corrupted transcript, at the
top level of the record (a sibling of `input`, not nested inside it) —
`text` is the known-true original it was derived from. `TranscriptRecord`
decodes it into `MissedText`, which `taskInput` (`main.go`) uses as the
correction task's actual input; `MissedText` is empty for every
summarization/extraction record and for the handful of text_correction
records the corruption simulation didn't cover — both are normal, not
errors, but a text_correction record with no `MissedText` can't be
benchmarked (there's nothing to correct) and fails as a per-sample error
rather than silently falling back to `text` and handing the model the
answer key.

## Judge model and self-grading

`-judge-model` (default `claude-opus-5`) is a fixed model used
only to grade output — never itself benchmarked as a candidate by default.
If you don't pass `-models`, the judge model is automatically excluded from
the candidate set so it never grades its own output (self-preference bias —
see Zheng et al., 2023, "Judging LLM-as-a-Judge and Benchmarking..."). If you
pass `-models` explicitly and include the judge model in it, that's respected
as a deliberate choice, with a warning logged.

text_correction makes two judge calls per sample, both through this judge model
and both under the same `-concurrency` cap: the reference-free one, and the
meaning comparison, which is the only prompt in the benchmark that is shown a
task's answer key (see "What it measures, and how"). If either fails, the sample
is left unjudged rather than scored on half a rubric.

## Concurrency and the gateway's rate limit — read this before raising `-concurrency`

The gateway enforces a per-account, per-model rate limit (default 30
requests/minute — `pkg/service/rate_limit`) that returns 429 under load. Two
things in this tool exist specifically because of that:

1. **No client-side retries.** `NewGateway` disables the OpenAI SDK's default
   of 2 retries. A retried request that succeeds on its second attempt would
   report that attempt's latency as the model's — silently corrupting exactly
   the numbers this tool exists to measure. A 429 is recorded as a failed
   sample instead (see `Cell.Errors` in the report).
2. **Two separate concurrency caps.** `-concurrency` bounds requests to each
   CANDIDATE model independently — safe to run high, since different
   candidates don't share a rate-limit bucket. But every candidate's samples
   also call the SAME judge model, which has no such isolation: with N
   candidates each running at `-concurrency`, up to N×`-concurrency` judge
   calls could fire at once. `main.go` shares one additional semaphore
   (`judgeSem`) across every candidate's goroutines specifically to cap judge
   concurrency too — see the doc comment on `judgeSem` in `main.go` and
   `TestJudgeConcurrency_isCappedAcrossModels` in `main_test.go`, which proves
   it.

## Layout

```
benchmark/
  voicebench.go  flags, model discovery, orchestration, concurrency
  gateway.go     GET /v1/models (raw HTTP, incl. global pricing) + synchronous chat completions (openai-go SDK)
  metrics.go     Sample, Stats (percentiles), Cell aggregation
  dataset.go     aai_transcripts.jsonl loading, bucket grouping/sampling
  judge.go       LLM-judge scoring, schema validation
  correction.go  text_correction's word-level context against the truth (truth_wer, input_wer) and its gross-failure check — the quality score is the meaning judge, see "What it measures" above
  correction_debug.go  the -debug aligned truth/input/output review — see "Debug mode" above
  editdistance.go  banded word-level Levenshtein distance (and alignment) behind both, sized for 15,000-word transcripts
  dictation.go   dictation's own dataset loading, runner, and mechanical scoring — see "Dictation" above
  report.go      stdout table + report JSON
  ttft.go        separate streaming-only TTFT path — see "Time-to-first-token" above
  summary.go     reads report-*.json/ttft-report*.json and writes a by-judge/combined summary.json+csv+table, incl. cost per quality
  use_cases.go   post-summary use_cases.json derivation for the canonical version dir — see "Use cases" above
  use_cases_html.go  renders use_cases.json into the standalone use_cases.html report — see "Use cases" above
  canonical.go   accumulating canonical result set, plus the run+review+merge orchestration — see "Canonical result set" above
  prompts/       every prompt sent, task vs. judge kept explicitly separate — see prompts/README.md
```

No interfaces, no factories, no plugin registry: `Gateway` is a concrete
struct, every dependency is a concrete type, and the whole tool is one flat
`package main`. There's exactly one thing this tool talks to (the gateway)
and one shape of data it reads (`TranscriptRecord`), so there was nothing to
abstract behind.

## References

Prior work that this benchmark's prompts and design cite. Each entry lists
where the code or docs actually cite it, then what it contributed.

- **Inspired**: an idea or convention re-implemented from scratch, with
  nothing taken from the reference's source.


- **[SummEval](https://arxiv.org/abs/2007.12626)** — Fabbri et al., 2021,
  "SummEval: Re-evaluating Summarization Evaluation"
  - **Cited in the code/docs:** [`src`](src/prompts/summarization.go),
    [`src`](src/prompts/README.md)
  - **Inspired:** the summarization judge's rubric dimensions. SummEval's
    consistency and relevance became `faithfulness` and `coverage`, its
    coherence and fluency were dropped, and `conciseness` was added (see
    `summarizationSpec` in [`src`](src/prompts/summarization.go)).

- **[G-Eval](https://arxiv.org/abs/2303.16634)** — Liu et al., 2023, "G-Eval:
  NLG Evaluation using GPT-4 with Better Human Alignment"
  - **Cited in the code/docs:** [`src`](src/prompts/summarization.go),
    [`src`](src/prompts/README.md)
  - **Inspired:** the judge-prompt shape: state the criteria, ask for a short
    chain of thought per dimension, then a numeric score (G-Eval's
    form-filling paradigm). The summarization judge is modeled on it
    explicitly, and the extraction and correction judges reuse the same shape
    (the `Judge` prompts in [`src`](prompts/), parsed by
    `parseJudgeScores` in [`src`](src/judge.go)). G-Eval's published prompt
    templates and its multi-sample scoring are not used.

- **[ChatIE](https://arxiv.org/abs/2302.10205)** — Wei et al., 2023, "ChatIE:
  Zero-Shot Information Extraction via Chatting with ChatGPT"
  - **Cited in the code/docs:** [`src`](src/prompts/extraction.go),
    [`src`](src/prompts/README.md)
  - **Inspired:** instruction-based extraction prompting. The output format
    is spelled out, the prompt says what to return when there is nothing to
    extract (ChatIE uses `[]` or "none"), and the model is told to output
    nothing else. Here that became one fixed JSON schema with empty arrays or
    `"unspecified"` instead of omitted keys (`extractionSpec` in
    [`src`](src/prompts/extraction.go)), which lets
    `ValidateExtraction` in [`src`](src/judge.go) check structure without a
    model call. ChatIE's two-stage, multi-turn pipeline and its prompt text
    are not used.

- **[GPT-NER](https://arxiv.org/abs/2304.10428)** — Wang et al., 2023,
  "GPT-NER: Named Entity Recognition via Large Language Models"
  - **Cited in the code/docs:** [`src`](src/prompts/extraction.go),
    named alongside ChatIE as an instruction-based extraction prompting
    convention.
  - **Inspired:** only that general convention, stating the task and the
    label set in the prompt. GPT-NER's prompt text, `@@…##` entity markup,
    kNN few-shot demonstrations, and self-verification pass are not used.

- **[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)**
  — Zheng et al., 2023
  - **Cited in the code/docs:** the "Judge model and self-grading" section of
    this README.
  - **Inspired:** its finding that LLM judges favor their own outputs
    (self-preference bias). The judge model is therefore excluded from the
    default candidate set, with a warning if `-models` includes it anyway
    (`runFrameworkBenchmarkPass` in [`src`](src/framework_pass.go)),
    and a sample graded by its own model is dropped when summarizing
    (`loadReports` in [`src`](src/summary.go)). Its MT-Bench judge prompts,
    `[[rating]]` verdict format, and FastChat judge code are not used.

- **[HIPE-OCRepair 2026](https://arxiv.org/abs/2607.08143)** — Ehrmann et al.,
  2026, "ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR
  Post-Correction for Historical Documents"
  - **Cited in the code/docs:** [`src`](src/prompts/correction.go),
    [`src`](src/prompts/README.md)
  - **Inspired:** the text_correction task contract. Post-correction is
    constrained rewriting: restore the intended wording and fix the surface
    form without paraphrasing or adding content. Applied here to ASR
    disfluencies instead of OCR errors, it is what the correction task
    prompt's rules and the judge's `faithfulness` dimension enforce
    (`correctionSpec` in [`src`](src/prompts/correction.go)).
    The competition's scorer (character match error rate, preference score)
    is not ported: `ScoreCorrection` in [`src`](src/correction.go) is this
    benchmark's own word-level edit-distance measure against the truth.
