// Package prompts holds the two distinct kinds of prompt the benchmark uses,
// kept apart because they play opposite roles and must never be confused:
//
//   - The TASK prompt runs the thing under test. It is the system prompt sent
//     to a candidate model — one of everything listed at GET /v1/models — to
//     perform summarization, information extraction, or text correction on a
//     call transcript.
//   - The JUDGE prompt runs the grading. It is the system prompt sent to a
//     separate, fixed judge model (see -judge-model) to score the candidate's
//     output. It never runs against the models being benchmarked, and the
//     judge model is never itself one of the candidates.
//
// Every rubric here is reference-free by necessity: the AssemblyAI ASR
// transcripts this benchmark reads (see ../dataset.go) carry no gold summary,
// no annotated entities, and no corrected reference text — see
// voice_bench/jobs/summarization.py's aai_transcripts adapter. So the judge
// always grades the candidate's output against the SOURCE TRANSCRIPT, never
// against a fixed answer key. Each task file documents which paper or
// established convention its rubric follows.
package prompts

// Task names, shared across the prompt lookup, the CLI's -tasks flag, and
// report grouping.
const (
	Summarization         = "summarization"
	InformationExtraction = "information_extraction"
	TextCorrection        = "text_correction"
	// Dictation is graded entirely differently from the other three — see
	// dictationSpec's doc comment in dictation.go for why it breaks nearly
	// every rule this package's own README lays out for "adding a task".
	Dictation = "dictation"
)

// All is the full task set, in report order.
var All = []string{Summarization, InformationExtraction, TextCorrection, Dictation}

// Spec pairs one task's execution prompt with its judge prompt and names the
// score fields the judge is asked to fill in.
type Spec struct {
	// Task is the system prompt sent to the candidate model under test. The
	// transcript is sent as the user message alongside it.
	Task string
	// Judge is the system prompt sent to the fixed judge model. The caller
	// fills in the transcript and the candidate's output as the user message
	// (see ../judge.go).
	Judge string
	// Rubric names the numeric score fields the judge prompt's JSON response
	// carries, in report order. Every name here must appear as a key in the
	// JSON shape described in Judge.
	Rubric []string
	// Compare, when set, is the system prompt of a SECOND, reference-based
	// judge call: the caller sends the task's known-true text and the
	// candidate's output (see ../judge.go's RunReferenceJudge) and the judge
	// scores how similar the two are in actual meaning. Only a task with a real
	// answer key has one (today, text_correction). Judge stays reference-free
	// and never sees that key; Compare is the one prompt that does.
	Compare string
	// CompareRubric names the numeric score fields Compare's JSON response
	// carries, like Rubric does for Judge.
	CompareRubric []string
}

var specs = map[string]Spec{
	Summarization:         summarizationSpec,
	InformationExtraction: extractionSpec,
	TextCorrection:        correctionSpec,
	Dictation:             dictationSpec,
}

// For looks up a task's prompt spec. ok is false for an unrecognized task
// name, which callers should treat as a configuration error (e.g. a typo in
// -tasks), not a skippable condition.
func For(task string) (spec Spec, ok bool) {
	spec, ok = specs[task]
	return spec, ok
}
