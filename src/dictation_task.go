// DictationTask refactors the original dictation benchmark (dictation.go,
// runDictationForModel/runDictationOne) into a Task[DictationRecord] that
// runs through Runner instead of its own bespoke fan-out. dictation.go
// itself is left untouched -- this file is a second, parallel path over the
// exact same data types and mechanical scorer, so its output can be
// compared directly against a run of the original -tasks dictation path.
//
// What's reused verbatim from dictation.go, unchanged: DictationRecord,
// LoadDictationRecords, SampleDictationRecords, and ScoreDictation --
// dictation's schema, loading, and mechanical grading are not what this
// refactor is about. What's reused from voicebench.go/gateway.go: the
// genuinely task-agnostic helpers every task already shared
// (contextLengthExceeded, estimatedTokens, recordFailure, RawAPIError) --
// none of these branch on task name, so there is nothing task-specific to
// pull out of them. What is NOT reused: taskMaxTokens (voicebench.go), whose
// dictation branch is duplicated into maxCompletionTokens below instead of
// called -- sizing the completion budget is exactly the kind of per-task
// decision a Task should own itself rather than reach via a shared
// task-name switch (see maxCompletionTokens' doc comment).
package main

import (
	"context"
	"fmt"
	"log"
	"path/filepath"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

var _ Task[DictationRecord] = (*DictationTask)(nil)

// DictationTask implements Task[DictationRecord]. Construct with
// NewDictationTask rather than a bare struct literal, so its prompts.Spec is
// resolved once up front.
type DictationTask struct {
	gateway     *Gateway
	datasetGlob string
	spec        prompts.Spec
}

// NewDictationTask returns a DictationTask that calls gateway to run
// completions. The task manages its own dataset location rather than
// taking one from the caller: every task's data lives under
// normalized/<task>, <task> being that task's own Name() -- here,
// normalized/dictation/*.jsonl (a wildcard, not a fixed filename, since
// dictation's dataset file -- synthetic.jsonl -- is named differently from
// the other tasks' aai_transcripts.jsonl; see LoadDictationRecords). The
// glob is resolved under normalizedRoot. It
// resolves dictation's prompts.Spec once, here, rather than on every
// RunRecord/RunTTFT call -- failing fast at construction if
// prompts.dictationSpec is ever deregistered, rather than silently sending
// an empty system prompt to every record the way a lazy per-call lookup
// could (see runDictationForModel's doc comment in dictation.go for why the
// original benchmark treats this as "unreachable in practice" but still
// guards it).
func NewDictationTask(gateway *Gateway) (*DictationTask, error) {
	spec, ok := prompts.For(prompts.Dictation)
	if !ok {
		return nil, fmt.Errorf("no prompts.Spec registered for %q", prompts.Dictation)
	}

	datasetGlob := filepath.Join(normalizedRoot, prompts.Dictation, "*.jsonl")

	return &DictationTask{gateway: gateway, datasetGlob: datasetGlob, spec: spec}, nil
}

func (t *DictationTask) Name() string {
	return prompts.Dictation
}

// LoadDataset delegates entirely to dictation.go's own loader -- dictation's
// schema and filename both differ from the other tasks' aai_transcripts.jsonl
// (see LoadDictationRecords' doc comment), which is exactly why that loader
// already exists as its own function rather than a mode of LoadTranscripts.
func (t *DictationTask) LoadDataset() ([]DictationRecord, error) {
	records, skipped, err := LoadDictationRecords(t.datasetGlob)
	if err != nil {
		return nil, err
	}

	if skipped > 0 {
		log.Printf("dictation: skipped %d malformed or incomplete record(s)", skipped)
	}

	return records, nil
}

func (t *DictationTask) SampleDataset(records []DictationRecord, maxSamples int) []DictationRecord {
	return SampleDictationRecords(records, maxSamples)
}

// maxCompletionTokens sizes the completion budget for one dictation record.
// Dictation's output is roughly the same length as its input --
// reformatting, not compression -- so the budget scales with the input,
// plus 20% headroom and a fixed thinking-token allowance, clamped to the
// model's own advertised max completion tokens when GET /v1/models reported
// one. This mirrors the dictation branch of the original benchmark's
// taskMaxTokens (voicebench.go) exactly -- same formula, same clamp -- so
// the refactored task sizes requests identically to the original and their
// outputs stay directly comparable; it is duplicated here rather than
// called because deciding how big a completion budget dictation needs is
// this task's own business, not a switch statement shared code should have
// to know dictation's name to reach.
func (t *DictationTask) maxCompletionTokens(input string, model ModelInfo) int {
	want := estimatedTokens(input) + estimatedTokens(input)/5 + 64 + 15000 // input-sized, plus 20% headroom, plus thinking tokens

	if model.MaxCompletionTokens > 0 && want > model.MaxCompletionTokens {
		want = model.MaxCompletionTokens
	}

	return want
}

// RunRecord mirrors runDictationOne's request-shaping exactly (same system
// prompt, same input, same context-length guard, same budget) up through
// the completion call -- but stops there: unlike runDictationOne, which also
// scores the result inline, scoring is RunJudge's job here (see its doc
// comment on the Task interface), so a successful call returns the raw
// CompletionResult for RunJudge to grade instead.
func (t *DictationTask) RunRecord(ctx context.Context, model ModelInfo, record DictationRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	if contextLengthExceeded(model, t.spec.Task, record.Input) {
		err := fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		sample.Err = err

		return sample, CompletionResult{}, err
	}

	result, err := t.gateway.RunCompletion(ctx, model, t.spec.Task, record.Input, t.maxCompletionTokens(record.Input, model))
	if err != nil {
		sample.Err = err

		if failures != nil {
			recordFailure(failures, model.ID, model.ReasoningEffortLabel(), t.Name(), record.Bucket, record.ID, FailureAPIError, err, result.RawRequest, RawAPIError(err))
		}

		return sample, result, err
	}

	sample.TotalLatency = result.TotalLatency
	sample.TimeToFirstToken = result.TimeToFirstToken
	sample.CompletionTokens = result.CompletionTokens
	sample.PromptTokens = result.PromptTokens

	return sample, result, nil
}

// RunTTFT is RunRecord's streaming counterpart -- the original dictation
// benchmark had no TTFT path at all (dictation records were never routed
// through ttft.go's runTTFTBenchmarkPass), so this is new, not refactored,
// but mirrors runTTFTOne's shape (ttft.go) for the other three tasks: same
// pre-flight guard and budget, RunCompletionStream instead of RunCompletion,
// never judged.
func (t *DictationTask) RunTTFT(ctx context.Context, model ModelInfo, record DictationRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	if contextLengthExceeded(model, t.spec.Task, record.Input) {
		err := fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		sample.Err = err

		return sample, CompletionResult{}, err
	}

	stream, err := t.gateway.RunCompletionStream(ctx, model, t.spec.Task, record.Input, t.maxCompletionTokens(record.Input, model))
	if err != nil {
		sample.Err = err

		if failures != nil {
			recordFailure(failures, model.ID, model.ReasoningEffortLabel(), t.Name(), record.Bucket, record.ID, FailureAPIError, err, stream.RawRequest, RawAPIError(err))
		}

		return sample, completionResultFromStream(stream), err
	}

	sample.TotalLatency = stream.TotalLatency
	sample.TimeToFirstToken = stream.TimeToFirstToken
	sample.CompletionTokens = stream.CompletionTokens
	sample.PromptTokens = stream.PromptTokens

	return sample, completionResultFromStream(stream), nil
}

// RunJudge ignores judgeModel entirely: dictation carries a real answer key
// (record.TruthText), so grading is ScoreDictation's mechanical word-error-
// rate/exact-match/punctuation-capitalization-formatting comparison, not an
// LLM call -- identical to runDictationOne's own scoring step in
// dictation.go. Only Judge is populated; Model/Task/Bucket/RecordID are set
// for readability/debugging only -- a Runner merges just Judge (and
// SchemaInvalid) onto the Sample RunRecord already produced (see RunJudge's
// doc comment on the Task interface), so nothing else here is load-bearing.
func (t *DictationTask) RunJudge(_ context.Context, model ModelInfo, record DictationRecord, _ string, result CompletionResult, _ *FailureRecorder) (Sample, error) {
	return Sample{
		Model:           model.ID,
		Task:            t.Name(),
		Bucket:          record.Bucket,
		RecordID:        record.ID,
		Judge:           ScoreDictation(record.TruthText, result.Text, record.Metrics),
		ReasoningEffort: model.ReasoningEffortLabel(),
	}, nil
}

// DimensionRanges reports the raw [lo, hi] scale of every score name
// ScoreDictation can produce, mirroring summary.go's dimensionRange table
// for these same five names. word_error_rate's entry is the correct raw
// bound (0-100) but NOT direction-aware -- scale carries no polarity, and
// unlike the other four names here, a LOWER word_error_rate is better. The
// original benchmark's normalize() (summary.go) special-cases this exact
// name outside its own range table for the same reason; any future
// consumer of DimensionRanges doing similar normalization needs to invert
// this one the same way.
func (t *DictationTask) DimensionRanges() map[string]scale {
	return map[string]scale{
		"exact_match":                 {lo: 0, hi: 1},
		"word_error_rate":             {lo: 0, hi: 100},
		"punctuation_accuracy":        {lo: 0, hi: 100},
		"capitalization_accuracy":     {lo: 0, hi: 100},
		"formatting_command_accuracy": {lo: 0, hi: 100},
	}
}

// completionResultFromStream maps a streaming call's result onto
// CompletionResult's shape so RunTTFT can return the same type RunRecord
// does. RawResponse is left empty: a streamed reply is a sequence of SSE
// chunks, not one JSON response body, so there is no single verbatim
// document to carry the way a synchronous call's RawResponse does (see
// StreamResult's doc comment in gateway.go).
func completionResultFromStream(s StreamResult) CompletionResult {
	return CompletionResult{
		Text:             s.Text,
		RawRequest:       s.RawRequest,
		TotalLatency:     s.TotalLatency,
		TimeToFirstToken: s.TimeToFirstToken,
		PromptTokens:     s.PromptTokens,
		CompletionTokens: s.CompletionTokens,
	}
}
