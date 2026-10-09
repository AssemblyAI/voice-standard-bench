// SummarizationTask refactors the summarization benchmark (voicebench.go's
// runOne/runModel, driven by TaskGroup/GroupByTaskAndBucket for
// prompts.Summarization) into a Task[TranscriptRecord] that runs through
// Runner instead. No existing task file is touched -- this is a second,
// parallel path over the same data types and the same judge call, so its
// output can be compared directly against a run of the original -tasks
// summarization path, mirroring dictation_task.go's own relationship to
// dictation.go.
//
// What's reused verbatim: LoadTranscripts/SampleByBucket/Buckets
// (dataset.go) for loading and sampling, and judge.go's RunJudge (the free
// function) for scoring -- summarization has no answer key, so grading is a
// real LLM judge call exactly like the original benchmark already makes,
// not a reimplementation the way dictation's mechanical ScoreDictation call
// is. What's reused from voicebench.go/gateway.go: the same task-agnostic
// helpers dictation_task.go already reuses (contextLengthExceeded,
// estimatedTokens via maxCompletionTokens, recordFailure, RawAPIError) and
// dictation_task.go's own completionResultFromStream. What's duplicated
// rather than called: taskMaxTokens's summarization branch (voicebench.go)
// -- see maxCompletionTokens' doc comment, same rationale as
// dictation_task.go's own maxCompletionTokens.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

var _ Task[TranscriptRecord] = (*SummarizationTask)(nil)

// SummarizationTask implements Task[TranscriptRecord]. Construct with
// NewSummarizationTask rather than a bare struct literal, so its
// prompts.Spec is resolved once up front.
type SummarizationTask struct {
	gateway     *Gateway
	datasetGlob string
	spec        prompts.Spec
}

// NewSummarizationTask returns a SummarizationTask that calls gateway to run
// completions and judge calls. The task manages its own dataset location
// rather than taking one from the caller: every task's data lives under
// normalized/<task>, <task> being that task's own Name() -- here,
// normalized/summarization/aai_transcripts.jsonl, matching the filename
// dataset.go's LoadTranscripts expects. The glob is resolved
// under normalizedRoot. It resolves summarization's
// prompts.Spec once, here, the same reason NewDictationTask does.
func NewSummarizationTask(gateway *Gateway) (*SummarizationTask, error) {
	spec, ok := prompts.For(prompts.Summarization)
	if !ok {
		return nil, fmt.Errorf("no prompts.Spec registered for %q", prompts.Summarization)
	}

	datasetGlob := filepath.Join(normalizedRoot, prompts.Summarization, "aai_transcripts.jsonl")

	return &SummarizationTask{gateway: gateway, datasetGlob: datasetGlob, spec: spec}, nil
}

func (t *SummarizationTask) Name() string {
	return prompts.Summarization
}

// LoadDataset delegates to dataset.go's LoadTranscripts, which groups
// whatever it loads by task then bucket (see its doc comment) -- but since
// t.datasetGlob names exactly one file, under exactly one task's folder,
// byTaskBucket[t.Name()] is the only entry LoadTranscripts could ever
// populate; a glob matching nothing is already an error LoadTranscripts
// itself returns, so there is no "loaded fine but this task's key is
// missing" case left to guard against the way a multi-task glob would need
// to. This is flattened to the flat []TranscriptRecord Task[R] expects.
func (t *SummarizationTask) LoadDataset() ([]TranscriptRecord, error) {
	byTaskBucket, skipped, err := LoadTranscripts(t.datasetGlob)
	if err != nil {
		return nil, err
	}

	if skipped > 0 {
		log.Printf("summarization: skipped %d malformed or incomplete record(s)", skipped)
	}

	byBucket := byTaskBucket[t.Name()]

	var records []TranscriptRecord

	for _, bucket := range Buckets(byBucket) {
		records = append(records, byBucket[bucket]...)
	}

	return records, nil
}

// SampleDataset re-groups the flat records by bucket, delegates the actual
// per-bucket capping to dataset.go's SampleByBucket -- the same sampler the
// original pipeline uses -- and flattens the result back. SampleByBucket
// operates on the nested map/bucket shape the original pipeline kept
// records in throughout, which Task[R]'s flat-slice contract doesn't, so
// the grouping/flattening here is purely an adapter around it, not new
// sampling logic.
func (t *SummarizationTask) SampleDataset(records []TranscriptRecord, maxSamples int) []TranscriptRecord {
	byBucket := map[string][]TranscriptRecord{}
	for _, r := range records {
		byBucket[r.Bucket] = append(byBucket[r.Bucket], r)
	}

	sampled := SampleByBucket(byBucket, maxSamples)

	var out []TranscriptRecord

	for _, bucket := range Buckets(sampled) {
		out = append(out, sampled[bucket]...)
	}

	return out
}

// maxCompletionTokens sizes the completion budget for one summarization
// record: a fixed 15000 tokens, clamped to the model's own advertised max
// completion tokens when GET /v1/models reported one -- summaries are
// short, roughly fixed-size output regardless of transcript length, unlike
// text_correction/dictation's input-scaled budget. Mirrors the
// summarization branch (the default case) of the original benchmark's
// taskMaxTokens (voicebench.go) exactly; duplicated here rather than called
// for the same reason as dictation_task.go's maxCompletionTokens -- sizing
// the request is this task's own decision to make, not a switch statement
// shared code has to know summarization's name to reach.
func (t *SummarizationTask) maxCompletionTokens(model ModelInfo) int {
	want := 15000

	if model.MaxCompletionTokens > 0 && want > model.MaxCompletionTokens {
		want = model.MaxCompletionTokens
	}

	return want
}

// RunRecord mirrors runOne's request-shaping exactly (same system prompt,
// same input -- record.Text, since taskInput only diverts to MissedText for
// text_correction, never summarization -- same context-length guard, same
// budget) up through the completion call, stopping there: scoring is
// RunJudge's job (see its doc comment on the Task interface).
func (t *SummarizationTask) RunRecord(ctx context.Context, model ModelInfo, record TranscriptRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	if contextLengthExceeded(model, t.spec.Task, record.Text) {
		err := fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		sample.Err = err

		return sample, CompletionResult{}, err
	}

	result, err := t.gateway.RunCompletion(ctx, model, t.spec.Task, record.Text, t.maxCompletionTokens(model))
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

// RunTTFT is RunRecord's streaming counterpart. Unlike dictation, the
// original benchmark already ran a TTFT pass over summarization (ttft.go's
// runTTFTOne, generic across all three transcript tasks via taskInput/spec)
// -- this mirrors that path's shape for summarization specifically: same
// pre-flight guard and budget, RunCompletionStream instead of
// RunCompletion, never judged. completionResultFromStream is
// dictation_task.go's helper, reused as-is -- mapping a StreamResult onto
// CompletionResult's shape has nothing task-specific about it.
func (t *SummarizationTask) RunTTFT(ctx context.Context, model ModelInfo, record TranscriptRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	if contextLengthExceeded(model, t.spec.Task, record.Text) {
		err := fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		sample.Err = err

		return sample, CompletionResult{}, err
	}

	stream, err := t.gateway.RunCompletionStream(ctx, model, t.spec.Task, record.Text, t.maxCompletionTokens(model))
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

// RunJudge delegates to judge.go's RunJudge (the free function): unlike
// dictation, summarization has no answer key, so grading is a real LLM call
// against judgeModel, exactly as the original benchmark already does it via
// runOne. Mirrors runOne's judge-failure handling precisely: a failed judge
// call is logged and recorded to the FailureRecorder as FailureJudgeError,
// with the judge's own request/response recovered from the *JudgeError
// RunJudge returns (see its doc comment in judge.go) when there is one.
func (t *SummarizationTask) RunJudge(ctx context.Context, model ModelInfo, record TranscriptRecord, judgeModel string, result CompletionResult, failures *FailureRecorder) (Sample, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	scores, err := RunJudge(ctx, t.gateway, judgeModel, t.spec, record.Text, result.Text)
	if err != nil {
		log.Printf("model %s task %s record %s: judge failed: %v", model.ID, t.Name(), record.ID, err)

		if failures != nil {
			var judgeErr *JudgeError

			var request, response string

			if errors.As(err, &judgeErr) {
				request, response = judgeErr.Request, judgeErr.Response
			}

			recordFailure(failures, model.ID, model.ReasoningEffortLabel(), t.Name(), record.Bucket, record.ID, FailureJudgeError, err, request, response)
		}

		return sample, err
	}

	sample.Judge = scores

	return sample, nil
}

// DimensionRanges derives its entries from t.spec.Rubric rather than
// hardcoding the three dimension names a second time: every summarization
// dimension (faithfulness, coverage, conciseness) shares the exact same 1-5
// Likert scale, unlike dictation's heterogeneous mechanical scores (see
// dictation_task.go's DimensionRanges, which can't be derived this way).
func (t *SummarizationTask) DimensionRanges() map[string]scale {
	ranges := make(map[string]scale, len(t.spec.Rubric))

	for _, dim := range t.spec.Rubric {
		ranges[dim] = scale{lo: 1, hi: 5}
	}

	return ranges
}
