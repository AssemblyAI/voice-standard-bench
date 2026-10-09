// InformationExtractionTask refactors the information_extraction benchmark
// (voicebench.go's runOne/runModel) into a Task[TranscriptRecord] that runs
// through Runner instead. No existing task file is touched -- this is a
// second, parallel path over the same data types and scoring, so its output
// can be compared directly against a run of the original -tasks
// information_extraction path, mirroring the other three tasks' own
// relationship to voicebench.go/dictation.go.
//
// Reused verbatim: LoadTranscripts/SampleByBucket/Buckets (dataset.go),
// judge.go's RunJudge/ValidateExtraction, and voicebench.go's
// recordFailure/worstRubric/RawAPIError/contextLengthExceeded/
// estimatedTokens. Duplicated rather than called: taskMaxTokens'
// information_extraction branch (voicebench.go) -- same rationale as
// summarization_task.go's maxCompletionTokens.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

var _ Task[TranscriptRecord] = (*InformationExtractionTask)(nil)

// InformationExtractionTask implements Task[TranscriptRecord]. Construct
// with NewInformationExtractionTask rather than a bare struct literal, so
// its prompts.Spec is resolved once up front.
type InformationExtractionTask struct {
	gateway     *Gateway
	datasetGlob string
	spec        prompts.Spec
}

// NewInformationExtractionTask returns an InformationExtractionTask calling
// gateway for completions and judge calls. Dataset location is fixed by
// convention -- normalized/information_extraction/aai_transcripts.jsonl --
// under normalizedRoot.
func NewInformationExtractionTask(gateway *Gateway) (*InformationExtractionTask, error) {
	spec, ok := prompts.For(prompts.InformationExtraction)
	if !ok {
		return nil, fmt.Errorf("no prompts.Spec registered for %q", prompts.InformationExtraction)
	}

	datasetGlob := filepath.Join(normalizedRoot, prompts.InformationExtraction, "aai_transcripts.jsonl")

	return &InformationExtractionTask{gateway: gateway, datasetGlob: datasetGlob, spec: spec}, nil
}

func (t *InformationExtractionTask) Name() string {
	return prompts.InformationExtraction
}

// LoadDataset mirrors SummarizationTask.LoadDataset: t.datasetGlob names
// exactly one file under exactly one task's folder, so byTaskBucket[t.Name()]
// is the only entry LoadTranscripts could populate; flattened to the flat
// []TranscriptRecord Task[R] expects.
func (t *InformationExtractionTask) LoadDataset() ([]TranscriptRecord, error) {
	byTaskBucket, skipped, err := LoadTranscripts(t.datasetGlob)
	if err != nil {
		return nil, err
	}

	if skipped > 0 {
		log.Printf("information_extraction: skipped %d malformed or incomplete record(s)", skipped)
	}

	byBucket := byTaskBucket[t.Name()]

	var records []TranscriptRecord

	for _, bucket := range Buckets(byBucket) {
		records = append(records, byBucket[bucket]...)
	}

	return records, nil
}

// SampleDataset re-groups by bucket, delegates capping to SampleByBucket,
// and flattens back -- mirrors SummarizationTask.SampleDataset exactly.
func (t *InformationExtractionTask) SampleDataset(records []TranscriptRecord, maxSamples int) []TranscriptRecord {
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

// maxCompletionTokens mirrors taskMaxTokens' information_extraction branch
// exactly: a small fixed budget regardless of transcript length -- output
// size is bounded by the schema, not the input, unlike text_correction/
// dictation's input-scaled budget. Duplicated here rather than called for
// the same reason as summarization_task.go's maxCompletionTokens.
func (t *InformationExtractionTask) maxCompletionTokens(model ModelInfo) int {
	want := 15000

	if model.MaxCompletionTokens > 0 && want > model.MaxCompletionTokens {
		want = model.MaxCompletionTokens
	}

	return want
}

// RunRecord mirrors runOne's request-shaping exactly (record.Text as input,
// same context-length guard, same fixed budget) up through the completion
// call, stopping there: the schema check and scoring are RunJudge's job (see
// its doc comment on the Task interface).
func (t *InformationExtractionTask) RunRecord(ctx context.Context, model ModelInfo, record TranscriptRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
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

// RunTTFT is RunRecord's streaming counterpart. Like summarization and
// text_correction, the original benchmark already ran a TTFT pass over
// information_extraction (ttft.go's runTTFTOne, generic across all three
// transcript tasks): same guard and budget, RunCompletionStream instead of
// RunCompletion, never judged (and so never schema-checked either).
func (t *InformationExtractionTask) RunTTFT(ctx context.Context, model ModelInfo, record TranscriptRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
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

// RunJudge mirrors runOne's information_extraction branch: ValidateExtraction
// runs FIRST, mechanically, with no model call. A structurally invalid
// response skips the LLM judge entirely and scores the worst possible rubric
// value (worstRubric, voicebench.go) -- grading the CONTENT of something
// that isn't even shaped right would be meaningless. Only a schema-valid
// response reaches the real, reference-free judge call.
func (t *InformationExtractionTask) RunJudge(ctx context.Context, model ModelInfo, record TranscriptRecord, judgeModel string, result CompletionResult, failures *FailureRecorder) (Sample, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	if _, err := ValidateExtraction(result.Text); err != nil {
		sample.SchemaInvalid = true
		sample.Judge = worstRubric(t.spec.Rubric)

		if failures != nil {
			recordFailure(failures, model.ID, model.ReasoningEffortLabel(), t.Name(), record.Bucket, record.ID, FailureInvalidSchema, err, result.RawRequest, result.RawResponse)
		}

		return sample, nil
	}

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

// DimensionRanges derives groundedness/completeness from t.spec.Rubric --
// both a 1-5 Likert scale, the same shape as SummarizationTask.DimensionRanges.
func (t *InformationExtractionTask) DimensionRanges() map[string]scale {
	ranges := make(map[string]scale, len(t.spec.Rubric))

	for _, dim := range t.spec.Rubric {
		ranges[dim] = scale{lo: 1, hi: 5}
	}

	return ranges
}
