package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

var _ Task[LiveTranslationRecord] = (*LiveTranslationTask)(nil)

// AsyncBaselineModelID is the report row for async speech understanding
// translation of the same clips. It is not an LLM Gateway model: its output
// is read from the record (computed once by prep.py), so it costs nothing to
// run, its latency is zero, and it is scored exactly like every candidate.
// It is the bar a streaming prompt and model have to reach.
const AsyncBaselineModelID = "async-su-translation"

// LiveTranslationTask implements Task[LiveTranslationRecord].
type LiveTranslationTask struct {
	gateway     *Gateway
	datasetGlob string
	spec        prompts.Spec
}

func NewLiveTranslationTask(gateway *Gateway) (*LiveTranslationTask, error) {
	spec, ok := prompts.For(prompts.LiveTranslation)
	if !ok {
		return nil, fmt.Errorf("no prompts.Spec registered for %q", prompts.LiveTranslation)
	}

	return &LiveTranslationTask{
		gateway:     gateway,
		datasetGlob: filepath.Join(normalizedRoot, prompts.LiveTranslation, "*.jsonl"),
		spec:        spec,
	}, nil
}

// withAsyncBaseline returns models plus the async baseline row, unless it is
// already there. It copies rather than appending in place, so the other
// tasks' model lists are untouched.
func withAsyncBaseline(models []ModelInfo) []ModelInfo {
	if containsModel(models, AsyncBaselineModelID) {
		return models
	}

	out := make([]ModelInfo, 0, len(models)+1)
	out = append(out, models...)

	return append(out, ModelInfo{ID: AsyncBaselineModelID, Name: "Async speech understanding translation"})
}

func (t *LiveTranslationTask) Name() string { return prompts.LiveTranslation }

func (t *LiveTranslationTask) LoadDataset() ([]LiveTranslationRecord, error) {
	records, skipped, err := LoadLiveTranslationRecords(t.datasetGlob)
	if err != nil {
		return nil, err
	}

	if skipped > 0 {
		log.Printf("live_translation: skipped %d malformed or incomplete record(s)", skipped)
	}

	return records, nil
}

func (t *LiveTranslationTask) SampleDataset(records []LiveTranslationRecord, maxSamples int) []LiveTranslationRecord {
	return SampleLiveTranslationRecords(records, maxSamples)
}

// maxCompletionTokens gives a turn's translation room for about four times
// the turn's own token count, and never less than 400, the max_tokens the
// streaming translation docs use. A reasoning model that spends that budget
// thinking fails here as it would in production.
func (t *LiveTranslationTask) maxCompletionTokens(input string, model ModelInfo) int {
	want := estimatedTokens(input) * 4
	if want < 400 {
		want = 400
	}

	if model.MaxCompletionTokens > 0 && want > model.MaxCompletionTokens {
		want = model.MaxCompletionTokens
	}

	return want
}

func (t *LiveTranslationTask) sample(model ModelInfo, record LiveTranslationRecord) Sample {
	return Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}
}

func (t *LiveTranslationTask) RunRecord(ctx context.Context, model ModelInfo, record LiveTranslationRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := t.sample(model, record)

	if model.ID == AsyncBaselineModelID {
		return sample, CompletionResult{Text: record.AsyncBaseline}, nil
	}

	input := liveTranslationUserMessage(record)

	result, err := t.gateway.RunCompletion(ctx, model, t.spec.Task, input, t.maxCompletionTokens(input, model))
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

// RunTTFT measures time to first token per turn, which is most of the delay
// a viewer sees between a speaker finishing a turn and its translation
// appearing. The async baseline has no per-turn latency, so its TTFT sample
// is reported empty.
func (t *LiveTranslationTask) RunTTFT(ctx context.Context, model ModelInfo, record LiveTranslationRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := t.sample(model, record)

	if model.ID == AsyncBaselineModelID {
		return sample, CompletionResult{Text: record.AsyncBaseline}, nil
	}

	input := liveTranslationUserMessage(record)

	stream, err := t.gateway.RunCompletionStream(ctx, model, t.spec.Task, input, t.maxCompletionTokens(input, model))
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

// RunJudge always scores chrF against the reference, mechanically. When a
// judge model is set it also asks it for translation_accuracy and fluency;
// chrF rewards matching the reference's wording, and the judge catches a
// correct translation worded differently and a fluent one that changed the
// meaning.
func (t *LiveTranslationTask) RunJudge(ctx context.Context, model ModelInfo, record LiveTranslationRecord, judgeModel string, result CompletionResult, failures *FailureRecorder) (Sample, error) {
	sample := t.sample(model, record)
	sample.Judge = map[string]float64{"chrf": ChrF(result.Text, record.Reference)}

	if judgeModel == "" {
		return sample, nil
	}

	material := fmt.Sprintf("TARGET LANGUAGE: %s\n\nSOURCE:\n%s\n\nREFERENCE:\n%s", languageName(record.TargetLanguage), record.Turn, record.Reference)

	scores, err := RunJudge(ctx, t.gateway, judgeModel, t.spec, material, result.Text)
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

	for k, v := range scores {
		sample.Judge[k] = v
	}

	return sample, nil
}

func (t *LiveTranslationTask) DimensionRanges() map[string]scale {
	return map[string]scale{
		"chrf":                 {lo: 0, hi: 100},
		"translation_accuracy": {lo: 1, hi: 5},
		"fluency":              {lo: 1, hi: 5},
	}
}
