// TextCorrectionTask refactors the text_correction benchmark (voicebench.go's
// runOne/runModel) into a Task[TranscriptRecord] that runs through Runner
// instead. No existing task file is touched -- this is a second, parallel
// path over the same data types and scoring, so its output can be compared
// directly against a run of the original -tasks text_correction path,
// mirroring summarization_task.go's and dictation_task.go's own relationship
// to their originals.
//
// Reused verbatim: LoadTranscripts/SampleByBucket/Buckets (dataset.go),
// judge.go's RunJudge, and voicebench.go's
// recordFailure/RawAPIError/contextLengthExceeded/
// estimatedTokens, plus dictation_task.go's completionResultFromStream.
// Duplicated rather than called: taskInput's and taskMaxTokens'
// text_correction branches (voicebench.go) -- see missedText and
// maxCompletionTokens below, same rationale as dictation_task.go's own
// maxCompletionTokens.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

var _ Task[TranscriptRecord] = (*TextCorrectionTask)(nil)

// TextCorrectionTask implements Task[TranscriptRecord]. Construct with
// NewTextCorrectionTask rather than a bare struct literal, so its
// prompts.Spec is resolved once up front.
type TextCorrectionTask struct {
	gateway     *Gateway
	datasetGlob string
	spec        prompts.Spec
}

// NewTextCorrectionTask returns a TextCorrectionTask calling gateway for
// completions and judge calls. Dataset location is fixed by convention --
// normalized/text_correction/aai_transcripts.jsonl, under normalizedRoot.
func NewTextCorrectionTask(gateway *Gateway) (*TextCorrectionTask, error) {
	spec, ok := prompts.For(prompts.TextCorrection)
	if !ok {
		return nil, fmt.Errorf("no prompts.Spec registered for %q", prompts.TextCorrection)
	}

	datasetGlob := filepath.Join(normalizedRoot, prompts.TextCorrection, "aai_transcripts.jsonl")

	return &TextCorrectionTask{gateway: gateway, datasetGlob: datasetGlob, spec: spec}, nil
}

func (t *TextCorrectionTask) Name() string {
	return prompts.TextCorrection
}

// LoadDataset mirrors SummarizationTask.LoadDataset: t.datasetGlob names
// exactly one file under exactly one task's folder, so byTaskBucket[t.Name()]
// is the only entry LoadTranscripts could populate; flattened to the flat
// []TranscriptRecord Task[R] expects.
func (t *TextCorrectionTask) LoadDataset() ([]TranscriptRecord, error) {
	byTaskBucket, skipped, err := LoadTranscripts(t.datasetGlob)
	if err != nil {
		return nil, err
	}

	if skipped > 0 {
		log.Printf("text_correction: skipped %d malformed or incomplete record(s)", skipped)
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
func (t *TextCorrectionTask) SampleDataset(records []TranscriptRecord, maxSamples int) []TranscriptRecord {
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

// errNoMissedText mirrors taskInput's text_correction error verbatim: a
// record the corruption simulation didn't cover has nothing to correct, and
// must fail closed rather than falling back to record.Text -- which would
// hand the model the answer key and score a trivially perfect accuracy.
var errNoMissedText = errors.New("record has no missed_text to correct")

// missedText returns record.MissedText -- the simulated corrupted transcript
// the model corrects, never record.Text, the hidden answer key (see
// ScoreCorrection in correction.go). Mirrors taskInput's text_correction branch
// (voicebench.go); duplicated rather than called since taskInput is a
// task-name switch and this task already knows which task it is.
func (t *TextCorrectionTask) missedText(record TranscriptRecord) (string, error) {
	if record.MissedText == "" {
		return "", errNoMissedText
	}

	return record.MissedText, nil
}

// maxCompletionTokens sizes the completion budget: input-sized (a correction
// is roughly as long as its input -- cleanup, not compression) plus 20%
// headroom and a fixed thinking-token allowance, clamped to the model's own
// advertised max. Mirrors taskMaxTokens' text_correction branch
// (voicebench.go, shared there with dictation's identical formula) exactly;
// duplicated here for the same reason as dictation_task.go's
// maxCompletionTokens -- sizing the request is this task's own call.
func (t *TextCorrectionTask) maxCompletionTokens(input string, model ModelInfo) int {
	want := estimatedTokens(input) + estimatedTokens(input)/5 + 64 + 40000 // input-sized, plus 20% headroom, plus thinking tokens

	if model.MaxCompletionTokens > 0 && want > model.MaxCompletionTokens {
		want = model.MaxCompletionTokens
	}

	return want
}

// RunRecord mirrors runOne's request-shaping for text_correction: the
// corrupted MissedText as input (never Text), same context-length guard,
// same input-scaled budget -- stopping short of scoring, which is RunJudge's
// job (see the Task interface).
func (t *TextCorrectionTask) RunRecord(ctx context.Context, model ModelInfo, record TranscriptRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	input, err := t.missedText(record)
	if err != nil {
		sample.Err = err

		return sample, CompletionResult{}, err
	}

	if contextLengthExceeded(model, t.spec.Task, input) {
		err := fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		sample.Err = err

		return sample, CompletionResult{}, err
	}

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

// RunTTFT is RunRecord's streaming counterpart, mirroring runTTFTOne's shape
// for text_correction (ttft.go already runs this task through its
// taskInput/spec-generic path): same guard and budget, RunCompletionStream
// instead of RunCompletion, never judged.
func (t *TextCorrectionTask) RunTTFT(ctx context.Context, model ModelInfo, record TranscriptRecord, failures *FailureRecorder) (Sample, CompletionResult, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	input, err := t.missedText(record)
	if err != nil {
		sample.Err = err

		return sample, CompletionResult{}, err
	}

	if contextLengthExceeded(model, t.spec.Task, input) {
		err := fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		sample.Err = err

		return sample, CompletionResult{}, err
	}

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

// RunJudge scores a completion with one reference-free LLM judge call, graded
// against the same corrupted input the model saw: faithfulness and
// improvement. record.Text -- the dataset's original transcript -- is not used
// to score anything: it turned out not to be an accurate answer key, so a
// comparison with it (a word-level distance or a second, reference-based judge)
// would measure the truth's own errors as much as the model's. The judge's
// two dimensions are the whole quality score.
//
// record.MissedText is used directly, not re-validated via missedText: RunJudge
// runs only after a successful RunRecord for the same record (see the Task
// interface), which already proved it's non-empty.
func (t *TextCorrectionTask) RunJudge(ctx context.Context, model ModelInfo, record TranscriptRecord, judgeModel string, result CompletionResult, failures *FailureRecorder) (Sample, error) {
	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	scores, err := RunJudge(ctx, t.gateway, judgeModel, t.spec, record.MissedText, result.Text)
	if err != nil {
		return sample, t.judgeFailed(model, record, failures, err)
	}

	sample.Judge = scores

	return sample, nil
}

// judgeFailed logs a failed judge call and records it as FailureJudgeError --
// with the judge's own request and response when err is a *JudgeError -- then
// hands err back for RunJudge to return.
func (t *TextCorrectionTask) judgeFailed(model ModelInfo, record TranscriptRecord, failures *FailureRecorder, err error) error {
	log.Printf("model %s task %s record %s: judge failed: %v", model.ID, t.Name(), record.ID, err)

	if failures != nil {
		var judgeErr *JudgeError

		var request, response string

		if errors.As(err, &judgeErr) {
			request, response = judgeErr.Request, judgeErr.Response
		}

		recordFailure(failures, model.ID, model.ReasoningEffortLabel(), t.Name(), record.Bucket, record.ID, FailureJudgeError, err, request, response)
	}

	return err
}

// DimensionRanges derives faithfulness/improvement from t.spec.Rubric: every
// quality dimension here is an LLM judge's 1-5 Likert score (like
// SummarizationTask.DimensionRanges).
func (t *TextCorrectionTask) DimensionRanges() map[string]scale {
	ranges := make(map[string]scale, len(t.spec.Rubric))

	for _, dim := range t.spec.Rubric {
		ranges[dim] = scale{lo: 1, hi: 5}
	}

	return ranges
}
