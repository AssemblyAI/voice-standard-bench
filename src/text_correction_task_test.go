package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestTextCorrectionTask constructs a TextCorrectionTask for tests that
// don't need a real gateway or dataset.
func newTestTextCorrectionTask(t *testing.T, gw *Gateway) *TextCorrectionTask {
	t.Helper()

	task, err := NewTextCorrectionTask(gw)
	require.NoError(t, err)

	return task
}

// textCorrectionJudgeReply is what the judge sends back for a text_correction
// judge call: faithfulness and improvement.
func textCorrectionJudgeReply(faithfulness, improvement int) string {
	return fmt.Sprintf(`{"faithfulness": %d, "improvement": %d, "notes": "ok"}`, faithfulness, improvement)
}

func TestNewTextCorrectionTaskResolvesSpecAndName(t *testing.T) {
	task, err := NewTextCorrectionTask(nil)
	require.NoError(t, err)
	assert.Equal(t, prompts.TextCorrection, task.Name())
}

// ---------------------------------------------------------------------------
// LoadDataset / SampleDataset -- thin adapters around dataset.go, mirroring
// summarization_task_test.go's coverage.
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskLoadDataset(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "text_correction", "aai_transcripts.jsonl"),
		recordLineWithMissedText("r1", "30s", "the true text", "the corrupted text"))

	task, err := NewTextCorrectionTask(nil)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "the true text", records[0].Text)
	assert.Equal(t, "the corrupted text", records[0].MissedText)
	assert.Equal(t, "30s", records[0].Bucket)
}

func TestTextCorrectionTaskLoadDatasetOnlyReturnsTextCorrectionRecords(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "text_correction", "aai_transcripts.jsonl"),
		recordLineWithMissedText("c1", "30s", "true", "corrupted"))
	// A sibling task folder must never leak in, same as SummarizationTask.
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"), recordLine("s1", "30s", "summarization transcript"))

	task, err := NewTextCorrectionTask(nil)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 1, "summarization's sibling folder must never leak into text_correction's own, task-scoped glob")
	assert.Equal(t, "c1", records[0].ID)
}

func TestTextCorrectionTaskLoadDatasetPropagatesGlobError(t *testing.T) {
	t.Chdir(t.TempDir()) // empty -- normalized/text_correction/aai_transcripts.jsonl matches nothing

	task, err := NewTextCorrectionTask(nil)
	require.NoError(t, err)

	_, err = task.LoadDataset()
	assert.Error(t, err)
}

func TestTextCorrectionTaskSampleDatasetMatchesSampleByBucket(t *testing.T) {
	task := newTestTextCorrectionTask(t, nil)

	records := []TranscriptRecord{
		{ID: "c", Bucket: "30s"}, {ID: "a", Bucket: "30s"}, {ID: "b", Bucket: "30s"},
		{ID: "z", Bucket: "120s"},
	}

	got := task.SampleDataset(records, 2)

	byBucket := map[string][]TranscriptRecord{}
	for _, r := range records {
		byBucket[r.Bucket] = append(byBucket[r.Bucket], r)
	}

	sampled := SampleByBucket(byBucket, 2)

	var want []TranscriptRecord
	for _, bucket := range Buckets(sampled) {
		want = append(want, sampled[bucket]...)
	}

	assert.Equal(t, want, got)
}

// ---------------------------------------------------------------------------
// maxCompletionTokens -- must match the text_correction branch of the
// original benchmark's taskMaxTokens exactly (shared there with dictation's
// identical formula).
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskMaxCompletionTokensMatchesOriginalTaskMaxTokens(t *testing.T) {
	task := newTestTextCorrectionTask(t, nil)

	cases := []struct {
		name  string
		input string
		model ModelInfo
	}{
		{"short input, no model cap", "the corrupted text", ModelInfo{ID: "m"}},
		{"empty input", "", ModelInfo{ID: "m"}},
		{"long input, no model cap", strings.Repeat("word ", 2000), ModelInfo{ID: "m"}},
		{"model cap clamps a small budget", "the corrupted text", ModelInfo{ID: "m", MaxCompletionTokens: 100}},
		{"model cap does not clamp a budget already under it", strings.Repeat("word ", 2000), ModelInfo{ID: "m", MaxCompletionTokens: 1_000_000}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := taskMaxTokens(prompts.TextCorrection, tc.input, tc.model)
			got := task.maxCompletionTokens(tc.input, tc.model)
			assert.Equal(t, want, got)
		})
	}
}

// ---------------------------------------------------------------------------
// RunRecord / RunJudge
// ---------------------------------------------------------------------------

// The hidden original transcript (record.Text) is not used to score anything,
// so it must reach neither the candidate model nor the judge: the model is shown
// the corrupted transcript, and the judge grades the output against that same
// corrupted input.
func TestTextCorrectionTaskNeverSendsTheOriginalTranscriptAnywhere(t *testing.T) {
	const trueText = "TRUE-TEXT-MUST-NOT-LEAK the meeting is at three"
	const corrupted = "the meating is at three"
	const modelOutput = "the meeting is at four" // a plausible but imperfect correction

	var taskRequestBody string

	var judgeBodies []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			judgeBodies = append(judgeBodies, body)

			writeChatCompletion(w, textCorrectionJudgeReply(4, 3), 20, 10)

			return
		}

		taskRequestBody = body
		writeChatCompletion(w, modelOutput, 30, 8)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)
	model := ModelInfo{ID: "candidate-model"}
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: trueText, MissedText: corrupted}

	sample, result, err := task.RunRecord(context.Background(), model, record, nil)
	require.NoError(t, err)
	assert.NoError(t, sample.Err)

	judged, err := task.RunJudge(context.Background(), model, record, "the-judge", result, nil)
	require.NoError(t, err)

	require.NotEmpty(t, taskRequestBody, "the task call must have happened")
	require.Len(t, judgeBodies, 1, "text_correction is judged by exactly one call")

	assert.Contains(t, taskRequestBody, corrupted, "the model must be given the corrupted transcript")
	assert.NotContains(t, taskRequestBody, "TRUE-TEXT-MUST-NOT-LEAK", "the model must never see the true text")

	assert.Contains(t, judgeBodies[0], corrupted, "the judge grades against what the model was shown")
	assert.Contains(t, judgeBodies[0], modelOutput)
	assert.NotContains(t, judgeBodies[0], "TRUE-TEXT-MUST-NOT-LEAK", "the judge must never see the true text")
	assert.NotContains(t, judgeBodies[0], "REFERENCE:", "there is no reference-based comparison")

	assert.Equal(t, 4.0, judged.Judge["faithfulness"])
	assert.Equal(t, 3.0, judged.Judge["improvement"])
}

// A judged sample carries exactly the reference-free rubric -- nothing computed
// from the original transcript, however far the output is from it.
func TestTextCorrectionTaskRunJudgeRecordsOnlyTheReferenceFreeRubric(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(4, 3), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{
		ID: "rec-1", Bucket: "30s",
		Text:       "completely unrelated true text here",
		MissedText: "we agreed to shop the budjet review on friday",
	}
	result := CompletionResult{Text: "We agreed to ship the budget review on Friday."}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", result, nil)

	require.NoError(t, err)
	assert.Equal(t, map[string]float64{"faithfulness": 4, "improvement": 3}, judged.Judge)
}

// A judge call that fails leaves the sample unjudged and records the failure,
// with the judge's request and response so it can be replayed.
func TestTextCorrectionTaskRunJudgeFailureCarriesTheJudgesRequestAndResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "I would rather not grade these.", 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the meeting is at three", MissedText: "the meating is at three"}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: "the meeting is at three"}, failures)

	require.Error(t, err)
	assert.Nil(t, judged.Judge, "no partial rubric")

	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, FailureJudgeError, meta[0].Reason)

	output := readFailureOutput(t, failures.RunDir(), meta[0].OutputFile)
	assert.Contains(t, string(output.Request), "the meating is at three", "the saved request is the judge's, so the failure can be replayed")
	assert.Contains(t, string(output.Response), "I would rather not grade these.")
}

// TestTextCorrectionTaskRunRecordNoMissedTextIsAFailedSample: a record the
// corruption simulation didn't cover fails closed instead of silently
// correcting the true text against itself.
func TestTextCorrectionTaskRunRecordNoMissedTextIsAFailedSample(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the true text", MissedText: ""}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen at all when there's nothing to correct")
}

func TestTextCorrectionTaskRunRecordRecordsAPIErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: "corrupted"}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, failures)

	require.Error(t, err)
	require.Error(t, sample.Err)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.TextCorrection, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureAPIError, meta[0].Reason)
}

func TestTextCorrectionTaskRunRecordContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen once the context-length guard trips")
}

func TestTextCorrectionTaskRunJudgeRecordsJudgeErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "I refuse to grade this.", 10, 8)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: "corrupted"}

	_, err = task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: "a correction"}, failures)

	require.Error(t, err, "a judge reply with no JSON object must surface as an error")
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.TextCorrection, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureJudgeError, meta[0].Reason)
}

// ---------------------------------------------------------------------------
// RunTTFT -- the original benchmark already ran a TTFT pass over
// text_correction (ttft.go's runTTFTOne, generic across the three transcript
// tasks), so this is a real comparison, like summarization_task_test.go's.
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskRunTTFTMatchesRunTTFTOne(t *testing.T) {
	var bodies []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, readBody(t, r))

		w.Header().Set("Content-Type", "text/event-stream")
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"the meeting "},"finish_reason":null}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"is at three"},"finish_reason":"stop"}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the meeting is at three", MissedText: "the meating is at three"}

	spec, ok := prompts.For(prompts.TextCorrection)
	require.True(t, ok)

	// Old path.
	want := runTTFTOne(context.Background(), gw, model, prompts.TextCorrection, spec, record, nil)
	require.NoError(t, want.Err)

	// New path.
	task := newTestTextCorrectionTask(t, gw)
	sample, result, err := task.RunTTFT(context.Background(), model, record, nil)
	require.NoError(t, err)

	assert.Equal(t, want.Model, sample.Model)
	assert.Equal(t, want.Task, sample.Task)
	assert.Equal(t, want.Bucket, sample.Bucket)
	assert.Equal(t, want.RecordID, sample.RecordID)
	assert.Equal(t, "the meeting is at three", result.Text)
	assert.Nil(t, sample.Judge, "a TTFT sample is never judged")

	require.Len(t, bodies, 2, "both paths must make exactly one streaming call each")

	var oldReq, newReq chatWireRequest
	require.NoError(t, json.Unmarshal([]byte(bodies[0]), &oldReq))
	require.NoError(t, json.Unmarshal([]byte(bodies[1]), &newReq))

	assert.Equal(t, oldReq.MaxTokens, newReq.MaxTokens)
	assert.Equal(t, oldReq.Messages, newReq.Messages, "both paths must send MissedText, never Text")
}

func TestTextCorrectionTaskRunTTFTContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called)
}

func TestTextCorrectionTaskRunTTFTNoMissedTextSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the true text", MissedText: ""}

	sample, _, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model"}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called)
}

// ---------------------------------------------------------------------------
// DimensionRanges
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskDimensionRanges(t *testing.T) {
	task := newTestTextCorrectionTask(t, nil)

	spec, ok := prompts.For(prompts.TextCorrection)
	require.True(t, ok)

	ranges := task.DimensionRanges()
	require.Len(t, ranges, len(spec.Rubric), "the reference-free rubric and nothing else")

	for _, dim := range spec.Rubric {
		got, ok := ranges[dim]
		require.True(t, ok, "missing DimensionRanges entry for %q", dim)
		assert.Equal(t, scale{lo: 1, hi: 5}, got, "%q is an LLM judge's 1-5 score", dim)
	}

	// Scores older reports carry from comparing against the original transcript
	// are not quality dimensions, so they must not appear -- and must agree with
	// summary.go's dimensionRange, which is what actually decides what reaches
	// quality_score.
	for _, retired := range []string{"meaning_similarity", "truth_wer", "input_wer", "accuracy", "edit_ratio", "error_reduction"} {
		assert.NotContains(t, ranges, retired)
		assert.NotContains(t, dimensionRange, retired)
	}

	for dim, got := range ranges {
		assert.Equal(t, dimensionRange[dim], got, "%q must be scored on the same scale as summary.go's dimensionRange", dim)
	}
}
