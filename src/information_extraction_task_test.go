package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// validExtraction and malformedExtraction are reused across this file's
// RunJudge tests -- shapes match judge_test.go's TestValidateExtraction.
const (
	validExtraction = `{"participants": ["Alice", "Bob"], "action_items": [{"owner": "Alice", "task": "send the proposal", "due": "Friday"}], ` +
		`"decisions": ["use Postgres"], "entities": {"people": [], "organizations": ["Acme"], "products": [], "dates": ["Friday"]}}`
	malformedExtraction = `{"participants": ["only one key, missing everything else"]}`
)

// newTestInformationExtractionTask constructs an InformationExtractionTask
// for tests that don't need a real gateway or dataset.
func newTestInformationExtractionTask(t *testing.T, gw *Gateway) *InformationExtractionTask {
	t.Helper()

	task, err := NewInformationExtractionTask(gw)
	require.NoError(t, err)

	return task
}

func TestNewInformationExtractionTaskResolvesSpecAndName(t *testing.T) {
	task, err := NewInformationExtractionTask(nil)
	require.NoError(t, err)
	assert.Equal(t, prompts.InformationExtraction, task.Name())
}

// ---------------------------------------------------------------------------
// LoadDataset / SampleDataset -- thin adapters around dataset.go, mirroring
// summarization_task_test.go's coverage.
// ---------------------------------------------------------------------------

func TestInformationExtractionTaskLoadDataset(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "information_extraction", "aai_transcripts.jsonl"),
		recordLine("r1", "30s", "hello world")+recordLine("r2", "120s", "a longer call"))

	task, err := NewInformationExtractionTask(nil)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 2)

	byID := map[string]TranscriptRecord{}
	for _, r := range records {
		byID[r.ID] = r
	}

	assert.Equal(t, "hello world", byID["r1"].Text)
	assert.Equal(t, "30s", byID["r1"].Bucket)
}

func TestInformationExtractionTaskLoadDatasetOnlyReturnsInformationExtractionRecords(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "information_extraction", "aai_transcripts.jsonl"), recordLine("e1", "30s", "extraction transcript"))
	// A sibling task folder must never leak in, same as SummarizationTask.
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"), recordLine("s1", "30s", "summarization transcript"))

	task, err := NewInformationExtractionTask(nil)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 1, "summarization's sibling folder must never leak into information_extraction's own, task-scoped glob")
	assert.Equal(t, "e1", records[0].ID)
}

func TestInformationExtractionTaskLoadDatasetPropagatesGlobError(t *testing.T) {
	t.Chdir(t.TempDir()) // empty -- normalized/information_extraction/aai_transcripts.jsonl matches nothing

	task, err := NewInformationExtractionTask(nil)
	require.NoError(t, err)

	_, err = task.LoadDataset()
	assert.Error(t, err)
}

func TestInformationExtractionTaskSampleDatasetMatchesSampleByBucket(t *testing.T) {
	task := newTestInformationExtractionTask(t, nil)

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
// maxCompletionTokens -- must match the information_extraction branch of the
// original benchmark's taskMaxTokens exactly.
// ---------------------------------------------------------------------------

func TestInformationExtractionTaskMaxCompletionTokensMatchesOriginalTaskMaxTokens(t *testing.T) {
	task := newTestInformationExtractionTask(t, nil)

	cases := []struct {
		name  string
		model ModelInfo
	}{
		{"no model cap", ModelInfo{ID: "m"}},
		{"model cap below the fixed budget clamps it", ModelInfo{ID: "m", MaxCompletionTokens: 100}},
		{"model cap above the fixed budget does not clamp it", ModelInfo{ID: "m", MaxCompletionTokens: 1_000_000}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// transcript length must not matter (fixed budget) -- passed only
			// because taskMaxTokens' signature requires one.
			for _, transcript := range []string{"short", strings.Repeat("word ", 5000)} {
				want := taskMaxTokens(prompts.InformationExtraction, transcript, tc.model)
				got := task.maxCompletionTokens(tc.model)
				assert.Equal(t, want, got)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// RunRecord / RunJudge -- both the schema-valid (real judge) and
// schema-invalid (mechanical, worstRubric) paths.
// ---------------------------------------------------------------------------

// TestInformationExtractionTaskRunJudgeSkipsJudgeCallOnInvalidSchema locks in
// that a structurally invalid completion scores mechanically (worstRubric,
// SchemaInvalid) and never reaches the judge.
func TestInformationExtractionTaskRunJudgeSkipsJudgeCallOnInvalidSchema(t *testing.T) {
	judgeCalls := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(readBody(t, r), `"model":"the-judge"`) {
			judgeCalls++
			writeChatCompletion(w, `{"groundedness": 5, "completeness": 5, "notes": "unexpected"}`, 20, 10)

			return
		}

		writeChatCompletion(w, malformedExtraction, 10, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}

	task := newTestInformationExtractionTask(t, gw)
	_, result, err := task.RunRecord(context.Background(), model, record, nil)
	require.NoError(t, err)

	judged, err := task.RunJudge(context.Background(), model, record, "the-judge", result, nil)
	require.NoError(t, err, "a schema-invalid sample is a successful mechanical score, not a RunJudge error")

	assert.True(t, judged.SchemaInvalid)
	assert.Equal(t, map[string]float64{"groundedness": 1, "completeness": 1}, judged.Judge)
	assert.Equal(t, 0, judgeCalls, "the judge must never be called for structurally invalid output")
}

func TestInformationExtractionTaskRunRecordRecordsAPIErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestInformationExtractionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, failures)

	require.Error(t, err)
	require.Error(t, sample.Err)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.InformationExtraction, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureAPIError, meta[0].Reason)
}

func TestInformationExtractionTaskRunRecordContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestInformationExtractionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen once the context-length guard trips")
}

// TestInformationExtractionTaskRunJudgeRecordsInvalidSchemaFailure locks in
// that the VERBATIM gateway response (message content AND usage), not just
// the malformed text extracted from it, must be saved.
func TestInformationExtractionTaskRunJudgeRecordsInvalidSchemaFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, malformedExtraction, 10, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestInformationExtractionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}

	sample, result, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, nil)
	require.NoError(t, err)
	assert.NoError(t, sample.Err)

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "the-judge", result, failures)
	require.NoError(t, err)

	assert.True(t, judged.SchemaInvalid)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, FailureInvalidSchema, meta[0].Reason)
	assert.NotEmpty(t, meta[0].Error)

	outputBytes, err := os.ReadFile(filepath.Join(failures.RunDir(), meta[0].OutputFile))
	require.NoError(t, err)

	var output failureOutput
	require.NoError(t, json.Unmarshal(outputBytes, &output))

	var envelope struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage map[string]any `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(output.Response, &envelope),
		"Response must be the verbatim ChatCompletion body, not the bare candidate text")

	require.Len(t, envelope.Choices, 1)
	assert.Equal(t, malformedExtraction, envelope.Choices[0].Message.Content)
	assert.NotEmpty(t, envelope.Usage)
	require.NotEmpty(t, output.Request)
}

func TestInformationExtractionTaskRunJudgeRecordsJudgeErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "I refuse to grade this.", 10, 8)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestInformationExtractionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}

	_, err = task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: validExtraction}, failures)

	require.Error(t, err, "a judge reply with no JSON object must surface as an error")
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.InformationExtraction, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureJudgeError, meta[0].Reason)
}

func TestInformationExtractionTaskRunJudgeSuccessDoesNotRecordAFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(readBody(t, r), `"model":"the-judge"`) {
			writeChatCompletion(w, `{"groundedness": 4, "completeness": 4, "notes": "ok"}`, 10, 10)
			return
		}

		writeChatCompletion(w, validExtraction, 10, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestInformationExtractionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}

	sample, result, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, failures)
	require.NoError(t, err)

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "the-judge", result, failures)
	require.NoError(t, err)

	assert.NoError(t, sample.Err)
	assert.False(t, judged.SchemaInvalid)
	assert.Equal(t, 0, failures.Count())
}

// ---------------------------------------------------------------------------
// RunTTFT -- the original benchmark already ran a TTFT pass over
// information_extraction (ttft.go's runTTFTOne, generic across the three
// transcript tasks), so this is a real comparison.
// ---------------------------------------------------------------------------

func TestInformationExtractionTaskRunTTFTMatchesRunTTFTOne(t *testing.T) {
	var bodies []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, readBody(t, r))

		w.Header().Set("Content-Type", "text/event-stream")
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"{\"participants\":"},"finish_reason":null}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"[]}"},"finish_reason":"stop"}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the source transcript"}

	spec, ok := prompts.For(prompts.InformationExtraction)
	require.True(t, ok)

	// Old path.
	want := runTTFTOne(context.Background(), gw, model, prompts.InformationExtraction, spec, record, nil)
	require.NoError(t, want.Err)

	// New path.
	task := newTestInformationExtractionTask(t, gw)
	sample, result, err := task.RunTTFT(context.Background(), model, record, nil)
	require.NoError(t, err)

	assert.Equal(t, want.Model, sample.Model)
	assert.Equal(t, want.Task, sample.Task)
	assert.Equal(t, want.Bucket, sample.Bucket)
	assert.Equal(t, want.RecordID, sample.RecordID)
	assert.Equal(t, `{"participants":[]}`, result.Text)
	assert.Nil(t, sample.Judge, "a TTFT sample is never judged")
	assert.False(t, sample.SchemaInvalid, "a TTFT sample is never schema-checked either")

	require.Len(t, bodies, 2, "both paths must make exactly one streaming call each")

	var oldReq, newReq chatWireRequest
	require.NoError(t, json.Unmarshal([]byte(bodies[0]), &oldReq))
	require.NoError(t, json.Unmarshal([]byte(bodies[1]), &newReq))

	assert.Equal(t, oldReq.MaxTokens, newReq.MaxTokens)
	assert.Equal(t, oldReq.Messages, newReq.Messages)
}

func TestInformationExtractionTaskRunTTFTContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestInformationExtractionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called)
}

// ---------------------------------------------------------------------------
// DimensionRanges
// ---------------------------------------------------------------------------

func TestInformationExtractionTaskDimensionRangesMatchesRubric(t *testing.T) {
	task := newTestInformationExtractionTask(t, nil)

	spec, ok := prompts.For(prompts.InformationExtraction)
	require.True(t, ok)

	ranges := task.DimensionRanges()
	require.Len(t, ranges, len(spec.Rubric))

	for _, dim := range spec.Rubric {
		got, ok := ranges[dim]
		require.True(t, ok, "missing DimensionRanges entry for %q", dim)
		assert.Equal(t, scale{lo: 1, hi: 5}, got)
	}
}
