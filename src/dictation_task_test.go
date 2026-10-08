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

// newTestDictationTask constructs a DictationTask for tests that don't need
// a real gateway or dataset (construction only resolves prompts.Spec -- see
// NewDictationTask's doc comment).
func newTestDictationTask(t *testing.T, gw *Gateway) *DictationTask {
	t.Helper()

	task, err := NewDictationTask(gw)
	require.NoError(t, err)

	return task
}

func TestNewDictationTaskResolvesSpecAndName(t *testing.T) {
	task, err := NewDictationTask(nil)
	require.NoError(t, err)
	assert.Equal(t, prompts.Dictation, task.Name())
}

// ---------------------------------------------------------------------------
// LoadDataset / SampleDataset -- thin delegation to dictation.go's own
// loader/sampler, so these just confirm the delegation actually happens.
// ---------------------------------------------------------------------------

func TestDictationTaskLoadDataset(t *testing.T) {
	// NewDictationTask's dataset location is a fixed, task-owned convention
	// (normalized/dictation/*.jsonl, relative to the working directory) --
	// not a constructor parameter a test can point elsewhere -- so this
	// Chdirs into a fresh temp directory and lays the fixture out under that
	// same relative path, exactly like a real invocation would from the
	// repo root.
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "dictation", "synthetic.jsonl"),
		dictationRecordLine("r1", "30s", "unused", "hey comma world period", "Hey, world.", []string{"exact_match"}))

	task, err := NewDictationTask(nil)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "r1", records[0].ID)
	assert.Equal(t, "Hey, world.", records[0].TruthText)
}

func TestDictationTaskLoadDatasetPropagatesGlobError(t *testing.T) {
	t.Chdir(t.TempDir()) // empty -- normalized/dictation/*.jsonl matches nothing

	task, err := NewDictationTask(nil)
	require.NoError(t, err)

	_, err = task.LoadDataset()
	assert.Error(t, err)
}

func TestDictationTaskSampleDatasetMatchesSampleDictationRecords(t *testing.T) {
	task := newTestDictationTask(t, nil)

	records := []DictationRecord{
		{ID: "c", Bucket: "30s"}, {ID: "a", Bucket: "30s"}, {ID: "b", Bucket: "30s"},
		{ID: "z", Bucket: "120s"},
	}

	assert.Equal(t, SampleDictationRecords(records, 2), task.SampleDataset(records, 2))
}

// ---------------------------------------------------------------------------
// maxCompletionTokens -- must match the dictation branch of the original
// benchmark's taskMaxTokens exactly (see maxCompletionTokens' doc comment):
// this is the piece the refactor moved from a shared task-name switch into
// the task itself, so proving the two formulas agree is the whole point.
// ---------------------------------------------------------------------------

func TestDictationTaskMaxCompletionTokensMatchesOriginalTaskMaxTokens(t *testing.T) {
	task := newTestDictationTask(t, nil)

	cases := []struct {
		name  string
		input string
		model ModelInfo
	}{
		{"short input, no model cap", "hey comma world period", ModelInfo{ID: "m"}},
		{"empty input", "", ModelInfo{ID: "m"}},
		{"long input, no model cap", strings.Repeat("word ", 2000), ModelInfo{ID: "m"}},
		{"model cap clamps a small budget", "hey comma world period", ModelInfo{ID: "m", MaxCompletionTokens: 100}},
		{"model cap does not clamp a budget already under it", strings.Repeat("word ", 2000), ModelInfo{ID: "m", MaxCompletionTokens: 1_000_000}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := taskMaxTokens(prompts.Dictation, tc.input, tc.model)
			got := task.maxCompletionTokens(tc.input, tc.model)
			assert.Equal(t, want, got)
		})
	}
}

// ---------------------------------------------------------------------------
// RunRecord / RunJudge -- direct comparisons against the original
// runDictationOne, proving the refactor sends the same request and scores
// the result identically.
// ---------------------------------------------------------------------------

// chatWireRequest is the slice of a chat-completion request body this
// package's task tests check: which model, the system/user messages, and
// max_tokens -- enough to prove the wire request (including sizing) is
// identical between an old and new path, without decoding the whole SDK
// request type. Shared across task test files (not just dictation's) --
// every Task's RunRecord/RunJudge sends this same request shape.
type chatWireRequest struct {
	Model     string `json:"model"`
	MaxTokens int    `json:"max_tokens"`
	Messages  []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
}

func TestDictationTaskRunRecordSendsIdenticalRequestToRunDictationOne(t *testing.T) {
	var bodies []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, readBody(t, r))
		writeChatCompletion(w, "Hey, world.", 10, 5)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}
	record := DictationRecord{ID: "rec-1", Bucket: "30s", Input: "hey comma world period", TruthText: "Hey, world.", Metrics: []string{"exact_match"}}

	spec, ok := prompts.For(prompts.Dictation)
	require.True(t, ok)

	// Old path.
	oldSample := runDictationOne(context.Background(), gw, model, spec, record, nil)
	require.NoError(t, oldSample.Err)

	// New path.
	task := newTestDictationTask(t, gw)
	newSample, _, err := task.RunRecord(context.Background(), model, record, nil)
	require.NoError(t, err)

	require.Len(t, bodies, 2, "both paths must make exactly one completion call each")

	var oldReq, newReq chatWireRequest
	require.NoError(t, json.Unmarshal([]byte(bodies[0]), &oldReq))
	require.NoError(t, json.Unmarshal([]byte(bodies[1]), &newReq))

	assert.Equal(t, oldReq.Model, newReq.Model)
	assert.Equal(t, oldReq.MaxTokens, newReq.MaxTokens, "max_tokens sizing must match, since maxCompletionTokens duplicates taskMaxTokens' dictation formula")
	assert.Equal(t, oldReq.Messages, newReq.Messages, "system prompt and user message must be identical")

	assert.Equal(t, oldSample.Bucket, newSample.Bucket)
	assert.Equal(t, oldSample.RecordID, newSample.RecordID)
}

func TestDictationTaskRunRecordAndRunJudgeMatchRunDictationOneScoring(t *testing.T) {
	const candidateOutput = "Hey, world."

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, candidateOutput, 10, 5)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}
	record := DictationRecord{
		ID: "rec-1", Bucket: "30s",
		Input:     "hey comma world period",
		TruthText: "Hey, world.",
		Metrics:   []string{"exact_match", "word_error_rate", "punctuation_accuracy"},
	}

	spec, ok := prompts.For(prompts.Dictation)
	require.True(t, ok)

	// Old path: one function does the completion call AND the scoring.
	want := runDictationOne(context.Background(), gw, model, spec, record, nil)
	require.NoError(t, want.Err)

	// New path: RunRecord does the completion call, RunJudge does the
	// scoring -- exercised separately, the way Runner drives them.
	task := newTestDictationTask(t, gw)

	sample, result, err := task.RunRecord(context.Background(), model, record, nil)
	require.NoError(t, err)

	judged, err := task.RunJudge(context.Background(), model, record, "unused-judge-model", result, nil)
	require.NoError(t, err)

	assert.Equal(t, want.Model, sample.Model)
	assert.Equal(t, want.Task, sample.Task)
	assert.Equal(t, want.Bucket, sample.Bucket)
	assert.Equal(t, want.RecordID, sample.RecordID)
	assert.Equal(t, want.Judge, judged.Judge, "scoring must match the original runDictationOne exactly for the same input/truth/metrics")
}

func TestDictationTaskRunRecordRecordsAPIErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestDictationTask(t, gw)
	record := DictationRecord{ID: "rec-1", Bucket: "30s", Input: "hello", TruthText: "Hello."}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, failures)

	require.Error(t, err)
	require.Error(t, sample.Err)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.Dictation, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureAPIError, meta[0].Reason)
}

func TestDictationTaskRunRecordContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestDictationTask(t, gw)

	record := DictationRecord{ID: "rec-1", Bucket: "30s", Input: strings.Repeat("word ", 1000), TruthText: "Hello."}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen once the context-length guard trips")
}

// ---------------------------------------------------------------------------
// RunTTFT -- new behavior: the original dictation benchmark never ran a TTFT
// pass at all (see RunTTFT's doc comment on DictationTask), so there is no
// runDictationOne-equivalent to compare against here, only correctness of
// the streaming path itself.
// ---------------------------------------------------------------------------

func TestDictationTaskRunTTFTMeasuresRealTimeToFirstTokenAndNeverJudges(t *testing.T) {
	const firstChunkDelay = 50 * time.Millisecond
	const secondChunkDelay = 50 * time.Millisecond

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// Mirrors gateway_test.go's TestGateway_RunCompletionStream: a delay
		// before EACH chunk, so TimeToFirstToken reflects the first delay and
		// TotalLatency reflects both -- a single delay in only one spot can't
		// prove the two fields are actually measuring different moments.
		time.Sleep(firstChunkDelay)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"Hey, "},"finish_reason":null}]}`)

		time.Sleep(secondChunkDelay)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"world."},"finish_reason":"stop"}]}`)

		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)

		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestDictationTask(t, gw)
	record := DictationRecord{ID: "rec-1", Bucket: "30s", Input: "hey comma world period", TruthText: "Hey, world."}

	sample, result, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model"}, record, nil)

	require.NoError(t, err)
	assert.Nil(t, sample.Judge, "a TTFT sample is never judged")
	assert.Equal(t, "Hey, world.", result.Text)
	assert.GreaterOrEqual(t, sample.TimeToFirstToken, firstChunkDelay)
	assert.GreaterOrEqual(t, sample.TotalLatency, firstChunkDelay+secondChunkDelay)
	assert.Less(t, sample.TimeToFirstToken, sample.TotalLatency,
		"unlike a synchronous call, streaming has a real, strictly earlier first-token moment")
}

func TestDictationTaskRunTTFTContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestDictationTask(t, gw)

	record := DictationRecord{ID: "rec-1", Bucket: "30s", Input: strings.Repeat("word ", 1000), TruthText: "Hello."}

	sample, _, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called)
}

// ---------------------------------------------------------------------------
// DimensionRanges
// ---------------------------------------------------------------------------

func TestDictationTaskDimensionRangesCoversEveryScoreDictationName(t *testing.T) {
	task := newTestDictationTask(t, nil)
	ranges := task.DimensionRanges()

	for _, name := range []string{"exact_match", "word_error_rate", "punctuation_accuracy", "capitalization_accuracy", "formatting_command_accuracy"} {
		_, ok := ranges[name]
		assert.True(t, ok, "missing DimensionRanges entry for %q", name)
	}
}

// ---------------------------------------------------------------------------
// Full pipeline: Runner + DictationTask vs. the original
// runDictationForModel, over the same records and a shared fake gateway --
// this is the comparison the refactor exists to support.
// ---------------------------------------------------------------------------

func TestDictationTaskThroughRunnerMatchesRunDictationForModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "Hey, world.", 10, 5)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}

	records := []DictationRecord{
		{ID: "r1", Bucket: "30s", Input: "hey comma world period", TruthText: "Hey, world.", Metrics: []string{"exact_match", "word_error_rate"}},
		{ID: "r2", Bucket: "30s", Input: "hey comma world period", TruthText: "Hi, world.", Metrics: []string{"exact_match", "word_error_rate"}},
		{ID: "r3", Bucket: "120s", Input: "hey comma world period", TruthText: "Hey, world.", Metrics: []string{"punctuation_accuracy"}},
	}

	// Old pipeline.
	oldSamples := runDictationForModel(context.Background(), gw, model, records, 2, nil)
	require.Len(t, oldSamples, len(records))

	// New pipeline.
	task, err := NewDictationTask(gw)
	require.NoError(t, err)

	runner := NewRunner[DictationRecord](task, RunConfig{
		Models:      []ModelInfo{model},
		Concurrency: 2,
	})
	newSamples := runner.Run(context.Background(), records)
	require.Len(t, newSamples, len(records))

	byRecordID := func(samples []Sample) map[string]Sample {
		out := make(map[string]Sample, len(samples))
		for _, s := range samples {
			out[s.RecordID] = s
		}

		return out
	}

	oldByID, newByID := byRecordID(oldSamples), byRecordID(newSamples)

	for id, old := range oldByID {
		newer, ok := newByID[id]
		require.True(t, ok, "record %s missing from the new pipeline's output", id)

		assert.NoError(t, old.Err, "record %s (old)", id)
		assert.NoError(t, newer.Err, "record %s (new)", id)
		assert.Equal(t, old.Model, newer.Model, "record %s", id)
		assert.Equal(t, old.Task, newer.Task, "record %s", id)
		assert.Equal(t, old.Bucket, newer.Bucket, "record %s", id)
		assert.Equal(t, old.Judge, newer.Judge, "record %s: scoring must match the original pipeline exactly", id)
	}
}
