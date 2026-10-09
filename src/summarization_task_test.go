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

// newTestSummarizationTask constructs a SummarizationTask for tests that
// don't need a real gateway or dataset (construction only resolves
// prompts.Spec -- see NewSummarizationTask's doc comment).
func newTestSummarizationTask(t *testing.T, gw *Gateway) *SummarizationTask {
	t.Helper()

	task, err := NewSummarizationTask(gw)
	require.NoError(t, err)

	return task
}

func TestNewSummarizationTaskResolvesSpecAndName(t *testing.T) {
	task, err := NewSummarizationTask(nil)
	require.NoError(t, err)
	assert.Equal(t, prompts.Summarization, task.Name())
}

// ---------------------------------------------------------------------------
// LoadDataset / SampleDataset -- thin adapters around dataset.go's own
// loader/sampler, so these mainly confirm the adapter does the right thing:
// pulls out only summarization's own records, and samples the same way
// SampleByBucket already does.
//
// NewSummarizationTask's dataset location is a fixed, task-owned convention
// (normalized/summarization/aai_transcripts.jsonl, under normalizedRoot, relative to the working
// directory) -- not a constructor parameter a test can point elsewhere --
// so every test below Chdirs into a fresh temp directory and lays fixtures
// out under that same relative path, exactly like a real invocation would
// from the repo root.
// ---------------------------------------------------------------------------

func TestSummarizationTaskLoadDataset(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"),
		recordLine("r1", "30s", "hello world")+recordLine("r2", "120s", "a longer call"))

	task, err := NewSummarizationTask(nil)
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
	assert.Equal(t, "a longer call", byID["r2"].Text)
}

func TestSummarizationTaskLoadDatasetOnlyReturnsSummarizationRecords(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"), recordLine("s1", "30s", "summarization transcript"))
	// A sibling task folder must never leak in now that the glob is scoped
	// to exactly one folder (normalized/summarization), not a wildcard
	// spanning every task the way the original dataset glob could.
	writeFile(t, filepath.Join(normalizedRoot, "text_correction", "aai_transcripts.jsonl"), recordLine("c1", "30s", "correction transcript"))

	task, err := NewSummarizationTask(nil)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 1, "text_correction's sibling folder must never leak into summarization's own, task-scoped glob")
	assert.Equal(t, "s1", records[0].ID)
}

// TestSummarizationTaskLoadDatasetPropagatesGlobError covers the "no
// data present at all" case: since the glob now names exactly one file
// under exactly one task's folder, LoadTranscripts' "no file matched" error
// is the only way an empty dataset shows up -- there is no longer a
// separate "matched other tasks' folders but not mine" case a multi-task
// glob used to allow for (see SummarizationTask.LoadDataset's doc comment).
func TestSummarizationTaskLoadDatasetPropagatesGlobError(t *testing.T) {
	t.Chdir(t.TempDir()) // empty -- normalized/summarization/aai_transcripts.jsonl matches nothing

	task, err := NewSummarizationTask(nil)
	require.NoError(t, err)

	_, err = task.LoadDataset()
	assert.Error(t, err)
}

func TestSummarizationTaskSampleDatasetMatchesSampleByBucket(t *testing.T) {
	task := newTestSummarizationTask(t, nil)

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

	var thirtySec []string
	for _, r := range got {
		if r.Bucket == "30s" {
			thirtySec = append(thirtySec, r.ID)
		}
	}

	assert.Equal(t, []string{"a", "b"}, thirtySec, "capped and sorted by id, same rule as SampleByBucket")
}

// ---------------------------------------------------------------------------
// maxCompletionTokens -- must match the summarization branch (the default
// case) of the original benchmark's taskMaxTokens exactly.
// ---------------------------------------------------------------------------

func TestSummarizationTaskMaxCompletionTokensMatchesOriginalTaskMaxTokens(t *testing.T) {
	task := newTestSummarizationTask(t, nil)

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
			// transcript length must not matter for summarization (fixed
			// budget) -- passed here only because taskMaxTokens' signature
			// requires one; both a short and a long value must agree.
			for _, transcript := range []string{"short", strings.Repeat("word ", 5000)} {
				want := taskMaxTokens(prompts.Summarization, transcript, tc.model)
				got := task.maxCompletionTokens(tc.model)
				assert.Equal(t, want, got)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// RunRecord / RunJudge
// ---------------------------------------------------------------------------

func TestSummarizationTaskRunRecordRecordsAPIErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestSummarizationTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, failures)

	require.Error(t, err)
	require.Error(t, sample.Err)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.Summarization, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureAPIError, meta[0].Reason)
}

func TestSummarizationTaskRunRecordContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestSummarizationTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen once the context-length guard trips")
}

func TestSummarizationTaskRunJudgeRecordsJudgeErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "I refuse to grade this.", 10, 8)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestSummarizationTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}

	_, err = task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: "a summary"}, failures)

	require.Error(t, err, "a judge reply with no JSON object must surface as an error")
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.Summarization, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureJudgeError, meta[0].Reason)
}

// ---------------------------------------------------------------------------
// RunTTFT -- unlike dictation, summarization already had TTFT support in
// the original benchmark (ttft.go's runTTFTOne, generic across the three
// transcript tasks), so this is a real comparison, not just new-behavior
// coverage.
// ---------------------------------------------------------------------------

func TestSummarizationTaskRunTTFTMatchesRunTTFTOne(t *testing.T) {
	var bodies []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, readBody(t, r))

		w.Header().Set("Content-Type", "text/event-stream")
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"a fine "},"finish_reason":null}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"summary"},"finish_reason":"stop"}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the source transcript"}

	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	// Old path.
	want := runTTFTOne(context.Background(), gw, model, prompts.Summarization, spec, record, nil)
	require.NoError(t, want.Err)

	// New path.
	task := newTestSummarizationTask(t, gw)
	sample, result, err := task.RunTTFT(context.Background(), model, record, nil)
	require.NoError(t, err)

	assert.Equal(t, want.Model, sample.Model)
	assert.Equal(t, want.Task, sample.Task)
	assert.Equal(t, want.Bucket, sample.Bucket)
	assert.Equal(t, want.RecordID, sample.RecordID)
	assert.Equal(t, "a fine summary", result.Text)
	assert.Nil(t, sample.Judge, "a TTFT sample is never judged")

	require.Len(t, bodies, 2, "both paths must make exactly one streaming call each")

	var oldReq, newReq chatWireRequest
	require.NoError(t, json.Unmarshal([]byte(bodies[0]), &oldReq))
	require.NoError(t, json.Unmarshal([]byte(bodies[1]), &newReq))

	assert.Equal(t, oldReq.MaxTokens, newReq.MaxTokens)
	assert.Equal(t, oldReq.Messages, newReq.Messages)
}

func TestSummarizationTaskRunTTFTContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestSummarizationTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called)
}

// ---------------------------------------------------------------------------
// DimensionRanges
// ---------------------------------------------------------------------------

func TestSummarizationTaskDimensionRangesMatchesRubric(t *testing.T) {
	task := newTestSummarizationTask(t, nil)

	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	ranges := task.DimensionRanges()
	require.Len(t, ranges, len(spec.Rubric))

	for _, dim := range spec.Rubric {
		got, ok := ranges[dim]
		require.True(t, ok, "missing DimensionRanges entry for %q", dim)
		assert.Equal(t, scale{lo: 1, hi: 5}, got)
	}
}
