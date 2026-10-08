package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFilterGroupsByBucket(t *testing.T) {
	groups := []TaskGroup{
		{Task: prompts.Summarization, Bucket: "30s"},
		{Task: prompts.Summarization, Bucket: "900s"},
		{Task: prompts.Summarization, Bucket: "120s"},
		{Task: prompts.TextCorrection, Bucket: "3600s"},
	}

	filtered := filterGroupsByBucket(groups, TTFTBuckets)

	require.Len(t, filtered, 2, "only 30s and 120s groups survive; 900s and 3600s must be dropped")
	assert.Equal(t, "30s", filtered[0].Bucket)
	assert.Equal(t, "120s", filtered[1].Bucket)
}

func TestFilterGroupsByBucket_noMatches(t *testing.T) {
	groups := []TaskGroup{{Task: prompts.Summarization, Bucket: "3600s"}}
	assert.Empty(t, filterGroupsByBucket(groups, TTFTBuckets))
}

// TestRunTTFTOne_success locks in that a successful streaming call produces
// a TTFTSample carrying real measurements and, critically, that it NEVER
// makes a second (judge) call the way runOne does — there is exactly one
// task-completion request on this path, full stop.
func TestRunTTFTOne_success(t *testing.T) {
	var calls atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		writeChatCompletionStream(w, []string{"a fine summary"}, 10, 4)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}
	sample := runTTFTOne(context.Background(), gw, ModelInfo{ID: "candidate-model"}, prompts.Summarization, spec, record, nil)

	require.NoError(t, sample.Err)
	assert.Equal(t, "candidate-model", sample.Model)
	assert.Equal(t, prompts.Summarization, sample.Task)
	assert.Equal(t, "30s", sample.Bucket)
	assert.Equal(t, "rec-1", sample.RecordID)
	assert.Equal(t, 10, sample.PromptTokens)
	assert.Equal(t, 4, sample.CompletionTokens)
	assert.Greater(t, sample.TimeToFirstToken, time.Duration(0))
	assert.GreaterOrEqual(t, sample.TotalLatency, sample.TimeToFirstToken)

	assert.Equal(t, int32(1), calls.Load(), "the TTFT path must never call a judge -- exactly one HTTP request per sample")
}

// TestRunTTFTOne_contextLengthExceeded mirrors runOne's own pre-flight
// guard: a transcript estimated to exceed the model's context window must
// never reach the network at all.
func TestRunTTFTOne_contextLengthExceeded(t *testing.T) {
	called := false

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletionStream(w, []string{"should never be requested"}, 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: strings.Repeat("word ", 1000)}
	sample := runTTFTOne(context.Background(), gw, ModelInfo{ID: "candidate-model", ContextLength: 100}, prompts.Summarization, spec, record, nil)

	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen once the context-length guard trips")
}

// TestRunTTFTOne_textCorrection_noMissedTextIsAFailedSample mirrors
// TestRunOne_textCorrection_noMissedTextIsAFailedSample (voicebench_test.go):
// taskInput's guard against a text_correction record with no simulated
// corruption applies identically on this path.
func TestRunTTFTOne_textCorrection_noMissedTextIsAFailedSample(t *testing.T) {
	called := false

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletionStream(w, []string{"should never be requested"}, 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	spec, ok := prompts.For(prompts.TextCorrection)
	require.True(t, ok)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the true text", MissedText: ""}
	sample := runTTFTOne(context.Background(), gw, ModelInfo{ID: "candidate-model"}, prompts.TextCorrection, spec, record, nil)

	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen at all when there's nothing to correct")
}

// TestRunTTFTOne_recordsAPIErrorFailure mirrors
// TestRunOne_recordsAPIErrorFailure (voicebench_test.go): a failed streaming
// call must be recorded as FailureAPIError, with the request that was sent,
// into the FailureRecorder passed in -- the SEPARATE one the TTFT pass sets
// up, never the standard benchmark's.
func TestRunTTFTOne_recordsAPIErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, "internal error")
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors_ttft"))
	require.NoError(t, err)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "hello"}
	sample := runTTFTOne(context.Background(), gw, ModelInfo{ID: "candidate-model"}, prompts.Summarization, spec, record, failures)

	require.Error(t, sample.Err)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, "candidate-model", meta[0].Model)
	assert.Equal(t, FailureAPIError, meta[0].Reason)
}

// TestRunTTFTModel_boundedByConcurrency proves runTTFTModel actually caps
// in-flight streaming requests, the same property
// TestJudgeConcurrency_isCappedAcrossModels locks in for the standard path's
// runModel.
func TestRunTTFTModel_boundedByConcurrency(t *testing.T) {
	var active, peak atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := active.Add(1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}

		time.Sleep(30 * time.Millisecond)
		active.Add(-1)

		writeChatCompletionStream(w, []string{"ok"}, 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)

	byTaskBucket := map[string]map[string][]TranscriptRecord{
		prompts.Summarization: {
			"30s": {
				{ID: "r1", Bucket: "30s", Text: "one"},
				{ID: "r2", Bucket: "30s", Text: "two"},
				{ID: "r3", Bucket: "30s", Text: "three"},
				{ID: "r4", Bucket: "30s", Text: "four"},
			},
		},
	}

	groups := GroupByTaskAndBucket(byTaskBucket, []string{prompts.Summarization})
	samples := runTTFTModel(context.Background(), gw, ModelInfo{ID: "m"}, groups, 2, nil)

	require.Len(t, samples, 4)
	assert.LessOrEqual(t, peak.Load(), int32(2), "concurrency=2 must never allow more than 2 in-flight streaming requests")
}

func TestAggregateTTFT(t *testing.T) {
	samples := []TTFTSample{
		{Model: "b", Task: "summarization", Bucket: "30s", TimeToFirstToken: 100 * time.Millisecond, TotalLatency: 400 * time.Millisecond},
		{Model: "b", Task: "summarization", Bucket: "30s", TimeToFirstToken: 200 * time.Millisecond, TotalLatency: 600 * time.Millisecond},
		{Model: "a", Task: "summarization", Bucket: "120s", TimeToFirstToken: 300 * time.Millisecond, TotalLatency: time.Second},
		{Model: "a", Task: "summarization", Bucket: "120s", Err: errors.New("timeout")},
	}

	cells := AggregateTTFT(samples)
	require.Len(t, cells, 2, "one cell per distinct (model, task, bucket)")

	// Sort order: model "a" (bucket 120s) before model "b" (bucket 30s).
	assert.Equal(t, "a", cells[0].Model)
	assert.Equal(t, "120s", cells[0].Bucket)
	assert.Equal(t, "b", cells[1].Model)
	assert.Equal(t, "30s", cells[1].Bucket)

	a120 := cells[0]
	assert.Equal(t, 2, a120.N)
	assert.Equal(t, 1, a120.Errors)
	require.Equal(t, 1, a120.TTFT.N, "the failed sample must not contribute a TTFT data point")
	assert.InDelta(t, 0.3, a120.TTFT.Mean, 1e-9)

	b30 := cells[1]
	assert.Equal(t, 0, b30.Errors)
	require.Equal(t, 2, b30.TTFT.N)
	assert.InDelta(t, 0.1, b30.TTFT.Min, 1e-9)
	assert.InDelta(t, 0.2, b30.TTFT.Max, 1e-9)
	assert.InDelta(t, 0.5, b30.Latency.Mean, 1e-9)
}

// TestPrintTTFTReport_sectionsByTask mirrors TestPrintReport_sectionsByTask
// (report_test.go): one header per task, when the task changes.
func TestPrintTTFTReport_sectionsByTask(t *testing.T) {
	cells := []TTFTCell{
		{Model: "a", Task: "summarization", Bucket: "30s"},
		{Model: "b", Task: "summarization", Bucket: "30s"},
		{Model: "a", Task: "text_correction", Bucket: "30s"},
	}

	var buf bytes.Buffer
	PrintTTFTReport(&buf, cells)

	output := buf.String()
	assert.Equal(t, 1, strings.Count(output, "=== ttft: summarization ==="))
	assert.Equal(t, 1, strings.Count(output, "=== ttft: text_correction ==="))
	assert.Contains(t, output, "a")
	assert.Contains(t, output, "b")
}

func TestPrintTTFTReport_doesNotPanicOnNoData(t *testing.T) {
	cells := []TTFTCell{{Model: "broken-model", Task: "summarization", Bucket: "30s", N: 2, Errors: 2}}

	assert.NotPanics(t, func() {
		var buf bytes.Buffer
		PrintTTFTReport(&buf, cells)
		assert.Contains(t, buf.String(), "no data")
	})
}

func TestWriteTTFTJSONReport_roundTrips(t *testing.T) {
	samples := []TTFTSample{
		{
			Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1",
			TimeToFirstToken: 200 * time.Millisecond, TotalLatency: time.Second,
			CompletionTokens: 40, PromptTokens: 20,
		},
		{
			Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r2",
			Err: errors.New("rate limited"),
		},
	}
	cells := AggregateTTFT(samples)

	// WriteTTFTJSONReport writes into a directory that already exists -- in a
	// real run, the one the TTFT pass made when it started, named after the
	// standard pass's (see NewNamedRunDir).
	ttftDir, err := NewNamedRunDir(filepath.Join(t.TempDir(), "results_ttft"), "a-run")
	require.NoError(t, err)
	require.NoError(t, WriteTTFTJSONReport(ttftDir, samples, cells))

	matches, err := filepath.Glob(filepath.Join(ttftDir, "ttft-report*.json"))
	require.NoError(t, err)
	require.Len(t, matches, 1, "exactly one ttft-report.json in the run directory")

	raw, err := os.ReadFile(matches[0])
	require.NoError(t, err)

	var parsed ttftJSONReport
	require.NoError(t, json.Unmarshal(raw, &parsed), "output must be valid, re-parseable JSON")

	require.Len(t, parsed.Samples, 2)

	success := parsed.Samples[0]
	assert.Equal(t, "model-a", success.Model)
	assert.InDelta(t, 0.2, success.TimeToFirstTokenS, 1e-9)
	assert.InDelta(t, 1.0, success.TotalLatencyS, 1e-9)
	assert.Empty(t, success.Error)

	failed := parsed.Samples[1]
	assert.Equal(t, "rate limited", failed.Error)

	require.Len(t, parsed.Aggregates, 1)
	assert.Equal(t, 1, parsed.Aggregates[0].Errors)
}

// TestWriteTTFTJSONReport_failsIfDirDoesNotExist: like WriteJSONReport, the
// report goes into a run directory somebody else already made.
func TestWriteTTFTJSONReport_failsIfDirDoesNotExist(t *testing.T) {
	err := WriteTTFTJSONReport(filepath.Join(t.TempDir(), "missing"), nil, nil)

	assert.Error(t, err)
}
