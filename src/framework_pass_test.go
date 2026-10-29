package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeFrameworkFixtures lays out both registered tasks' datasets under a
// Chdir'd temp root -- frameworkTasks always builds every registered task
// (see framework_pass.go), so every test driving runFrameworkBenchmarkPass
// needs a dataset for each of them, not just the one it's actually testing.
func writeFrameworkFixtures(t *testing.T) {
	t.Helper()

	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "dictation", "synthetic.jsonl"),
		dictationRecordLine("r1", "30s", "unused", "hey comma world period", "Hey, world.", []string{"exact_match"}))
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"),
		recordLine("r2", "30s", "a call transcript"))
}

// runFrameworkTestCanonicalConfig sets ApproveMerge -- every framework run
// now ends with a canonical merge (see framework_pass.go's package doc comment),
// so every test driving runFrameworkBenchmarkPass needs this to avoid
// blocking on the interactive stdin confirmation. The merge itself lands
// under the fixed, relative frameworkCanonicalRoot, which resolves against
// the parent of writeFrameworkFixtures'/t.Chdir's own temp root.
func runFrameworkTestCanonicalConfig(t *testing.T, cfg frameworkPassConfig) frameworkPassConfig {
	t.Helper()

	cfg.ApproveMerge = true

	return cfg
}

// frameworkOutputRunDirs lists frameworkOutputPath's fresh run dir(s),
// excluding the "canonical" entry -- frameworkCanonicalRoot ("../results/canonical") is
// physically nested inside frameworkOutputPath ("../results"), so a merged run
// always leaves one extra entry there alongside the run's own (see NewRunDir).
func frameworkOutputRunDirs(t *testing.T) []os.DirEntry {
	t.Helper()

	all, err := os.ReadDir(frameworkOutputPath)
	require.NoError(t, err)

	runDirs := make([]os.DirEntry, 0, len(all))
	for _, e := range all {
		if e.Name() != "canonical" {
			runDirs = append(runDirs, e)
		}
	}

	return runDirs
}

// TestRunFrameworkBenchmarkPass exercises the actual framework wiring
// end to end: CLI-flag-shaped config in, real dataset files and a fake
// gateway underneath, a report combining both tasks staged under the fixed
// frameworkOutputPath out. ApproveMerge is set (see runFrameworkTestCanonicalConfig)
// purely so the now-mandatory canonical-merge step at the end of every
// framework run has somewhere to land -- this test's own assertions
// are unrelated to the merge.
func TestRunFrameworkBenchmarkPass(t *testing.T) {
	writeFrameworkFixtures(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			writeChatCompletion(w, `{"faithfulness": 5, "coverage": 5, "conciseness": 5, "notes": "ok"}`, 20, 10)

			return
		}

		if strings.Contains(body, `"stream":true`) {
			writeChatCompletionStream(w, []string{"Hey, world."}, 10, 5)

			return
		}

		writeChatCompletion(w, "Hey, world.", 10, 5)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation,summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1, "the run must have staged into exactly one fresh run directory")

	reportPath := filepath.Join(frameworkOutputPath, entries[0].Name(), "report-the-judge.json")
	data, err := os.ReadFile(reportPath)
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	require.Len(t, report.Samples, 2)

	byTask := map[string]jsonSample{}
	for _, s := range report.Samples {
		byTask[s.Task] = s
	}

	require.Contains(t, byTask, "dictation")
	require.Contains(t, byTask, "summarization")
	assert.Equal(t, "r1", byTask["dictation"].RecordID)
	assert.Equal(t, "r2", byTask["summarization"].RecordID)
	assert.Equal(t, map[string]float64{"exact_match": 1}, byTask["dictation"].Judge)
	assert.Equal(t, map[string]float64{"faithfulness": 5, "coverage": 5, "conciseness": 5}, byTask["summarization"].Judge)

	require.Len(t, report.Aggregates, 2, "one Cell per task")
}

// TestRunFrameworkBenchmarkPassRecordsFailures confirms failure recording
// reaches every task the framework path runs, under the fixed
// frameworkErrorsPath.
func TestRunFrameworkBenchmarkPassRecordsFailures(t *testing.T) {
	writeFrameworkFixtures(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation,summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries, err := os.ReadDir(frameworkErrorsPath)
	require.NoError(t, err)
	require.Len(t, entries, 1)

	meta := readFailureMeta(t, filepath.Join(frameworkErrorsPath, entries[0].Name()))
	require.Len(t, meta, 2, "both tasks' candidate calls fail against this server")

	for _, m := range meta {
		assert.Equal(t, FailureAPIError, m.Reason)
	}
}

// TestRunFrameworkBenchmarkPass_alwaysRunsTTFTAndAppendsCanonical confirms
// the framework's default behavior (see framework_pass.go's package doc
// comment): every call runs the TTFT pass alongside the standard one and
// merges both staged results into canonical -- no flag needed.
func TestRunFrameworkBenchmarkPass_alwaysRunsTTFTAndAppendsCanonical(t *testing.T) {
	writeFrameworkFixtures(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			writeChatCompletion(w, `{"faithfulness": 5, "coverage": 5, "conciseness": 5, "notes": "ok"}`, 20, 10)

			return
		}

		if strings.Contains(body, `"stream":true`) {
			// A real delay before the content chunk -- otherwise a loopback
			// round-trip can land under 0.5ms and round to a TTFT of exactly
			// 0.000 (see toTTFTJSONSample's 3-decimal rounding), making the
			// "real measurement" assertion below flaky.
			time.Sleep(5 * time.Millisecond)
			writeChatCompletionStream(w, []string{"Hey, world."}, 10, 5)

			return
		}

		writeChatCompletion(w, "Hey, world.", 10, 5)
	}))
	defer server.Close()

	runFrameworkBenchmarkPass(context.Background(), frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation,summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
		ApproveMerge:        true,
	})

	// frameworkOutputPath/frameworkTTFTOutputPath/frameworkCanonicalRoot are all
	// fixed and relative -- they resolve against the parent of
	// writeFrameworkFixtures' own t.Chdir'd temp root.
	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1)
	assert.FileExists(t, filepath.Join(frameworkOutputPath, entries[0].Name(), "report-the-judge.json"))

	ttftEntries, err := os.ReadDir(frameworkTTFTOutputPath)
	require.NoError(t, err)
	require.Len(t, ttftEntries, 1, "the TTFT pass must stage its own report even though no flag requested it")

	ttftData, err := os.ReadFile(filepath.Join(frameworkTTFTOutputPath, ttftEntries[0].Name(), "ttft-report.json"))
	require.NoError(t, err)

	var ttftReport ttftJSONReport
	require.NoError(t, json.Unmarshal(ttftData, &ttftReport))
	require.Len(t, ttftReport.Samples, 2, "one TTFT sample per task")

	for _, s := range ttftReport.Samples {
		assert.Empty(t, s.Error)
		assert.Greater(t, s.TimeToFirstTokenS, 0.0, "a real streaming call must measure a nonzero TTFT")
	}

	versionDir := filepath.Join(frameworkCanonicalRoot, CanonicalVersionDirName(DefaultCanonicalVersion))
	assert.FileExists(t, filepath.Join(versionDir, "report-the-judge.json"))
	assert.FileExists(t, filepath.Join(versionDir, "ttft-report.json"))
	assert.FileExists(t, filepath.Join(versionDir, "summary.json"), "the merge must refresh canonical's own summary")
	assert.FileExists(t, filepath.Join(versionDir, "summary.csv"))
}

// TestRunFrameworkBenchmarkPass_bucketsRestrictsRunAndMergesOnlyThatBucket is
// an end-to-end proof of -buckets: a first, unrestricted run benchmarks both
// buckets the dataset has and merges both into canonical; a second run scoped
// to -buckets 3600s only must (1) never call the candidate or judge for the
// 30s record, (2) never make a single streaming call at all -- 3600s isn't
// in TTFTBuckets, so the TTFT pass has nothing to measure -- and (3) merge
// into canonical leaving the first run's 30s row exactly as it was. (3) is the
// property mergeJSONSamples/mergeCells's (model, reasoning effort, task,
// bucket) replace key (canonical.go) exists to guarantee; back when that key
// stopped at (model, reasoning effort, task), a rerun scoped to just 3600s
// would have deleted the 30s row entirely.
func TestRunFrameworkBenchmarkPass_bucketsRestrictsRunAndMergesOnlyThatBucket(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"),
		recordLine("r-30s", "30s", "a short call")+recordLine("r-3600s", "3600s", "a long call"))

	var candidateCalls, judgeCalls, streamCalls atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			judgeCalls.Add(1)
			writeChatCompletion(w, `{"faithfulness": 5, "coverage": 5, "conciseness": 5}`, 20, 10)

			return
		}

		if strings.Contains(body, `"stream":true`) {
			streamCalls.Add(1)
			writeChatCompletionStream(w, []string{"a summary"}, 10, 5)

			return
		}

		candidateCalls.Add(1)
		writeChatCompletion(w, "a summary", 10, 5)
	}))
	defer server.Close()

	baseCfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
		ApproveMerge:        true,
	}

	// First run: no -buckets, covers both 30s and 3600s.
	runFrameworkBenchmarkPass(context.Background(), baseCfg)
	assert.EqualValues(t, 2, candidateCalls.Load(), "the unrestricted first run must benchmark both records")

	canonicalPath := filepath.Join(frameworkCanonicalRoot, CanonicalVersionDirName(DefaultCanonicalVersion), "report-the-judge.json")
	assert.ElementsMatch(t, []string{"30s", "3600s"}, canonicalSampleBuckets(t, canonicalPath))

	candidateCalls.Store(0)
	judgeCalls.Store(0)
	streamCalls.Store(0)

	// Second run: -buckets 3600s only.
	scopedCfg := baseCfg
	scopedCfg.BucketsFlag = "3600s"
	runFrameworkBenchmarkPass(context.Background(), scopedCfg)

	assert.EqualValues(t, 1, candidateCalls.Load(), "only the 3600s record should reach the candidate model")
	assert.EqualValues(t, 1, judgeCalls.Load(), "only the 3600s record should be judged")
	assert.EqualValues(t, 0, streamCalls.Load(), "3600s is outside TTFTBuckets, so the TTFT pass must make no streaming call at all")

	assert.ElementsMatch(t, []string{"30s", "3600s"}, canonicalSampleBuckets(t, canonicalPath),
		"the 30s row from the first, unrestricted run must survive a merge scoped to a different bucket")
}

// canonicalSampleBuckets reads path (a merged canonical report-<judge>.json) and
// returns every sample's Bucket, so a test can assert on which buckets
// survived a merge without caring about the rest of each sample's shape.
func canonicalSampleBuckets(t *testing.T, path string) []string {
	t.Helper()

	data, err := os.ReadFile(path)
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	buckets := make([]string, 0, len(report.Samples))
	for _, s := range report.Samples {
		buckets = append(buckets, s.Bucket)
	}

	return buckets
}

// TestRunFrameworkBenchmarkPassFiltersTasks confirms -tasks actually limits
// which tasks run, end to end -- a summarization fixture exists but must be
// ignored since only dictation was requested.
func TestRunFrameworkBenchmarkPassFiltersTasks(t *testing.T) {
	writeFrameworkFixtures(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		if strings.Contains(readBody(t, r), `"stream":true`) {
			writeChatCompletionStream(w, []string{"Hey, world."}, 10, 5)

			return
		}

		writeChatCompletion(w, "Hey, world.", 10, 5)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1)

	data, err := os.ReadFile(filepath.Join(frameworkOutputPath, entries[0].Name(), "report-the-judge.json"))
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	require.Len(t, report.Samples, 1)
	assert.Equal(t, "dictation", report.Samples[0].Task)
}

// TestRunFrameworkBenchmarkPassTextCorrection confirms text_correction's
// framework port is reachable through the framework run, and that the
// reference-free judge's rubric scores -- and nothing computed from the
// original transcript -- land in the report. Self-contained fixture rather than
// writeFrameworkFixtures -- that helper's tasks (dictation/summarization)
// don't carry missed_text.
func TestRunFrameworkBenchmarkPassTextCorrection(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "text_correction", "aai_transcripts.jsonl"),
		recordLineWithMissedText("r1", "30s", "the meeting is at three", "the meating is at three"))

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			writeChatCompletion(w, textCorrectionJudgeReply(4, 4), 20, 10)

			return
		}

		if strings.Contains(body, `"stream":true`) {
			writeChatCompletionStream(w, []string{"the meeting is at three"}, 10, 5)

			return
		}

		writeChatCompletion(w, "the meeting is at three", 10, 5)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "text_correction",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1)

	data, err := os.ReadFile(filepath.Join(frameworkOutputPath, entries[0].Name(), "report-the-judge.json"))
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	require.Len(t, report.Samples, 1)
	sample := report.Samples[0]
	assert.Equal(t, "text_correction", sample.Task)
	assert.Equal(t, "r1", sample.RecordID)
	assert.Equal(t, 4.0, sample.Judge["faithfulness"])
	assert.Equal(t, 4.0, sample.Judge["improvement"])
	assert.Len(t, sample.Judge, 2, "only the reference-free rubric is recorded")

	for _, retired := range []string{"meaning_similarity", "truth_wer", "input_wer", "accuracy", "edit_ratio", "error_reduction"} {
		assert.NotContains(t, sample.Judge, retired, "%s compared the output with the original transcript and is no longer recorded", retired)
	}
}

// TestRunFrameworkBenchmarkPassInformationExtraction confirms
// information_extraction's framework port is reachable through
// the framework run, and that a schema-valid completion reaches the real judge
// (see TestRunFrameworkBenchmarkPassInformationExtractionInvalidSchema for
// the mechanical-scoring path). Self-contained fixture, same reason as
// TestRunFrameworkBenchmarkPassTextCorrection.
func TestRunFrameworkBenchmarkPassInformationExtraction(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "information_extraction", "aai_transcripts.jsonl"),
		recordLine("r1", "30s", "a call transcript"))

	const extraction = `{"participants": ["Alice"], "action_items": [], "decisions": [], ` +
		`"entities": {"people": [], "organizations": [], "products": [], "dates": []}}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			writeChatCompletion(w, `{"groundedness": 5, "completeness": 4, "notes": "ok"}`, 20, 10)

			return
		}

		if strings.Contains(body, `"stream":true`) {
			writeChatCompletionStream(w, []string{extraction}, 10, 5)

			return
		}

		writeChatCompletion(w, extraction, 10, 5)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "information_extraction",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1)

	data, err := os.ReadFile(filepath.Join(frameworkOutputPath, entries[0].Name(), "report-the-judge.json"))
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	require.Len(t, report.Samples, 1)
	sample := report.Samples[0]
	assert.Equal(t, "information_extraction", sample.Task)
	assert.Equal(t, "r1", sample.RecordID)
	assert.False(t, sample.SchemaInvalid)
	assert.Equal(t, map[string]float64{"groundedness": 5, "completeness": 4}, sample.Judge)
}

// TestRunFrameworkBenchmarkPassInformationExtractionInvalidSchema confirms a
// structurally invalid completion is scored mechanically (worst rubric,
// SchemaInvalid) end to end, never reaching the judge.
func TestRunFrameworkBenchmarkPassInformationExtractionInvalidSchema(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "information_extraction", "aai_transcripts.jsonl"),
		recordLine("r1", "30s", "a call transcript"))

	var judgeCalled atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			judgeCalled.Store(true)
			writeChatCompletion(w, `{"groundedness": 5, "completeness": 5, "notes": "unexpected"}`, 20, 10)

			return
		}

		if strings.Contains(body, `"stream":true`) {
			writeChatCompletionStream(w, []string{`{"participants": ["missing every other required key"]}`}, 10, 5)

			return
		}

		writeChatCompletion(w, `{"participants": ["missing every other required key"]}`, 10, 5)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "information_extraction",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1)

	data, err := os.ReadFile(filepath.Join(frameworkOutputPath, entries[0].Name(), "report-the-judge.json"))
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	require.Len(t, report.Samples, 1)
	sample := report.Samples[0]
	assert.True(t, sample.SchemaInvalid)
	assert.Equal(t, map[string]float64{"groundedness": 1, "completeness": 1}, sample.Judge)
	assert.False(t, judgeCalled.Load(), "a structurally invalid extraction must never reach the judge")
}

// TestRunFrameworkBenchmarkPass_reasoningEffort exercises -reasoning-effort's
// whole path end to end: the candidate model's GET /v1/models entry
// advertises support via supported_parameters, so the flag must reach the
// actual gateway request (reasoning_effort on the wire) AND the staged
// report's sample must carry its own ReasoningEffort field alongside the
// unchanged Model -- the (model, reasoning effort) pair every downstream
// summary/use-case row groups by (see ModelInfo.ReasoningEffortLabel in
// gateway.go) -- never the judge model, which is passed to RunJudge by bare
// ID and never touches -reasoning-effort at all.
func TestRunFrameworkBenchmarkPass_reasoningEffort(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"),
		recordLine("r1", "30s", "a call transcript"))

	var sawReasoningEffort atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme","supported_parameters":["reasoning_effort"]}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			writeChatCompletion(w, `{"faithfulness": 5, "coverage": 5, "conciseness": 5, "notes": "ok"}`, 20, 10)

			return
		}

		if strings.Contains(body, `"reasoning_effort":"low"`) {
			sawReasoningEffort.Store(true)
		}

		if strings.Contains(body, `"stream":true`) {
			writeChatCompletionStream(w, []string{"Hey, world."}, 10, 5)

			return
		}

		writeChatCompletion(w, "Hey, world.", 10, 5)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "summarization",
		JudgeModels:         []string{"the-judge"},
		ReasoningEffort:     "low",
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1)

	data, err := os.ReadFile(filepath.Join(frameworkOutputPath, entries[0].Name(), "report-the-judge.json"))
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	require.Len(t, report.Samples, 1)
	assert.Equal(t, "candidate-model", report.Samples[0].Model)
	assert.Equal(t, "low", report.Samples[0].ReasoningEffort)
	assert.True(t, sawReasoningEffort.Load(), "the candidate call must actually carry reasoning_effort on the wire, not just in the reported name")
}

// TestRunFrameworkBenchmarkPass_stashesEachTaskRunBesideTheReport exercises
// the whole path end to end for two different tasks: every successful candidate
// call lands as its own file under task_runs/<task>/ in THE SAME run directory
// as the report scored from it -- one directory per run, not one for each --
// carrying who it was for, what the model said, what the call measured, and the
// raw request and response. The TTFT pass stashes its streaming calls the same
// way, in the directory of the same name under results_ttft/.
func TestRunFrameworkBenchmarkPass_stashesEachTaskRunBesideTheReport(t *testing.T) {
	writeFrameworkFixtures(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			writeChatCompletion(w, `{"faithfulness": 5, "coverage": 5, "conciseness": 5, "notes": "ok"}`, 20, 10)

			return
		}

		if strings.Contains(body, `"stream":true`) {
			writeChatCompletionStream(w, []string{"streamed"}, 10, 5)

			return
		}

		writeChatCompletion(w, "Hey, world.", 10, 5)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation,summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1, "the stash must share the report's run directory, not make a second one")

	runDir := filepath.Join(frameworkOutputPath, entries[0].Name())
	assert.FileExists(t, filepath.Join(runDir, "report-the-judge.json"))

	// One file per task run, under the task it belongs to -- and the files
	// carry the candidate's own output ("Hey, world."), not the TTFT pass's
	// streamed one, which is never stashed.
	wantUserMessages := map[string]string{
		"dictation":     "hey comma world period",
		"summarization": "a call transcript",
	}
	wantRecordIDs := map[string]string{"dictation": "r1", "summarization": "r2"}

	for task, userMessage := range wantUserMessages {
		paths, err := filepath.Glob(filepath.Join(runDir, taskRunsDirName, task, "*.json"))
		require.NoError(t, err)
		require.Len(t, paths, 1, "one task run for %s", task)

		wantName := "candidate-model__default__30s__" + wantRecordIDs[task] + ".json"
		assert.Equal(t, wantName, filepath.Base(paths[0]))

		file := readTaskRunFile(t, paths[0])
		assert.Equal(t, "candidate-model", file.Model)
		assert.Equal(t, "default", file.ReasoningEffort)
		assert.Equal(t, task, file.Task)
		assert.Equal(t, "30s", file.Bucket)
		assert.Equal(t, wantRecordIDs[task], file.RecordID)
		assert.Equal(t, "Hey, world.", file.Text)
		assert.Equal(t, 10, file.PromptTokens)
		assert.Equal(t, 5, file.CompletionTokens)
		assert.Greater(t, file.TotalLatencyS, 0.0, "the call's latency is stashed too")

		// The raw exchange is there verbatim: the request names the candidate
		// and carries the record as the user message, and the response is the
		// whole gateway body, not only the extracted text.
		var request failureRequestBody
		require.NoError(t, json.Unmarshal(file.Request, &request))
		assert.Equal(t, "candidate-model", request.Model)
		require.NotEmpty(t, request.Messages)
		assert.Equal(t, userMessage, request.Messages[len(request.Messages)-1].Content)

		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}
		require.NoError(t, json.Unmarshal(file.Response, &response))
		require.Len(t, response.Choices, 1)
		assert.Equal(t, "Hey, world.", response.Choices[0].Message.Content)
		assert.Equal(t, "stop", response.Choices[0].FinishReason)
	}

	// Every stashed run is one the report scored, and the other way around.
	reportData, err := os.ReadFile(filepath.Join(runDir, "report-the-judge.json"))
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(reportData, &report))
	require.Len(t, report.Samples, 2)

	for _, sample := range report.Samples {
		stashed, err := filepath.Glob(filepath.Join(runDir, taskRunsDirName, sample.Task, "*"+sample.RecordID+".json"))
		require.NoError(t, err)
		assert.Len(t, stashed, 1, "sample %s/%s has no stashed task run", sample.Task, sample.RecordID)
	}

	// The TTFT pass stashes its own calls, in a directory of its own -- named
	// like the standard pass's, so one name finds both -- never in the standard
	// pass's.
	ttftEntries, err := os.ReadDir(frameworkTTFTOutputPath)
	require.NoError(t, err)
	require.Len(t, ttftEntries, 1)
	assert.Equal(t, entries[0].Name(), ttftEntries[0].Name(), "a run is one name across results/ and results_ttft/")

	ttftRunDir := filepath.Join(frameworkTTFTOutputPath, ttftEntries[0].Name())
	assert.FileExists(t, filepath.Join(ttftRunDir, "ttft-report.json"))

	for task, recordID := range wantRecordIDs {
		ttftPath := filepath.Join(ttftRunDir, taskRunsDirName, task, "candidate-model__default__30s__"+recordID+".json")
		require.FileExists(t, ttftPath)

		file := readTaskRunFile(t, ttftPath)
		assert.Equal(t, "streamed", file.Text, "what the streaming call returned, not the synchronous one's")
		assert.Greater(t, file.TimeToFirstTokenS, 0.0, "a streaming call measures a real time to first token")
		assert.Empty(t, file.Response, "a stream has no single response document to keep")
		assert.NotEmpty(t, file.Request)
	}

	// ...and the canonical merge, which already ran, has no use for task_runs/.
	versionDir := filepath.Join(frameworkCanonicalRoot, CanonicalVersionDirName(DefaultCanonicalVersion))
	assert.FileExists(t, filepath.Join(versionDir, "report-the-judge.json"))
	assert.NoDirExists(t, filepath.Join(versionDir, taskRunsDirName))
}

// TestRunFrameworkBenchmarkPass_doesNotStashFailedTaskRuns: a candidate call
// that fails has no output to keep, so it leaves no file -- and, since nothing
// is ever written, no empty task_runs directory either. The failure itself is
// still recorded under errors/, as before.
func TestRunFrameworkBenchmarkPass_doesNotStashFailedTaskRuns(t *testing.T) {
	writeFrameworkFixtures(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation,summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	}
	runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))

	entries := frameworkOutputRunDirs(t)
	require.Len(t, entries, 1)

	assert.NoDirExists(t, filepath.Join(frameworkOutputPath, entries[0].Name(), taskRunsDirName))

	failureDirs, err := os.ReadDir(frameworkErrorsPath)
	require.NoError(t, err)
	require.Len(t, failureDirs, 1)
	assert.Len(t, readFailureMeta(t, filepath.Join(frameworkErrorsPath, failureDirs[0].Name())), 2, "both failures are recorded where they always were")
}

// ---------------------------------------------------------------------------
// -replay -- end to end
// ---------------------------------------------------------------------------

// countingGateway is a fake llm_gateway that counts the calls it gets by kind --
// a candidate's synchronous completion, a candidate's streaming one (the TTFT
// pass), a judge's -- and keeps what the judge was sent, so a replay test can
// prove which calls were NOT made and what the judge was shown instead.
// Candidates answer with output; a judge scores by name: "the-judge" gives every
// dimension a 5, "the-new-judge" a 2.
type countingGateway struct {
	*httptest.Server

	candidateCalls atomic.Int32
	streamCalls    atomic.Int32
	judgeCalls     atomic.Int32

	mu          sync.Mutex
	output      string
	judgeBodies []string
}

func newCountingGateway(t *testing.T) *countingGateway {
	t.Helper()

	g := &countingGateway{output: "Hey, world."}

	g.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		g.mu.Lock()
		output := g.output
		g.mu.Unlock()

		switch {
		case strings.Contains(body, `"model":"the-judge"`), strings.Contains(body, `"model":"the-new-judge"`):
			g.judgeCalls.Add(1)

			g.mu.Lock()
			g.judgeBodies = append(g.judgeBodies, body)
			g.mu.Unlock()

			score := 5
			if strings.Contains(body, `"model":"the-new-judge"`) {
				score = 2
			}

			writeChatCompletion(w, fmt.Sprintf(`{"faithfulness": %d, "coverage": %d, "conciseness": %d, "notes": "ok"}`, score, score, score), 20, 10)
		case strings.Contains(body, `"stream":true`):
			g.streamCalls.Add(1)
			time.Sleep(5 * time.Millisecond) // a real gap before the first chunk, so a TTFT is measurable
			writeChatCompletionStream(w, []string{output}, 10, 5)
		default:
			g.candidateCalls.Add(1)
			writeChatCompletion(w, output, 10, 5)
		}
	}))

	t.Cleanup(g.Server.Close)

	return g
}

// startAgain makes a candidate say output from here on and zeroes the counters,
// so what the gateway sees from now on is only what the next run does.
func (g *countingGateway) startAgain(output string) {
	g.mu.Lock()
	g.output = output
	g.judgeBodies = nil
	g.mu.Unlock()

	g.candidateCalls.Store(0)
	g.streamCalls.Store(0)
	g.judgeCalls.Store(0)
}

func (g *countingGateway) sentToTheJudge() []string {
	g.mu.Lock()
	defer g.mu.Unlock()

	return append([]string(nil), g.judgeBodies...)
}

// readRunReport reads runDir's report-<judge>.json, keyed by task (every test
// here runs one record per task).
func readRunReport(t *testing.T, runDir, judge string) map[string]jsonSample {
	t.Helper()

	data, err := os.ReadFile(filepath.Join(runDir, "report-"+judge+".json"))
	require.NoError(t, err)

	var report jsonReport
	require.NoError(t, json.Unmarshal(data, &report))

	byTask := map[string]jsonSample{}
	for _, s := range report.Samples {
		byTask[s.Task] = s
	}

	return byTask
}

// readTTFTRunReport is readRunReport for a TTFT run directory.
func readTTFTRunReport(t *testing.T, runDir string) map[string]ttftJSONSample {
	t.Helper()

	data, err := os.ReadFile(filepath.Join(runDir, "ttft-report.json"))
	require.NoError(t, err)

	var report ttftJSONReport
	require.NoError(t, json.Unmarshal(data, &report))

	byTask := map[string]ttftJSONSample{}
	for _, s := range report.Samples {
		byTask[s.Task] = s
	}

	return byTask
}

// runNamesOtherThan lists the run directories under results/ other than skip.
func runNamesOtherThan(t *testing.T, skip ...string) []string {
	t.Helper()

	var names []string

	for _, e := range frameworkOutputRunDirs(t) {
		if !slices.Contains(skip, e.Name()) {
			names = append(names, e.Name())
		}
	}

	return names
}

// TestRunFrameworkBenchmarkPass_replayReusesAnEarlierRunAndOnlyJudges is -replay
// end to end: a normal run, then a replay of it with a different judge. The
// replay calls no candidate model and streams nothing -- yet its report has the
// new judge's scores over the earlier run's outputs, the earlier run's
// measurements, and a stash of its own that can be replayed in turn.
func TestRunFrameworkBenchmarkPass_replayReusesAnEarlierRunAndOnlyJudges(t *testing.T) {
	writeFrameworkFixtures(t)

	g := newCountingGateway(t)
	cfg := frameworkPassConfig{
		GatewayURL:          g.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation,summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
		ApproveMerge:        true,
	}

	// The earlier run: an ordinary one.
	runFrameworkBenchmarkPass(context.Background(), cfg)

	require.EqualValues(t, 2, g.candidateCalls.Load())
	require.EqualValues(t, 2, g.streamCalls.Load())
	require.EqualValues(t, 1, g.judgeCalls.Load(), "dictation is scored mechanically: only summarization has a judge call")

	firstRuns := runNamesOtherThan(t)
	require.Len(t, firstRuns, 1)

	first := firstRuns[0]

	// From here on a candidate that IS called would say something else, so an
	// output that still reads "Hey, world." can only have come out of the stash.
	g.startAgain("LIVE OUTPUT")

	logged := captureLog(t)

	replayCfg := cfg
	replayCfg.Replay = first
	replayCfg.JudgeModels = []string{"the-new-judge"}
	runFrameworkBenchmarkPass(context.Background(), replayCfg)

	assert.EqualValues(t, 0, g.candidateCalls.Load(), "no candidate model is called again")
	assert.EqualValues(t, 0, g.streamCalls.Load(), "...nor streamed from again")
	assert.EqualValues(t, 1, g.judgeCalls.Load(), "the judge is, and it is the new one")

	judgeSaw := g.sentToTheJudge()
	require.Len(t, judgeSaw, 1)
	assert.Contains(t, judgeSaw[0], "Hey, world.", "the judge is shown the stashed output")
	assert.NotContains(t, judgeSaw[0], "LIVE OUTPUT")

	// The log says what was reused, per pass and task.
	assert.Contains(t, logged.String(), "replay: summarization: reused 1 of 1 task run(s)")
	assert.Contains(t, logged.String(), "ttft: replay: dictation: reused 1 of 1 task run(s)")
	assert.Contains(t, logged.String(), "called 0 live")

	secondRuns := runNamesOtherThan(t, first)
	require.Len(t, secondRuns, 1)

	second := secondRuns[0]
	firstDir, secondDir := filepath.Join(frameworkOutputPath, first), filepath.Join(frameworkOutputPath, second)

	// The report: the new judge's scores, and the mechanical score recomputed
	// over the stashed text.
	firstReport, secondReport := readRunReport(t, firstDir, "the-judge"), readRunReport(t, secondDir, "the-new-judge")

	assert.Equal(t, map[string]float64{"faithfulness": 5, "coverage": 5, "conciseness": 5}, firstReport["summarization"].Judge)
	assert.Equal(t, map[string]float64{"faithfulness": 2, "coverage": 2, "conciseness": 2}, secondReport["summarization"].Judge)
	assert.Equal(t, map[string]float64{"exact_match": 1}, secondReport["dictation"].Judge)

	// What the candidate calls measured is the earlier run's, not new numbers.
	for task, a := range firstReport {
		b := secondReport[task]
		assert.Equal(t, a.TotalLatencyS, b.TotalLatencyS, task)
		assert.Equal(t, a.TimeToFirstTokenS, b.TimeToFirstTokenS, task)
		assert.Equal(t, a.PromptTokens, b.PromptTokens, task)
		assert.Equal(t, a.CompletionTokens, b.CompletionTokens, task)
		assert.Equal(t, a.CostUSD, b.CostUSD, task)
	}

	// ...and so is the TTFT, which was replayed under the same name.
	firstTTFT := readTTFTRunReport(t, filepath.Join(frameworkTTFTOutputPath, first))
	secondTTFT := readTTFTRunReport(t, filepath.Join(frameworkTTFTOutputPath, second))
	require.Len(t, secondTTFT, 2)

	for task, a := range firstTTFT {
		b := secondTTFT[task]
		assert.Greater(t, a.TimeToFirstTokenS, 0.0, task)
		assert.Equal(t, a.TimeToFirstTokenS, b.TimeToFirstTokenS, task)
		assert.Equal(t, a.TotalLatencyS, b.TotalLatencyS, task)
	}

	// The replay stashed what it reused, in the new run's directories: complete
	// in themselves, holding the stashed output -- not what a live call would say.
	for _, root := range []string{frameworkOutputPath, frameworkTTFTOutputPath} {
		for _, task := range []string{"dictation", "summarization"} {
			paths, err := filepath.Glob(filepath.Join(root, second, taskRunsDirName, task, "*.json"))
			require.NoError(t, err)
			require.Len(t, paths, 1, "%s/%s", root, task)
			assert.Equal(t, "Hey, world.", readTaskRunFile(t, paths[0]).Text)
		}
	}

	// Both judges' reports merged into canonical.
	versionDir := filepath.Join(frameworkCanonicalRoot, CanonicalVersionDirName(DefaultCanonicalVersion))
	assert.FileExists(t, filepath.Join(versionDir, "report-the-judge.json"))
	assert.FileExists(t, filepath.Join(versionDir, "report-the-new-judge.json"))

	// A replay can itself be replayed -- the first run's directories out of the
	// picture, as if deleted -- and still nothing is called.
	g.startAgain("LIVE OUTPUT")

	require.NoError(t, os.RemoveAll(firstDir))
	require.NoError(t, os.RemoveAll(filepath.Join(frameworkTTFTOutputPath, first)))

	thirdCfg := cfg
	thirdCfg.Replay = second
	runFrameworkBenchmarkPass(context.Background(), thirdCfg)

	assert.EqualValues(t, 0, g.candidateCalls.Load())
	assert.EqualValues(t, 0, g.streamCalls.Load())
	assert.EqualValues(t, 1, g.judgeCalls.Load())
}

// TestRunFrameworkBenchmarkPass_replayCallsLiveWhatTheEarlierRunDidNotStash: a
// replay skips the calls that were made and makes the ones that weren't -- here a
// whole task the earlier run never ran, both passes' calls for it.
func TestRunFrameworkBenchmarkPass_replayCallsLiveWhatTheEarlierRunDidNotStash(t *testing.T) {
	writeFrameworkFixtures(t)

	g := newCountingGateway(t)
	cfg := frameworkPassConfig{
		GatewayURL:          g.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
		ApproveMerge:        true,
	}

	runFrameworkBenchmarkPass(context.Background(), cfg)

	firstRuns := runNamesOtherThan(t)
	require.Len(t, firstRuns, 1)

	g.startAgain("Hey, world.")

	// The replay adds summarization, which the earlier run never ran.
	replayCfg := cfg
	replayCfg.TasksFlag = "dictation,summarization"
	replayCfg.Replay = firstRuns[0]
	runFrameworkBenchmarkPass(context.Background(), replayCfg)

	assert.EqualValues(t, 1, g.candidateCalls.Load(), "only summarization's candidate call is made; dictation's is replayed")
	assert.EqualValues(t, 1, g.streamCalls.Load(), "...and only its streaming call")
	assert.EqualValues(t, 1, g.judgeCalls.Load())

	secondRuns := runNamesOtherThan(t, firstRuns[0])
	require.Len(t, secondRuns, 1)

	report := readRunReport(t, filepath.Join(frameworkOutputPath, secondRuns[0]), "the-judge")
	assert.Contains(t, report, "dictation")
	assert.Contains(t, report, "summarization")
	assert.Equal(t, map[string]float64{"faithfulness": 5, "coverage": 5, "conciseness": 5}, report["summarization"].Judge)
}

// TestRunFrameworkBenchmarkPass_skipJudgeGathersThenReplayJudges: -skip-judge makes
// every candidate call (both passes), stashes them, judges nothing -- dictation's
// mechanical scoring included -- and merges nothing into canonical; replaying the
// run afterwards judges it without a single new candidate call.
func TestRunFrameworkBenchmarkPass_skipJudgeGathersThenReplayJudges(t *testing.T) {
	writeFrameworkFixtures(t)

	g := newCountingGateway(t)
	cfg := frameworkPassConfig{
		GatewayURL:          g.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "dictation,summarization",
		JudgeModels:         []string{"the-judge"},
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
		ApproveMerge:        true,
		SkipJudge:           true,
	}

	runFrameworkBenchmarkPass(context.Background(), cfg)

	assert.EqualValues(t, 2, g.candidateCalls.Load())
	assert.EqualValues(t, 2, g.streamCalls.Load())
	assert.EqualValues(t, 0, g.judgeCalls.Load())

	runs := runNamesOtherThan(t)
	require.Len(t, runs, 1)

	report := readRunReport(t, filepath.Join(frameworkOutputPath, runs[0]), "the-judge")
	require.Len(t, report, 2)

	for task, sample := range report {
		assert.Empty(t, sample.Judge, task)
		assert.Empty(t, sample.Error, task)
	}

	assert.NoDirExists(t, frameworkCanonicalRoot, "unjudged samples must not be merged into canonical")

	for _, root := range []string{frameworkOutputPath, frameworkTTFTOutputPath} {
		paths, err := filepath.Glob(filepath.Join(root, runs[0], taskRunsDirName, "*", "*.json"))
		require.NoError(t, err)
		assert.Len(t, paths, 2, root)
	}

	// Judge it later.
	g.startAgain("LIVE OUTPUT")

	replayCfg := cfg
	replayCfg.SkipJudge = false
	replayCfg.Replay = runs[0]
	runFrameworkBenchmarkPass(context.Background(), replayCfg)

	assert.EqualValues(t, 0, g.candidateCalls.Load())
	assert.EqualValues(t, 0, g.streamCalls.Load())
	assert.EqualValues(t, 1, g.judgeCalls.Load())

	judged := readRunReport(t, filepath.Join(frameworkOutputPath, runNamesOtherThan(t, runs[0])[0]), "the-judge")
	assert.Equal(t, map[string]float64{"faithfulness": 5, "coverage": 5, "conciseness": 5}, judged["summarization"].Judge)
	assert.Equal(t, map[string]float64{"exact_match": 1}, judged["dictation"].Judge)
	assert.DirExists(t, frameworkCanonicalRoot)
}

// ---------------------------------------------------------------------------
// frameworkTasks -- filtering logic in isolation, no gateway/filesystem
// needed since task construction alone does no I/O.
// ---------------------------------------------------------------------------

func TestFrameworkTasksFiltersToRequested(t *testing.T) {
	runners, err := frameworkTasks(nil, RunConfig{}, []string{"dictation"})
	require.NoError(t, err)
	require.Len(t, runners, 1)
	assert.Equal(t, "dictation", runners[0].Name())
}

// "unknown_task" stands in for "a requested task with no framework port" --
// every real task in prompts.All is registered now, so this uses a name
// that will never collide with a future port, rather than a real task name
// that would need updating again the next time one is added.
func TestFrameworkTasksSkipsTasksWithNoFrameworkPort(t *testing.T) {
	runners, err := frameworkTasks(nil, RunConfig{}, []string{"dictation", "unknown_task"})
	require.NoError(t, err)
	require.Len(t, runners, 1)
	assert.Equal(t, "dictation", runners[0].Name())
}

func TestFrameworkTasksErrorsWhenNoRequestedTaskHasAPort(t *testing.T) {
	_, err := frameworkTasks(nil, RunConfig{}, []string{"unknown_task"})
	assert.Error(t, err)
}

func TestFrameworkTasksPreservesRequestedOrder(t *testing.T) {
	runners, err := frameworkTasks(nil, RunConfig{}, []string{"summarization", "dictation"})
	require.NoError(t, err)
	require.Len(t, runners, 2)
	assert.Equal(t, "summarization", runners[0].Name())
	assert.Equal(t, "dictation", runners[1].Name())
}
