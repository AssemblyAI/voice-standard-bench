package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
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
		JudgeModel:          "the-judge",
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
		JudgeModel:          "the-judge",
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
		JudgeModel:          "the-judge",
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
		JudgeModel:          "the-judge",
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
		JudgeModel:          "the-judge",
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
// framework port is reachable through the framework run, and that its
// reference-based meaning_similarity -- and the word-level context
// (truth_wer, input_wer) -- land in the report alongside the reference-free
// judge's rubric scores. Self-contained fixture rather than
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
			writeChatCompletion(w, textCorrectionJudgeReply(body, 4, 4, 5), 20, 10)

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
		JudgeModel:          "the-judge",
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
	assert.Equal(t, 5.0, sample.Judge["meaning_similarity"], "the reference-based judge's score is recorded")
	assert.InDelta(t, 0.0, sample.Judge["truth_wer"], 1e-9, "no words differ from the hidden true text")
	assert.InDelta(t, 20.0, sample.Judge["input_wer"], 1e-9, "one wrong word of five in the corrupted input the model was shown")
	assert.NotContains(t, sample.Judge, "accuracy", "the retired word-overlap score must not be recorded")
	assert.NotContains(t, sample.Judge, "edit_ratio", "the retired drift score must not be recorded")
	assert.NotContains(t, sample.Judge, "error_reduction", "the retired word-for-word score must not be recorded")
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
		JudgeModel:          "the-judge",
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
		JudgeModel:          "the-judge",
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
		JudgeModel:          "the-judge",
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
// raw request and response, and nothing else is stashed.
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
		JudgeModel:          "the-judge",
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

	// The TTFT pass never judges what it gets back, so it stashes nothing.
	ttftEntries, err := os.ReadDir(frameworkTTFTOutputPath)
	require.NoError(t, err)
	require.Len(t, ttftEntries, 1)
	assert.NoDirExists(t, filepath.Join(frameworkTTFTOutputPath, ttftEntries[0].Name(), taskRunsDirName))

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
		JudgeModel:          "the-judge",
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
