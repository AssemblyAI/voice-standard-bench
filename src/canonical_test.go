package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCanonicalVersionDirName(t *testing.T) {
	assert.Equal(t, "0-1-2", CanonicalVersionDirName("0.1.2"))
}

// TestMergeJSONSamples_replacesByModelTaskAndBucket proves a rerun of one
// (model, task, bucket) replaces ONLY that bucket's samples -- the
// bucket-scoped counterpart to a -buckets run (see
// RunConfig.Buckets in framework.go), which must be able to update, say,
// just "3600s" without deleting "120s"'s already-merged data for the same
// model/task.
func TestMergeJSONSamples_replacesByModelTaskAndBucket(t *testing.T) {
	existing := []jsonSample{
		{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "old-1"},
		{Model: "model-a", Task: "summarization", Bucket: "120s", RecordID: "old-2"},
		{Model: "model-a", Task: "text_correction", Bucket: "30s", RecordID: "old-3"},
		{Model: "model-b", Task: "summarization", Bucket: "30s", RecordID: "old-4"},
	}
	incoming := []jsonSample{
		{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "new-1"},
	}

	merged := mergeJSONSamples(existing, incoming)

	var ids []string
	for _, s := range merged {
		ids = append(ids, s.RecordID)
	}

	// Only the model-a/summarization/30s sample is replaced. old-2 shares the
	// same model and task but a DIFFERENT bucket (120s) that incoming never
	// touched, so it must survive -- unlike the old (model, task)-only key,
	// which would have deleted it too.
	assert.ElementsMatch(t, []string{"old-2", "old-3", "old-4", "new-1"}, ids)
}

// TestMergeCells_replacesByModelTaskAndBucket mirrors
// TestMergeJSONSamples_replacesByModelTaskAndBucket for a report's
// Aggregates.
func TestMergeCells_replacesByModelTaskAndBucket(t *testing.T) {
	existing := []Cell{
		{Model: "model-a", Task: "summarization", Bucket: "30s", N: 1},
		{Model: "model-a", Task: "summarization", Bucket: "120s", N: 1},
		{Model: "model-b", Task: "summarization", Bucket: "30s", N: 1},
	}
	incoming := []Cell{
		{Model: "model-a", Task: "summarization", Bucket: "30s", N: 2},
	}

	merged := mergeCells(existing, incoming)

	assert.ElementsMatch(t, []Cell{
		{Model: "model-a", Task: "summarization", Bucket: "120s", N: 1},
		{Model: "model-b", Task: "summarization", Bucket: "30s", N: 1},
		{Model: "model-a", Task: "summarization", Bucket: "30s", N: 2},
	}, merged)
}

// TestMergeTTFTJSONSamples_replacesByModelTaskAndBucket mirrors
// TestMergeJSONSamples_replacesByModelTaskAndBucket for the TTFT path's
// sample type.
func TestMergeTTFTJSONSamples_replacesByModelTaskAndBucket(t *testing.T) {
	existing := []ttftJSONSample{
		{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "old-1"},
		{Model: "model-a", Task: "summarization", Bucket: "120s", RecordID: "old-2"},
	}
	incoming := []ttftJSONSample{
		{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "new-1"},
	}

	merged := mergeTTFTJSONSamples(existing, incoming)

	var ids []string
	for _, s := range merged {
		ids = append(ids, s.RecordID)
	}

	assert.ElementsMatch(t, []string{"old-2", "new-1"}, ids,
		"old-2's bucket (120s) was never in incoming, so it must survive")
}

// TestMergeTTFTCells_replacesByModelTaskAndBucket mirrors
// TestMergeCells_replacesByModelTaskAndBucket for the TTFT path's cell type.
func TestMergeTTFTCells_replacesByModelTaskAndBucket(t *testing.T) {
	existing := []TTFTCell{
		{Model: "model-a", Task: "summarization", Bucket: "30s", N: 1},
		{Model: "model-a", Task: "summarization", Bucket: "120s", N: 1},
	}
	incoming := []TTFTCell{
		{Model: "model-a", Task: "summarization", Bucket: "30s", N: 2},
	}

	merged := mergeTTFTCells(existing, incoming)

	assert.ElementsMatch(t, []TTFTCell{
		{Model: "model-a", Task: "summarization", Bucket: "120s", N: 1},
		{Model: "model-a", Task: "summarization", Bucket: "30s", N: 2},
	}, merged)
}

// TestMergeJSONSamples_reasoningEffortIsPartOfTheKey confirms re-appending
// the same (model, task) at a NEW reasoning effort adds a row instead of
// replacing the old effort's data -- the merge-time counterpart to
// Aggregate/summarize treating (model, reasoning effort) as the unique key
// (see gateway.go's ModelInfo.ReasoningEffortLabel).
func TestMergeJSONSamples_reasoningEffortIsPartOfTheKey(t *testing.T) {
	existing := []jsonSample{
		{Model: "model-a", ReasoningEffort: "default", Task: "summarization", Bucket: "30s", RecordID: "old-default"},
	}
	incoming := []jsonSample{
		{Model: "model-a", ReasoningEffort: "low", Task: "summarization", Bucket: "30s", RecordID: "new-low"},
	}

	merged := mergeJSONSamples(existing, incoming)

	var ids []string
	for _, s := range merged {
		ids = append(ids, s.RecordID)
	}

	assert.ElementsMatch(t, []string{"old-default", "new-low"}, ids,
		"a different reasoning effort must be a different row, not a replacement of the old effort's data")
}

// TestMergeCells_reasoningEffortIsPartOfTheKey mirrors
// TestMergeJSONSamples_reasoningEffortIsPartOfTheKey for a report's
// Aggregates.
func TestMergeCells_reasoningEffortIsPartOfTheKey(t *testing.T) {
	existing := []Cell{
		{Model: "model-a", ReasoningEffort: "default", Task: "summarization", Bucket: "30s", N: 1},
	}
	incoming := []Cell{
		{Model: "model-a", ReasoningEffort: "low", Task: "summarization", Bucket: "30s", N: 2},
	}

	merged := mergeCells(existing, incoming)

	assert.ElementsMatch(t, []Cell{
		{Model: "model-a", ReasoningEffort: "default", Task: "summarization", Bucket: "30s", N: 1},
		{Model: "model-a", ReasoningEffort: "low", Task: "summarization", Bucket: "30s", N: 2},
	}, merged)
}

// TestRunAppendCanonical_leavesUntouchedRowsByteIdentical is the regression test
// for the float-drift bug this file's package doc comment describes: an
// earlier version rebuilt every existing sample into a Sample (time.Duration
// latency), recomputed derived fields like TokensPerSecond from that, and
// re-ran Aggregate over the whole set on every append -- so a row nobody
// touched could still come out with different numbers than it went in with.
// model-b here is never mentioned by the second append; its sample and cell
// must survive with the exact float values (including a TokensPerSecond
// that a naive recompute from TotalLatencyS/CompletionTokens would NOT
// reproduce -- 100 tokens / 1.235s recomputes to 80.972, not 81.005) they
// were written with.
func TestRunAppendCanonical_leavesUntouchedRowsByteIdentical(t *testing.T) {
	dir := t.TempDir()
	canonicalRoot := filepath.Join(dir, "canonical")
	source1 := filepath.Join(dir, "run1")
	source2 := filepath.Join(dir, "run2")
	require.NoError(t, os.MkdirAll(source1, 0o777))
	require.NoError(t, os.MkdirAll(source2, 0o777))

	modelBSample := jsonSample{
		Model: "model-b", Task: "summarization", Bucket: "30s", RecordID: "r2",
		TotalLatencyS: 1.235, CompletionTokens: 100, TokensPerSecond: 81.005,
	}
	modelBCell := Cell{
		Model: "model-b", Task: "summarization", Bucket: "30s", N: 1,
		Latency:    Stats{N: 1, Min: 1.235, Max: 1.235, Mean: 1.235, P50: 1.235, P95: 1.235},
		Throughput: Stats{N: 1, Min: 81.005, Max: 81.005, Mean: 81.005, P50: 81.005, P95: 81.005},
		JudgeMeans: map[string]float64{"faithfulness": 4.667},
	}

	writeReport(t, filepath.Join(source1, "report-claude-opus-5.json"), jsonReport{
		Judge: "claude-opus-5",
		Samples: []jsonSample{
			{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1", TotalLatencyS: 1},
			modelBSample,
		},
		Aggregates: []Cell{
			{Model: "model-a", Task: "summarization", Bucket: "30s", N: 1},
			modelBCell,
		},
	})

	require.NoError(t, RunAppendCanonical(canonicalRoot, "0.1.2", source1))

	// Second append only mentions model-a; model-b must pass through
	// untouched.
	writeReport(t, filepath.Join(source2, "report-claude-opus-5.json"), jsonReport{
		Judge: "claude-opus-5",
		Samples: []jsonSample{
			{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r3", TotalLatencyS: 2},
		},
		Aggregates: []Cell{
			{Model: "model-a", Task: "summarization", Bucket: "30s", N: 2},
		},
	})

	require.NoError(t, RunAppendCanonical(canonicalRoot, "0.1.2", source2))

	canonicalPath := filepath.Join(canonicalRoot, "0-1-2", "report-claude-opus-5.json")

	var out jsonReport

	raw, err := os.ReadFile(canonicalPath)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(raw, &out))

	var ids []string
	for _, s := range out.Samples {
		ids = append(ids, s.RecordID)
	}

	assert.ElementsMatch(t, []string{"r2", "r3"}, ids, "model-a's old sample (r1) must be replaced by r3; model-b's (r2) must survive")
	assert.Contains(t, out.Samples, modelBSample, "model-b's sample must be byte-identical to what was appended in run1, not recomputed")
	assert.Contains(t, out.Aggregates, modelBCell, "model-b's cell must be byte-identical to what run1's own Aggregate() produced, not recomputed")
}

func TestRunAppendCanonical_mergesTTFTSeparatelyFromReports(t *testing.T) {
	dir := t.TempDir()
	canonicalRoot := filepath.Join(dir, "canonical")
	source := filepath.Join(dir, "run")
	require.NoError(t, os.MkdirAll(source, 0o777))

	writeReport(t, filepath.Join(source, "report-claude-opus-5.json"), jsonReport{
		Judge:   "claude-opus-5",
		Samples: []jsonSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1"}},
	})
	writeTTFTReport(t, filepath.Join(source, "ttft-report.json"), ttftJSONReport{
		Samples:    []ttftJSONSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "t1"}},
		Aggregates: []TTFTCell{{Model: "model-a", Task: "summarization", Bucket: "30s", N: 1}},
	})

	require.NoError(t, RunAppendCanonical(canonicalRoot, "0.1.2", source))

	versionDir := filepath.Join(canonicalRoot, "0-1-2")
	assert.FileExists(t, filepath.Join(versionDir, "report-claude-opus-5.json"))

	var ttftOut ttftJSONReport
	raw, err := os.ReadFile(filepath.Join(versionDir, "ttft-report.json"))
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(raw, &ttftOut))

	require.Len(t, ttftOut.Samples, 1)
	assert.Equal(t, "t1", ttftOut.Samples[0].RecordID)
	require.Len(t, ttftOut.Aggregates, 1)
	assert.Equal(t, "model-a", ttftOut.Aggregates[0].Model)
}

func TestRunAppendCanonical_errorsOnEmptySourceDir(t *testing.T) {
	dir := t.TempDir()
	err := RunAppendCanonical(filepath.Join(dir, "canonical"), "0.1.2", filepath.Join(dir, "empty-source"))
	assert.Error(t, err)
}

func TestRunAppendCanonical_errorsOnMissingJudgeField(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "run")
	require.NoError(t, os.MkdirAll(source, 0o777))

	writeReport(t, filepath.Join(source, "report-.json"), jsonReport{
		Samples: []jsonSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1"}},
	})

	err := RunAppendCanonical(filepath.Join(dir, "canonical"), "0.1.2", source)
	assert.ErrorContains(t, err, "judge")
}

func writeReport(t *testing.T, path string, report jsonReport) {
	t.Helper()
	data, err := json.Marshal(report)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
}

func writeTTFTReport(t *testing.T, path string, report ttftJSONReport) {
	t.Helper()
	data, err := json.Marshal(report)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
}

// withStdin swaps os.Stdin for a pipe pre-loaded with input, for confirmYes
// (which reads a line from it) to consume. Returns a restore func the caller
// must defer.
func withStdin(t *testing.T, input string) func() {
	t.Helper()

	r, w, err := os.Pipe()
	require.NoError(t, err)

	_, err = w.WriteString(input)
	require.NoError(t, err)
	require.NoError(t, w.Close())

	original := os.Stdin
	os.Stdin = r

	return func() {
		os.Stdin = original
		r.Close()
	}
}

func TestConfirmYes(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"exact yes", "yes\n", true},
		{"uppercase", "YES\n", true},
		{"surrounding whitespace", "  Yes  \n", true},
		{"no trailing newline (EOF)", "yes", true},
		{"abbreviation is not accepted", "y\n", false},
		{"bare enter", "\n", false},
		{"empty input (EOF immediately)", "", false},
		{"anything else", "sure\n", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer withStdin(t, tt.input)()
			assert.Equal(t, tt.want, confirmYes("merge? "))
		})
	}
}

func TestReviewAndMergeIntoCanonical_errorsWhenNothingStaged(t *testing.T) {
	dir := t.TempDir()
	err := ReviewAndMergeIntoCanonical(filepath.Join(dir, "canonical"), "0.1.2", "", "", true)
	assert.Error(t, err)
}

// TestReviewAndMergeIntoCanonical_autoApprove covers the end-to-end
// run-and-merge flow with -approve-merge set: both a staged
// report dir and a staged ttft dir get merged into canonical, and canonical's own
// summary.json/summary.csv are refreshed afterward (refreshCanonicalSummary) --
// the "using the summary code" step requested alongside the merge.
func TestReviewAndMergeIntoCanonical_autoApprove(t *testing.T) {
	dir := t.TempDir()
	canonicalRoot := filepath.Join(dir, "canonical")
	reportDir := filepath.Join(dir, "report-run")
	ttftDir := filepath.Join(dir, "ttft-run")
	require.NoError(t, os.MkdirAll(reportDir, 0o777))
	require.NoError(t, os.MkdirAll(ttftDir, 0o777))

	writeReport(t, filepath.Join(reportDir, "report-claude-opus-5.json"), jsonReport{
		Judge: "claude-opus-5",
		Samples: []jsonSample{
			{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1", Judge: map[string]float64{"faithfulness": 4}},
		},
		Aggregates: []Cell{
			{Model: "model-a", Task: "summarization", Bucket: "30s", N: 1},
		},
	})
	writeTTFTReport(t, filepath.Join(ttftDir, "ttft-report.json"), ttftJSONReport{
		Samples:    []ttftJSONSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "t1"}},
		Aggregates: []TTFTCell{{Model: "model-a", Task: "summarization", Bucket: "30s", N: 1}},
	})

	require.NoError(t, ReviewAndMergeIntoCanonical(canonicalRoot, "0.1.2", reportDir, ttftDir, true))

	versionDir := filepath.Join(canonicalRoot, "0-1-2")
	assert.FileExists(t, filepath.Join(versionDir, "report-claude-opus-5.json"))
	assert.FileExists(t, filepath.Join(versionDir, "ttft-report.json"))
	assert.FileExists(t, filepath.Join(versionDir, "summary.json"), "refreshCanonicalSummary must regenerate the canonical summary after a merge")
	assert.FileExists(t, filepath.Join(versionDir, "summary.csv"))
	assert.FileExists(t, filepath.Join(versionDir, "use_cases.json"), "the post-summary use-case step must run alongside every canonical summary refresh")

	// Staged directories are left in place -- ReviewAndMergeIntoCanonical never
	// deletes what it merged.
	assert.DirExists(t, reportDir)
	assert.DirExists(t, ttftDir)
}

func TestReviewAndMergeIntoCanonical_declineLeavesCanonicalUntouched(t *testing.T) {
	dir := t.TempDir()
	canonicalRoot := filepath.Join(dir, "canonical")
	reportDir := filepath.Join(dir, "report-run")
	require.NoError(t, os.MkdirAll(reportDir, 0o777))

	writeReport(t, filepath.Join(reportDir, "report-claude-opus-5.json"), jsonReport{
		Judge:   "claude-opus-5",
		Samples: []jsonSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1"}},
	})

	defer withStdin(t, "no\n")()

	require.NoError(t, ReviewAndMergeIntoCanonical(canonicalRoot, "0.1.2", reportDir, "", false))

	assert.NoDirExists(t, filepath.Join(canonicalRoot, "0-1-2"), "declining must not create or touch the canonical version dir at all")
	assert.FileExists(t, filepath.Join(reportDir, "report-claude-opus-5.json"), "staged data must survive a decline for a later manual merge")
}

func TestRefreshCanonicalSummary_skipsWhenNoReportsYet(t *testing.T) {
	dir := t.TempDir()
	versionDir := filepath.Join(dir, "0-1-2")
	require.NoError(t, os.MkdirAll(versionDir, 0o777))

	writeTTFTReport(t, filepath.Join(versionDir, "ttft-report.json"), ttftJSONReport{
		Samples: []ttftJSONSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "t1"}},
	})

	require.NoError(t, refreshCanonicalSummary(versionDir), "a TTFT-only canonical version has nothing to summarize and must not error")
	assert.NoFileExists(t, filepath.Join(versionDir, "summary.json"))
}

func TestRunRefreshCanonicalSummary(t *testing.T) {
	canonicalRoot := t.TempDir()
	versionDir := filepath.Join(canonicalRoot, "0-1-2")
	require.NoError(t, os.MkdirAll(versionDir, 0o777))

	writeReport(t, filepath.Join(versionDir, "report-claude-opus-5.json"), jsonReport{
		Judge:   "claude-opus-5",
		Samples: []jsonSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1", Judge: map[string]float64{"faithfulness": 4}}},
	})

	require.NoError(t, RunRefreshCanonicalSummary(canonicalRoot, "0.1.2"))

	assert.FileExists(t, filepath.Join(versionDir, "summary.json"))
	assert.FileExists(t, filepath.Join(versionDir, "use_cases.json"))
}

func TestRunRefreshCanonicalSummary_defaultsVersionWhenEmpty(t *testing.T) {
	canonicalRoot := t.TempDir()
	versionDir := filepath.Join(canonicalRoot, CanonicalVersionDirName(DefaultCanonicalVersion))
	require.NoError(t, os.MkdirAll(versionDir, 0o777))

	writeReport(t, filepath.Join(versionDir, "report-claude-opus-5.json"), jsonReport{
		Judge:   "claude-opus-5",
		Samples: []jsonSample{{Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1", Judge: map[string]float64{"faithfulness": 4}}},
	})

	require.NoError(t, RunRefreshCanonicalSummary(canonicalRoot, ""))
	assert.FileExists(t, filepath.Join(versionDir, "summary.json"))
}

func TestAnyFileMatches(t *testing.T) {
	dir := t.TempDir()

	ok, err := anyFileMatches(dir, "report-*.json")
	require.NoError(t, err)
	assert.False(t, ok)

	require.NoError(t, os.WriteFile(filepath.Join(dir, "report-x.json"), []byte("{}"), 0o644))

	ok, err = anyFileMatches(dir, "report-*.json")
	require.NoError(t, err)
	assert.True(t, ok)
}
