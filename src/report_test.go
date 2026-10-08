package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPrintReport_sectionsByTask locks down the one piece of real control
// flow in PrintReport: a task header is printed once per task, when the task
// changes — not once per cell, and not once total regardless of how many
// tasks are present. The exact column formatting is deliberately not
// asserted here; that's cosmetic and would make this test brittle for no
// correctness benefit.
func TestPrintReport_sectionsByTask(t *testing.T) {
	cells := []Cell{
		{Model: "a", Task: "summarization", Bucket: "30s", JudgeMeans: map[string]float64{}},
		{Model: "b", Task: "summarization", Bucket: "30s", JudgeMeans: map[string]float64{}},
		{Model: "a", Task: "text_correction", Bucket: "30s", JudgeMeans: map[string]float64{}},
	}

	var buf bytes.Buffer
	PrintReport(&buf, cells)

	output := buf.String()
	assert.Equal(t, 1, strings.Count(output, "=== summarization ==="), "one header for two summarization cells, not two")
	assert.Equal(t, 1, strings.Count(output, "=== text_correction ==="))
	assert.Contains(t, output, "a")
	assert.Contains(t, output, "b")
}

func TestPrintReport_doesNotPanicOnUnjudgedOrErroredCells(t *testing.T) {
	cells := []Cell{
		// No successful samples at all: every Stats is the N==0 zero value,
		// and JudgeMeans is empty -- both must render as "no data"/"not
		// judged", not panic on an empty map or a divide-by-zero Stats.
		{Model: "broken-model", Task: "summarization", Bucket: "30s", N: 3, Errors: 3, JudgeMeans: map[string]float64{}},
		{
			Model: "schema-failing-model", Task: "information_extraction", Bucket: "120s",
			N: 2, SchemaInvalid: 2, Judged: 2,
			JudgeMeans: map[string]float64{"groundedness": 1, "completeness": 1},
		},
	}

	assert.NotPanics(t, func() {
		var buf bytes.Buffer
		PrintReport(&buf, cells)
	})
}

func TestWriteJSONReport_roundTrips(t *testing.T) {
	samples := []Sample{
		{
			Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r1",
			TotalLatency: 2 * time.Second, TimeToFirstToken: 500 * time.Millisecond,
			CompletionTokens: 100, PromptTokens: 50, CostUSD: 0.00105,
			Judge: map[string]float64{"faithfulness": 4},
		},
		{
			Model: "model-a", Task: "summarization", Bucket: "30s", RecordID: "r2",
			Err: errors.New("rate limited"),
		},
	}
	cells := Aggregate(samples)

	dir := t.TempDir()
	reportDir, err := WriteJSONReport(dir, samples, cells, "")
	require.NoError(t, err)
	assert.DirExists(t, reportDir)

	// WriteJSONReport creates a fresh, uniquely-named subdirectory under dir
	// on every call (see its doc comment) rather than writing to dir itself,
	// so the report file has to be located rather than read back at a fixed
	// path -- reportDir (just asserted above) is that returned path, but
	// re-derive it via glob too so this test still catches a regression in
	// the returned value itself, not just in the file's actual location.
	matches, err := filepath.Glob(filepath.Join(dir, "*", "report-.json"))
	require.NoError(t, err)
	require.Len(t, matches, 1, "expected exactly one report file under a fresh run subdirectory")
	assert.Equal(t, reportDir, filepath.Dir(matches[0]), "returned path must be the same directory the file was actually written under")

	raw, err := os.ReadFile(matches[0])
	require.NoError(t, err)

	var parsed jsonReport
	require.NoError(t, json.Unmarshal(raw, &parsed), "output must be valid, re-parseable JSON")

	require.Len(t, parsed.Samples, 2)

	success := parsed.Samples[0]
	assert.Equal(t, "model-a", success.Model)
	assert.InDelta(t, 2.0, success.TotalLatencyS, 1e-9)
	assert.InDelta(t, 50.0, success.TokensPerSecond, 1e-9) // 100 tokens / 2s
	assert.InDelta(t, 0.00105, success.CostUSD, 1e-9)
	assert.Empty(t, success.Error)

	failed := parsed.Samples[1]
	assert.Equal(t, "rate limited", failed.Error, "a failed sample's error must be preserved as a plain string")
	assert.Equal(t, 0.0, failed.TotalLatencyS)

	require.Len(t, parsed.Aggregates, 1)
	assert.Equal(t, 1, parsed.Aggregates[0].Errors)
}
