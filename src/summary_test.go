package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		name      string
		dimension string
		value     float64
		want      float64
		wantOK    bool
	}{
		{"likert worst", "faithfulness", 1, 0, true},
		{"likert best", "coverage", 5, 1, true},
		{"likert midpoint", "improvement", 3, 0.5, true},
		{"exact_match is a plain 0/1", "exact_match", 1, 1, true},
		{"exact_match false", "exact_match", 0, 0, true},
		{"punctuation_accuracy is a 0-100 percentage", "punctuation_accuracy", 87.5, 0.875, true},
		{"capitalization_accuracy is a 0-100 percentage", "capitalization_accuracy", 100, 1, true},
		{"formatting_command_accuracy is a 0-100 percentage", "formatting_command_accuracy", 0, 0, true},
		{"word_error_rate is inverted: high error is low quality", "word_error_rate", 20, 0.8, true},
		{"word_error_rate of zero is perfect quality", "word_error_rate", 0, 1, true},
		{"word_error_rate over 100 (more insertions than truth words) clamps to zero, not negative", "word_error_rate", 150, 0, true},
		{"unknown dimension is excluded, not guessed", "notes", 5, 0, false},
		// Everything text_correction once scored against the dataset's original
		// transcript -- which turned out not to be an accurate answer key -- is
		// retired, so an older report's copy of it must not count either way.
		{"the retired meaning_similarity no longer counts", "meaning_similarity", 4, 0, false},
		{"the retired truth_wer no longer counts", "truth_wer", 4.2, 0, false},
		{"the retired input_wer no longer counts", "input_wer", 4.8, 0, false},
		{"the retired word-overlap accuracy no longer counts", "accuracy", 0.99, 0, false},
		{"the retired edit_ratio no longer counts", "edit_ratio", 0, 0, false},
		{"the retired error_reduction no longer counts", "error_reduction", 0.5, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := normalize(tt.dimension, tt.value)
			require.Equal(t, tt.wantOK, ok)
			assert.InDelta(t, tt.want, got, 1e-9)
		})
	}
}

func TestSampleQuality(t *testing.T) {
	tests := []struct {
		name   string
		judge  map[string]float64
		want   float64
		wantOK bool
	}{
		{
			name:   "nil judge (failed or unjudged sample)",
			judge:  nil,
			wantOK: false,
		},
		{
			name:   "three likert dims average before the group does",
			judge:  map[string]float64{"faithfulness": 5, "coverage": 5, "conciseness": 3},
			want:   (1 + 1 + 0.5) / 3,
			wantOK: true,
		},
		{
			name:   "correction averages its two reference-free judge dims",
			judge:  map[string]float64{"faithfulness": 3, "improvement": 5},
			want:   (0.5 + 1) / 2,
			wantOK: true,
		},
		{
			name:   "scores retired for comparing against the original transcript never enter the average",
			judge:  map[string]float64{"faithfulness": 3, "improvement": 5, "meaning_similarity": 4, "truth_wer": 1.2, "input_wer": 4.3, "accuracy": 0.99, "edit_ratio": 0, "error_reduction": 0.5},
			want:   (0.5 + 1) / 2,
			wantOK: true,
		},
		{
			name:   "a report that only carries retired scores has nothing left to score",
			judge:  map[string]float64{"accuracy": 0.99, "edit_ratio": 0, "error_reduction": 0.5},
			wantOK: false,
		},
		{
			name: "dictation mixes five mechanical dims, none of them Likert",
			judge: map[string]float64{
				"exact_match": 0, "word_error_rate": 10,
				"punctuation_accuracy": 100, "capitalization_accuracy": 50, "formatting_command_accuracy": 100,
			},
			// exact_match=0 -> 0; word_error_rate=10 -> 1-0.1=0.9;
			// punctuation_accuracy=100 -> 1; capitalization_accuracy=50 -> 0.5;
			// formatting_command_accuracy=100 -> 1.
			want:   (0 + 0.9 + 1 + 0.5 + 1) / 5,
			wantOK: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := sampleQuality(tt.judge)
			require.Equal(t, tt.wantOK, ok)
			assert.InDelta(t, tt.want, got, 1e-9)
		})
	}
}

func TestSummaryComputeStats(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   stats
	}{
		{
			name:   "empty input returns the zero value",
			values: nil,
			want:   stats{},
		},
		{
			name:   "single value: every stat equals it",
			values: []float64{42},
			want:   stats{N: 1, Min: 42, Max: 42, Mean: 42, P50: 42, P95: 42},
		},
		{
			name:   "five values, unsorted input",
			values: []float64{5, 1, 4, 2, 3},
			// sorted: [1,2,3,4,5]; nearest-rank: p50 -> ceil(0.5*5)-1=2 -> 3,
			// p95 -> ceil(0.95*5)-1=4 -> 5.
			want: stats{N: 5, Min: 1, Max: 5, Mean: 3, P50: 3, P95: 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeStats(tt.values)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestSummarize(t *testing.T) {
	samples := []sourceSample{
		// model "b", bucket "30s": two perfect judged samples.
		{Model: "b", Task: "summarization", Bucket: "30s", TotalLatencyS: 1, TokensPerSecond: 100, Judge: map[string]float64{"faithfulness": 5, "coverage": 5}},
		{Model: "b", Task: "summarization", Bucket: "30s", TotalLatencyS: 3, TokensPerSecond: 200, Judge: map[string]float64{"faithfulness": 5, "coverage": 5}},
		// model "a", bucket "120s": one judged success, one hard failure
		// (no judge, zeroed latency/throughput — must not pollute the means).
		{Model: "a", Task: "summarization", Bucket: "120s", TotalLatencyS: 2, TokensPerSecond: 50, Judge: map[string]float64{"faithfulness": 1, "coverage": 1}},
		{Model: "a", Task: "summarization", Bucket: "120s", Error: "boom"},
	}

	entries := summarize(samples)
	require.Len(t, entries, 2, "one entry per distinct (model, bucket)")

	// Sort order: model "a" before "b".
	assert.Equal(t, "a", entries[0].Model)
	assert.Equal(t, "120s", entries[0].Bucket)
	assert.Equal(t, "b", entries[1].Model)
	assert.Equal(t, "30s", entries[1].Bucket)

	a120 := entries[0]
	require.Equal(t, 1, a120.Latency.N, "the failed sample must not contribute a latency data point")
	assert.Equal(t, 1, a120.ErrorCount, "the one hard-failure sample must be tallied")
	assert.InDelta(t, 2.0, a120.Latency.Mean, 1e-9)
	assert.InDelta(t, 50.0, a120.TokensPerSecond, 1e-9)
	assert.Equal(t, 1, a120.QualityScore, "worst score on every dimension maps to the bottom of the 1-100 range")
	assert.Equal(t, map[string]int{"summarization": 1}, a120.TaskQuality, "with only one task present, its own score matches the combined one")

	b30 := entries[1]
	require.Equal(t, 2, b30.Latency.N)
	assert.Equal(t, 0, b30.ErrorCount, "no failed samples in this group")
	assert.InDelta(t, 1.0, b30.Latency.Min, 1e-9)
	assert.InDelta(t, 3.0, b30.Latency.Max, 1e-9)
	assert.InDelta(t, 2.0, b30.Latency.Mean, 1e-9)
	assert.InDelta(t, 150.0, b30.TokensPerSecond, 1e-9)
	assert.Equal(t, 100, b30.QualityScore, "perfect score on every dimension maps to the top of the 1-100 range")
	assert.Equal(t, map[string]int{"summarization": 100}, b30.TaskQuality)
}

// TestSummarize_reasoningEffortIsPartOfTheKey confirms the same model
// benchmarked at two different reasoning efforts in the same bucket produces
// two distinct rows, each carrying its own ReasoningEffort label, rather
// than pooling into one row that loses which effort produced which numbers.
func TestSummarize_reasoningEffortIsPartOfTheKey(t *testing.T) {
	samples := []sourceSample{
		{Model: "m", ReasoningEffort: "low", Task: "summarization", Bucket: "30s", TotalLatencyS: 1},
		{Model: "m", ReasoningEffort: "high", Task: "summarization", Bucket: "30s", TotalLatencyS: 9},
	}

	entries := summarize(samples)
	require.Len(t, entries, 2, "reasoning effort must be part of the grouping key, not just (model, bucket)")

	byEffort := map[string]summaryEntry{entries[0].ReasoningEffort: entries[0], entries[1].ReasoningEffort: entries[1]}

	require.Contains(t, byEffort, "low")
	require.Contains(t, byEffort, "high")
	assert.InDelta(t, 1.0, byEffort["low"].Latency.Mean, 1e-9)
	assert.InDelta(t, 9.0, byEffort["high"].Latency.Mean, 1e-9, "a blended row would average to 5.0 instead")
}

// TestSummarize_taskQualityIsUndilutedByOtherTasks is the core test for
// task_quality: two tasks sharing the SAME (model, bucket) must each keep
// their own score in TaskQuality even though they're pooled together into
// one QualityScore -- proving the per-task breakdown is genuinely a second,
// independent computation, not just a copy of the combined number.
func TestSummarize_taskQualityIsUndilutedByOtherTasks(t *testing.T) {
	samples := []sourceSample{
		// summarization: two perfect samples.
		{Model: "m", Task: "summarization", Bucket: "30s", Judge: map[string]float64{"faithfulness": 5, "coverage": 5}},
		{Model: "m", Task: "summarization", Bucket: "30s", Judge: map[string]float64{"faithfulness": 5, "coverage": 5}},
		// text_correction, same model+bucket: one worst-possible sample. If
		// this leaked into summarization's own score, it would drag it down.
		{Model: "m", Task: "text_correction", Bucket: "30s", Judge: map[string]float64{"faithfulness": 1, "improvement": 1}},
		// information_extraction never judged (task call itself failed) --
		// must be entirely absent from TaskQuality, not a 0 entry.
		{Model: "m", Task: "information_extraction", Bucket: "30s", Error: "boom"},
	}

	entries := summarize(samples)
	require.Len(t, entries, 1, "every sample shares the same (model, bucket), so they all pool into one row")

	entry := entries[0]
	// Each task weighs equally regardless of sample count: summarization's
	// mean is 1.0 and text_correction's is 0, so the combined mean is 0.5 ->
	// round(1 + 0.5*99) = round(50.5) = 51 (a per-sample pool would give 67).
	assert.Equal(t, 51, entry.QualityScore, "each task counts equally in the combined score, not each sample")
	assert.Equal(t, map[string]int{"summarization": 100, "text_correction": 1}, entry.TaskQuality,
		"each task keeps its OWN undiluted score; a task with no judged sample in this row has no key at all")
}

// TestSummarize_tasksWeightedEquallyAndEmptyTasksExcluded proves the combined
// QualityScore averages per-task means (so a task with many samples doesn't
// outweigh one with few) and that a task with no judged samples at all is
// excluded from the average rather than dragging it toward 0.
func TestSummarize_tasksWeightedEquallyAndEmptyTasksExcluded(t *testing.T) {
	perfect := map[string]float64{"faithfulness": 5, "coverage": 5}

	samples := []sourceSample{
		// summarization: 4 perfect samples.
		{Model: "m", Task: "summarization", Bucket: "30s", Judge: perfect},
		{Model: "m", Task: "summarization", Bucket: "30s", Judge: perfect},
		{Model: "m", Task: "summarization", Bucket: "30s", Judge: perfect},
		{Model: "m", Task: "summarization", Bucket: "30s", Judge: perfect},
		// text_correction: a single worst sample -- same weight as all four above.
		{Model: "m", Task: "text_correction", Bucket: "30s", Judge: map[string]float64{"faithfulness": 1, "improvement": 1}},
		// information_extraction: never judged -- must not count as a 0 task.
		{Model: "m", Task: "information_extraction", Bucket: "30s", Error: "boom"},
	}

	entries := summarize(samples)
	require.Len(t, entries, 1)
	assert.Equal(t, 51, entries[0].QualityScore, "mean of 100% and 0% over the two judged tasks only")
}

// TestSummarize_costPerQuality locks in CostPerQuality/TaskCostPerQuality's
// arithmetic -- the MEAN cost of one judged sample over the score, not the
// total over the score -- and, critically, that CostUSD is only ever
// accumulated over the SAME population as quality (judged samples): the
// errored sample below carries a deliberately huge CostUSD that must never
// reach TotalCostUSD or the mean.
func TestSummarize_costPerQuality(t *testing.T) {
	samples := []sourceSample{
		// summarization: two perfect samples, $0.01 and $0.02.
		{Model: "m", Task: "summarization", Bucket: "30s", CostUSD: 0.01, Judge: map[string]float64{"faithfulness": 5, "coverage": 5}},
		{Model: "m", Task: "summarization", Bucket: "30s", CostUSD: 0.02, Judge: map[string]float64{"faithfulness": 5, "coverage": 5}},
		// text_correction: one worst-possible sample, $0.05.
		{Model: "m", Task: "text_correction", Bucket: "30s", CostUSD: 0.05, Judge: map[string]float64{"faithfulness": 1, "improvement": 1}},
		// information_extraction: task call itself failed -- never judged, so
		// this $999 must be excluded from every cost total, not just diluted.
		{Model: "m", Task: "information_extraction", Bucket: "30s", CostUSD: 999, Error: "boom"},
	}

	entries := summarize(samples)
	require.Len(t, entries, 1)

	entry := entries[0]
	require.Equal(t, 51, entry.QualityScore, "same combined score as TestSummarize_taskQualityIsUndilutedByOtherTasks")

	assert.InDelta(t, 0.08, entry.TotalCostUSD, 1e-9, "0.01+0.02+0.05 -- the $999 errored sample must never contribute")
	assert.InDelta(t, 0.05229, entry.CostPerQuality, 1e-12, "(8 cents over the 3 judged samples) / 51, to 4 significant digits")

	require.Len(t, entry.TaskCostUSD, 2, "information_extraction contributed no judged sample, so it has no key at all")
	assert.InDelta(t, 0.03, entry.TaskCostUSD["summarization"], 1e-9, "the per-task TOTAL stays a total")
	assert.InDelta(t, 0.05, entry.TaskCostUSD["text_correction"], 1e-9)
	assert.InDelta(t, (3.0/2)/100, entry.TaskCostPerQuality["summarization"], 1e-12, "summarization's own mean over its 2 samples, its own perfect score, not the diluted combined one")
	assert.InDelta(t, (5.0/1)/1, entry.TaskCostPerQuality["text_correction"], 1e-12, "text_correction's own single sample, its own worst score")
}

// TestSummarize_costPerHour locks in the arithmetic: the model's hourly tokens
// are the mean of its 3600s-bucket samples (errored ones excluded), and every
// row -- any bucket, and each task within it -- is priced at its own blended
// cost per token times those hourly tokens.
func TestSummarize_costPerHour(t *testing.T) {
	judge := map[string]float64{"faithfulness": 5, "coverage": 5}
	samples := []sourceSample{
		// 3600s: mean prompt 3000, mean completion 1000 -> 4000 hourly tokens.
		{Model: "m", Task: "summarization", Bucket: "3600s", PromptTokens: 2000, CompletionTokens: 500, CostUSD: 1, Judge: judge},
		{Model: "m", Task: "summarization", Bucket: "3600s", PromptTokens: 4000, CompletionTokens: 1500, CostUSD: 1, Judge: judge},
		{Model: "m", Task: "summarization", Bucket: "3600s", PromptTokens: 999999, CompletionTokens: 999999, CostUSD: 999, Error: "boom"},
		// 30s: 1000 tokens for $0.10 across two tasks.
		{Model: "m", Task: "summarization", Bucket: "30s", PromptTokens: 400, CompletionTokens: 100, CostUSD: 0.05, Judge: judge},
		{Model: "m", Task: "text_correction", Bucket: "30s", PromptTokens: 250, CompletionTokens: 250, CostUSD: 0.05, Judge: map[string]float64{"faithfulness": 5, "improvement": 5}},
		// Another model with no 3600s samples has no hourly basis at all.
		{Model: "other", Task: "summarization", Bucket: "30s", PromptTokens: 100, CompletionTokens: 100, CostUSD: 1, Judge: judge},
	}

	byKey := map[string]summaryEntry{}
	for _, e := range summarize(samples) {
		byKey[e.Model+"/"+e.Bucket] = e
	}

	hour := byKey["m/3600s"]
	assert.Equal(t, 3000, hour.HourlyPromptTokens)
	assert.Equal(t, 1000, hour.HourlyCompletionTokens)
	assert.InDelta(t, 1.0, hour.CostPerHourUSD, 1e-9, "$2 over its 8000 judged tokens, times 4000 hourly tokens")

	short := byKey["m/30s"]
	assert.Equal(t, 3000, short.HourlyPromptTokens, "every row of a model shows the same hourly basis")
	assert.InDelta(t, 0.1/1000*4000, short.CostPerHourUSD, 1e-9, "$0.10 over 1000 tokens, scaled to 4000")
	assert.InDelta(t, 0.05/500*4000, short.TaskCostPerHourUSD["summarization"], 1e-9)
	assert.InDelta(t, 0.05/500*4000, short.TaskCostPerHourUSD["text_correction"], 1e-9)

	other := byKey["other/30s"]
	assert.Zero(t, other.HourlyPromptTokens)
	assert.Zero(t, other.CostPerHourUSD, "no 3600s samples, no hourly basis, no cost per hour")
	assert.Empty(t, other.TaskCostPerHourUSD)
}

// TestSummarize_costPerHourKeepsReasoningEffortsApart: the same model at two
// efforts has different completion tokens, so each gets its own hourly basis.
func TestSummarize_costPerHourKeepsReasoningEffortsApart(t *testing.T) {
	judge := map[string]float64{"faithfulness": 5, "coverage": 5}
	samples := []sourceSample{
		{Model: "m", ReasoningEffort: "low", Task: "summarization", Bucket: "3600s", PromptTokens: 1000, CompletionTokens: 100, CostUSD: 1, Judge: judge},
		{Model: "m", ReasoningEffort: "high", Task: "summarization", Bucket: "3600s", PromptTokens: 1000, CompletionTokens: 900, CostUSD: 1, Judge: judge},
	}

	got := map[string]int{}
	for _, e := range summarize(samples) {
		got[e.ReasoningEffort] = e.HourlyCompletionTokens
	}

	assert.Equal(t, map[string]int{"low": 100, "high": 900}, got)
}

// TestSummarize_costPerHourOmittedForZeroCost: an unpriced model has a zero
// cost per hour, which is omitted rather than reported as a real $0.
func TestSummarize_costPerHourOmittedForZeroCost(t *testing.T) {
	judge := map[string]float64{"faithfulness": 5, "coverage": 5}
	entries := summarize([]sourceSample{
		{Model: "m", Task: "summarization", Bucket: "3600s", PromptTokens: 1000, CompletionTokens: 100, Judge: judge},
	})

	require.Len(t, entries, 1)
	assert.Equal(t, 1000, entries[0].HourlyPromptTokens)
	assert.Zero(t, entries[0].CostPerHourUSD)
}

// TestSummarize_costPerQualityIsIndependentOfSampleCount is the regression
// test for cost_per_quality being a TOTAL over the score: two models with the
// exact same per-call cost and the exact same quality must get the exact same
// cost_per_quality however many samples their rows happen to hold (a run with
// a smaller -max-samples-per-bucket, or a model whose calls errored more
// often and so judged fewer of them). TotalCostUSD is still a true total, so
// it must keep scaling with the sample count.
func TestSummarize_costPerQualityIsIndependentOfSampleCount(t *testing.T) {
	perCall := func(model string, n int) []sourceSample {
		out := make([]sourceSample, 0, n)
		for i := 0; i < n; i++ {
			out = append(out, sourceSample{Model: model, Task: "summarization", Bucket: "30s", CostUSD: 0.01,
				Judge: map[string]float64{"faithfulness": 5, "coverage": 5, "conciseness": 5}})
		}

		return out
	}

	entries := summarize(append(perCall("many", 30), perCall("few", 10)...))
	require.Len(t, entries, 2)

	byModel := map[string]summaryEntry{}
	for _, e := range entries {
		byModel[e.Model] = e
	}

	many, few := byModel["many"], byModel["few"]

	require.Equal(t, 100, many.QualityScore)
	require.Equal(t, 100, few.QualityScore)

	assert.InDelta(t, 0.30, many.TotalCostUSD, 1e-9)
	assert.InDelta(t, 0.10, few.TotalCostUSD, 1e-9, "the total honestly reflects fewer samples")

	assert.InDelta(t, 0.01, many.CostPerQuality, 1e-12, "1 cent per call / quality 100")
	assert.Equal(t, many.CostPerQuality, few.CostPerQuality, "same per-call cost and quality, so the same cost_per_quality at any sample count")
	assert.Equal(t, many.TaskCostPerQuality["summarization"], few.TaskCostPerQuality["summarization"])
}

// TestSummarize_costPerQualityKeepsSignificantDigitsForCheapCalls pins down
// why CostPerQuality is rounded to significant digits rather than six decimal
// places: a $0.0000503 call at quality 100 is 5.03e-05 cents per quality point,
// which six decimals would flatten to 1e-06 (and a cheaper call to a bare 0,
// which every consumer reads as "no cost data").
func TestSummarize_costPerQualityKeepsSignificantDigitsForCheapCalls(t *testing.T) {
	judge := map[string]float64{"faithfulness": 5, "coverage": 5, "conciseness": 5}
	samples := []sourceSample{
		{Model: "cheap", Task: "summarization", Bucket: "30s", CostUSD: 0.0000503, Judge: judge},
		{Model: "cheaper", Task: "summarization", Bucket: "30s", CostUSD: 0.0000003, Judge: judge},
	}

	byModel := map[string]summaryEntry{}
	for _, e := range summarize(samples) {
		byModel[e.Model] = e
	}

	assert.InDelta(t, 5.03e-05, byModel["cheap"].CostPerQuality, 1e-13)
	assert.InDelta(t, 3e-07, byModel["cheaper"].CostPerQuality, 1e-15, "must not round away to 0, which would read as unpriced")
	assert.InDelta(t, 5.03e-05, byModel["cheap"].TaskCostPerQuality["summarization"], 1e-13)
	assert.Less(t, byModel["cheaper"].CostPerQuality, byModel["cheap"].CostPerQuality, "the ordering between cheap models survives")
}

// TestSummarize_zeroCostDataYieldsZeroCostPerQuality proves a row with
// judged samples but every CostUSD left at its zero value (e.g. a report
// written before cost tracking existed) resolves to a real $0 rather than
// panicking or dividing by zero -- QualityScore is never 0 once judged (see
// summarize's doc comment), so the division is always by a real number.
func TestSummarize_zeroCostDataYieldsZeroCostPerQuality(t *testing.T) {
	samples := []sourceSample{
		{Model: "m", Task: "summarization", Bucket: "30s", Judge: map[string]float64{"faithfulness": 5}},
	}

	entries := summarize(samples)
	require.Len(t, entries, 1)

	assert.Equal(t, 100, entries[0].QualityScore)
	assert.Zero(t, entries[0].TotalCostUSD)
	assert.Zero(t, entries[0].CostPerQuality)
}

func TestSummarize_bucketDurationOrder(t *testing.T) {
	samples := []sourceSample{
		{Model: "m", Bucket: "900s", TotalLatencyS: 1, TokensPerSecond: 1},
		{Model: "m", Bucket: "30s", TotalLatencyS: 1, TokensPerSecond: 1},
		{Model: "m", Bucket: "120s", TotalLatencyS: 1, TokensPerSecond: 1},
	}

	entries := summarize(samples)
	require.Len(t, entries, 3)

	// Lexical order would put "120s" before "30s" before "900s"; duration
	// order (what a report about scaling with audio length needs) must not.
	assert.Equal(t, []string{"30s", "120s", "900s"}, []string{entries[0].Bucket, entries[1].Bucket, entries[2].Bucket})
}

func TestJudgeFor(t *testing.T) {
	tests := []struct {
		name   string
		path   string
		report sourceReport
		want   string
	}{
		{
			name:   "the report's own judge field wins, filename suffix or not",
			path:   "report-2026-08-27T18:14:04-06:00-gpt-5.6-sol.json",
			report: sourceReport{Judge: "gpt-5.6-sol"},
			want:   "gpt-5.6-sol",
		},
		{
			name: "falls back to a -<judge> filename suffix when the field is empty",
			path: "report-2026-08-27T18:06:45-06:00-claude-opus-5.json",
			want: "claude-opus-5",
		},
		{
			name: "a plain report-<timestamp>.json with neither is unknown, not guessed",
			path: "report-2026-08-26T14:05:32-06:00.json",
			want: unknownJudge,
		},
		{
			name: "a filename that doesn't match the report- pattern at all is unknown",
			path: "notes.json",
			want: unknownJudge,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, judgeFor(tt.path, tt.report))
		})
	}
}

func TestLoadReports(t *testing.T) {
	dir := t.TempDir()

	writeSummaryReport(t, dir, "report-2026-08-26T10:00:00Z.json", sourceReport{
		Samples: []sourceSample{{Model: "gemini", Bucket: "30s", TotalLatencyS: 1, Judge: map[string]float64{"faithfulness": 5}}},
	})
	writeSummaryReport(t, dir, "report-2026-08-27T18:14:04-06:00-gpt-5.6-sol.json", sourceReport{
		Judge: "gpt-5.6-sol",
		Samples: []sourceSample{
			{Model: "gemini", Bucket: "30s", TotalLatencyS: 3, Judge: map[string]float64{"faithfulness": 3}},
			// A judge grading itself can't happen from a real run (see
			// voicebench.go); simulate a stale/hand-edited file and confirm
			// it's dropped rather than silently counted.
			{Model: "gpt-5.6-sol", Bucket: "30s", TotalLatencyS: 9, Judge: map[string]float64{"faithfulness": 5}},
		},
	})
	// A non-report file in the same directory must be ignored.
	require.NoError(t, os.WriteFile(filepath.Join(dir, "summary-2026-08-26T10:00:00Z.json"), []byte(`{}`), 0o644))

	names, samples, err := loadReports(dir)
	require.NoError(t, err)
	assert.Equal(t, []string{"report-2026-08-26T10:00:00Z.json", "report-2026-08-27T18:14:04-06:00-gpt-5.6-sol.json"}, names)

	require.Len(t, samples, 2, "the self-graded sample must be dropped")
	assert.Equal(t, unknownJudge, samples[0].judge)
	assert.Equal(t, "gpt-5.6-sol", samples[1].judge)
	assert.Equal(t, "gemini", samples[1].Model)
}

func TestLoadReports_noMatchingFiles(t *testing.T) {
	_, _, err := loadReports(t.TempDir())
	require.Error(t, err)
}

func TestSummarizeByJudge(t *testing.T) {
	samples := []judgedSample{
		{judge: "claude-opus-5", sourceSample: sourceSample{Model: "gemini", Bucket: "30s", TotalLatencyS: 1, Judge: map[string]float64{"faithfulness": 5}}},
		{judge: "claude-opus-5", sourceSample: sourceSample{Model: "gemini", Bucket: "30s", TotalLatencyS: 3, Judge: map[string]float64{"faithfulness": 3}}},
		{judge: "gpt-5.6-sol", sourceSample: sourceSample{Model: "gemini", Bucket: "30s", TotalLatencyS: 5, Judge: map[string]float64{"faithfulness": 1}}},
	}

	byJudge := summarizeByJudge(samples)
	require.Len(t, byJudge, 2)

	// Sorted by judge name: "claude-opus-5" before "gpt-5.6-sol".
	assert.Equal(t, "claude-opus-5", byJudge[0].Judge)
	require.Len(t, byJudge[0].Summary, 1)
	assert.Equal(t, 2, byJudge[0].Summary[0].Latency.N, "both claude-opus-5 samples pool into one entry")

	assert.Equal(t, "gpt-5.6-sol", byJudge[1].Judge)
	require.Len(t, byJudge[1].Summary, 1)
	assert.Equal(t, 1, byJudge[1].Summary[0].Latency.N)
}

// TestPrintTable_sectionsByJudge locks down the real control flow — one
// header per judge, plus a final combined section — the same way
// TestPrintReport_sectionsByTask does for report.go's PrintReport. Exact
// column formatting is deliberately not asserted; that's cosmetic.
func TestPrintTable_sectionsByJudge(t *testing.T) {
	out := summaryReport{
		ByJudge: []judgeSummary{
			{Judge: "claude-opus-5", Summary: []summaryEntry{{Model: "gemini", Bucket: "30s", QualityScore: 83}}},
			{Judge: "gpt-5.6-sol", Summary: []summaryEntry{{Model: "gemini", Bucket: "30s", QualityScore: 69}}},
		},
		Combined: []summaryEntry{{Model: "gemini", Bucket: "30s", QualityScore: 76}},
	}

	var buf bytes.Buffer
	printTable(&buf, out)

	output := buf.String()
	assert.Equal(t, 1, strings.Count(output, "=== judge: claude-opus-5 ==="))
	assert.Equal(t, 1, strings.Count(output, "=== judge: gpt-5.6-sol ==="))
	assert.Equal(t, 1, strings.Count(output, "=== combined (all judges) ==="))

	// The combined section must come after both judge sections.
	assert.Less(t, strings.Index(output, "gpt-5.6-sol"), strings.Index(output, "=== combined"))
	assert.Contains(t, output, "83")
	assert.Contains(t, output, "69")
	assert.Contains(t, output, "76")
}

func TestPrintTable_doesNotPanicOnNoData(t *testing.T) {
	out := summaryReport{
		ByJudge:  []judgeSummary{{Judge: "j", Summary: []summaryEntry{{Model: "m", Bucket: "30s"}}}},
		Combined: []summaryEntry{{Model: "m", Bucket: "30s"}},
	}

	assert.NotPanics(t, func() {
		var buf bytes.Buffer
		printTable(&buf, out)
		assert.Contains(t, buf.String(), "n/a")
		assert.Contains(t, buf.String(), "no data")
	})
}

func TestWriteCSV(t *testing.T) {
	out := summaryReport{
		ByJudge: []judgeSummary{
			{Judge: "claude-opus-5", Summary: []summaryEntry{
				{
					Model: "gemini", Bucket: "30s", QualityScore: 83,
					TotalCostUSD: 0.083, CostPerQuality: 0.001,
					HourlyPromptTokens: 12000, HourlyCompletionTokens: 3000, CostPerHourUSD: 1.25,
					TaskCostPerHourUSD: map[string]float64{"summarization": 1.5, "text_correction": 0.75},
					Latency:            stats{N: 4, Min: 1, Max: 3, Mean: 2, P50: 1.5, P95: 2.5}, ErrorCount: 2, TokensPerSecond: 117.3,
					TTFT:               stats{N: 4, Min: 0.1, Max: 0.3, Mean: 0.2, P50: 0.2, P95: 0.25},
					TaskQuality:        map[string]int{"summarization": 90, "text_correction": 76},
					TaskCostUSD:        map[string]float64{"summarization": 0.18, "text_correction": 0.038},
					TaskCostPerQuality: map[string]float64{"summarization": 0.002, "text_correction": 0.0005},
				},
			}},
		},
		// No judged samples, no successful task calls, no TTFT data, and no
		// per-task breakdown at all for this (model, bucket): every field
		// must render as a blank cell, not "0" -- including a task column
		// that DOES exist (because the other row above has it) but this row
		// has no value for -- except error_count, which is always a real
		// number (this row's zero-value ErrorCount genuinely means zero
		// errors, not "no data").
		Combined: []summaryEntry{{Model: "broken", Bucket: "30s"}},
	}

	path := filepath.Join(t.TempDir(), "summary.csv")
	require.NoError(t, writeCSV(path, out))

	f, err := os.Open(path)
	require.NoError(t, err)
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	require.NoError(t, err)
	require.Len(t, rows, 3, "header + one by-judge row + one combined row")

	wantHeader := append(append([]string{}, csvHeader...),
		"quality_summarization", "quality_text_correction",
		"cost_usd_summarization", "cost_usd_text_correction",
		"cost_per_quality_summarization", "cost_per_quality_text_correction",
		"cost_per_hour_usd_summarization", "cost_per_hour_usd_text_correction",
	)
	require.Len(t, rows[1], len(wantHeader), "one cell per header column, including the discovered task columns")
	require.Len(t, rows[2], len(wantHeader))

	assert.Equal(t, wantHeader, rows[0], "quality_<task> columns are appended in sorted task-name order")
	assert.Equal(t, []string{
		"claude-opus-5", "gemini", "", "30s", "83", "0.083", "0.001",
		"12000", "3000", "1.25",
		"4", "2", "1", "2", "1.5", "2.5", "3", "117.3",
		"4", "0.1", "0.2", "0.2", "0.25", "0.3",
		"90", "76",
		"0.18", "0.038",
		"0.002", "0.0005",
		"1.5", "0.75",
	}, rows[1])

	// error_count is the one exception to the blank-for-zero convention: this
	// row has zero errors (a real, meaningful count), not "no data", so it
	// must render as a literal "0" even though every other cell here is
	// blank -- see csvRow.
	wantBrokenRow := []string{"combined", "broken", "", "30s", "", "", "", "", "", "", "", "0"}
	wantBrokenRow = append(wantBrokenRow, make([]string, len(wantHeader)-len(wantBrokenRow))...)
	assert.Equal(t, wantBrokenRow, rows[2],
		"no data anywhere -- including no per-task breakdown for this (model, bucket) -- must render as blank cells, except error_count which is always a real number")
}

// TestTaskColumns locks in the column-discovery behavior writeCSV's header
// depends on: every task appearing in ANY entry's TaskQuality, across both
// ByJudge and Combined, sorted for a stable order -- not hardcoded to
// whatever prompts/ happens to define (this mode deliberately doesn't
// import it).
func TestTaskColumns(t *testing.T) {
	out := summaryReport{
		ByJudge: []judgeSummary{
			{Judge: "j", Summary: []summaryEntry{
				{Model: "m", Bucket: "30s", TaskQuality: map[string]int{"text_correction": 50}},
			}},
		},
		Combined: []summaryEntry{
			{Model: "m", Bucket: "30s", TaskQuality: map[string]int{"summarization": 90, "information_extraction": 80}},
		},
	}

	assert.Equal(t, []string{"information_extraction", "summarization", "text_correction"}, taskColumns(out))
}

func TestTaskColumns_noTaskDataAtAll(t *testing.T) {
	out := summaryReport{Combined: []summaryEntry{{Model: "m", Bucket: "30s"}}}
	assert.Empty(t, taskColumns(out))
}

// TestTTFTLatencyByModelBucket exercises the TTFT path's own grouping (mirrors
// summarize's error-exclusion and per-group stats, but keyed for lookup
// rather than rendered as its own report rows -- see attachTTFT).
func TestTTFTLatencyByModelBucket(t *testing.T) {
	samples := []ttftSourceSample{
		// model "b", bucket "30s": two successes.
		{Model: "b", Bucket: "30s", TimeToFirstTokenS: 0.1},
		{Model: "b", Bucket: "30s", TimeToFirstTokenS: 0.3},
		// model "a", bucket "120s": one success, one hard failure (must not
		// pollute the stats).
		{Model: "a", Bucket: "120s", TimeToFirstTokenS: 0.5},
		{Model: "a", Bucket: "120s", Error: "timeout"},
	}

	byModelBucket := ttftLatencyByModelBucket(samples)
	require.Len(t, byModelBucket, 2, "one entry per distinct (model, bucket)")

	a120 := byModelBucket[modelBucketKey{"a", "", "120s"}]
	require.Equal(t, 1, a120.N, "the failed sample must not contribute a TTFT data point")
	assert.InDelta(t, 0.5, a120.Mean, 1e-9)

	b30 := byModelBucket[modelBucketKey{"b", "", "30s"}]
	require.Equal(t, 2, b30.N)
	assert.InDelta(t, 0.1, b30.Min, 1e-9)
	assert.InDelta(t, 0.3, b30.Max, 1e-9)
	assert.InDelta(t, 0.2, b30.Mean, 1e-9)

	_, ok := byModelBucket[modelBucketKey{"nonexistent", "", "900s"}]
	assert.False(t, ok, "an unmeasured (model, bucket) pair must have no entry at all, not a zero-valued one")
}

// TestAttachTTFT locks in the actual merge behavior this feature is about:
// entries sharing a (model, bucket) with a TTFT measurement get it attached;
// entries with none (an unmeasured bucket, or no TTFT data loaded at all)
// are left at the zero value -- never a wrong-but-plausible-looking number.
func TestAttachTTFT(t *testing.T) {
	byModelBucket := map[modelBucketKey]stats{
		{"gemini", "", "30s"}: {N: 2, Mean: 0.2, Min: 0.1, Max: 0.3, P50: 0.2, P95: 0.3},
	}

	entries := []summaryEntry{
		{Model: "gemini", Bucket: "30s"},
		{Model: "gemini", Bucket: "900s"}, // TTFT never measures this bucket.
		{Model: "other-model", Bucket: "30s"},
	}

	attachTTFT(entries, byModelBucket)

	assert.Equal(t, stats{N: 2, Mean: 0.2, Min: 0.1, Max: 0.3, P50: 0.2, P95: 0.3}, entries[0].TTFT)
	assert.Equal(t, stats{}, entries[1].TTFT, "a bucket TTFT never measured must be left blank, not zeroed-but-claimed")
	assert.Equal(t, stats{}, entries[2].TTFT, "a model TTFT never measured must be left blank too")

	t.Run("a nil map (no TTFT results dir at all) leaves every entry blank", func(t *testing.T) {
		entries := []summaryEntry{{Model: "gemini", Bucket: "30s"}}
		attachTTFT(entries, nil)
		assert.Equal(t, stats{}, entries[0].TTFT)
	})
}

func TestLoadTTFTReports(t *testing.T) {
	dir := t.TempDir()

	writeSummaryTTFTReport(t, dir, "ttft-report.json", ttftSourceReport{
		Samples: []ttftSourceSample{
			{Model: "gemini", Bucket: "30s", TimeToFirstTokenS: 0.2},
			{Model: "gemini", Bucket: "120s", TimeToFirstTokenS: 0.4},
		},
	})
	// A non-ttft-report file in the same directory must be ignored.
	require.NoError(t, os.WriteFile(filepath.Join(dir, "report-2026-08-26T10:00:00Z.json"), []byte(`{}`), 0o644))

	names, samples, err := loadTTFTReports(dir)
	require.NoError(t, err)
	assert.Equal(t, []string{"ttft-report.json"}, names)
	require.Len(t, samples, 2)
	assert.Equal(t, "gemini", samples[0].Model)
}

func TestLoadTTFTReports_noMatchingFiles(t *testing.T) {
	_, _, err := loadTTFTReports(t.TempDir())
	require.Error(t, err)
}

// TestPrintEntries_includesTTFTColumn locks in that a merged TTFT value
// actually reaches the printed table -- the row-level counterpart to
// TestWriteCSV's CSV-column check -- and that a blank (unmeasured) TTFT
// renders as "no data" rather than panicking on the zero stats{}.
func TestPrintEntries_includesTTFTColumn(t *testing.T) {
	entries := []summaryEntry{
		{Model: "gemini", Bucket: "30s", TTFT: stats{N: 2, Min: 0.1, Mean: 0.2, P50: 0.2, P95: 0.25, Max: 0.3}},
		{Model: "gemini", Bucket: "900s"}, // TTFT never measures this bucket.
	}

	var buf bytes.Buffer
	assert.NotPanics(t, func() { printEntries(&buf, entries) })

	output := buf.String()
	assert.Contains(t, output, "0.20", "the measured row's TTFT mean must appear in the table")
	assert.Contains(t, output, "no data", "the unmeasured row's blank TTFT must render as no data, not a false zero")
}

// TestPrintEntries_includesTaskQualityColumn locks in that the per-task
// breakdown actually reaches the printed table, sorted by task name, and
// that a row with no breakdown at all renders as "n/a" rather than an empty
// string or a panic on the nil map.
func TestPrintEntries_includesTaskQualityColumn(t *testing.T) {
	entries := []summaryEntry{
		{Model: "gemini", Bucket: "30s", TaskQuality: map[string]int{"text_correction": 42, "summarization": 87}},
		{Model: "gemini", Bucket: "900s"}, // no per-task breakdown for this row.
	}

	var buf bytes.Buffer
	assert.NotPanics(t, func() { printEntries(&buf, entries) })

	output := buf.String()
	assert.Contains(t, output, "summarization=87 text_correction=42", "sorted by task name, regardless of map iteration order")
	assert.Contains(t, output, "n/a")
}

// TestPrintEntries_includesCostPerQualityColumns mirrors
// TestPrintEntries_includesTaskQualityColumn for the overall and per-task
// cost-per-quality columns.
func TestPrintEntries_includesCostPerQualityColumns(t *testing.T) {
	entries := []summaryEntry{
		{
			Model: "gemini", Bucket: "30s", QualityScore: 83, CostPerQuality: 0.001,
			TaskCostPerQuality: map[string]float64{"text_correction": 0.0005, "summarization": 0.002},
		},
		{Model: "gemini", Bucket: "900s"}, // no cost data for this row.
	}

	var buf bytes.Buffer
	assert.NotPanics(t, func() { printEntries(&buf, entries) })

	output := buf.String()
	assert.Contains(t, output, "0.001¢", "the overall cost-per-quality must reach the printed table")
	assert.Contains(t, output, "summarization=0.002¢ text_correction=0.0005¢", "sorted by task name")
	assert.Contains(t, output, "n/a")
}

func TestPrintEntries_includesCostPerHourColumns(t *testing.T) {
	entries := []summaryEntry{
		{
			Model: "gemini", Bucket: "30s", QualityScore: 83, CostPerHourUSD: 1.25,
			TaskCostPerHourUSD: map[string]float64{"text_correction": 0.75, "summarization": 1.5},
		},
	}

	var buf bytes.Buffer
	printEntries(&buf, entries)

	output := buf.String()
	assert.Contains(t, output, "$1.25")
	assert.Contains(t, output, "summarization=$1.5 text_correction=$0.75", "sorted by task name")
}

// TestPrintEntries_cheapCostPerQualityIsNotFlattened proves the printed table
// shows a sub-millionth-of-a-dollar cost per quality point with its real
// digits instead of "0.000001¢".
func TestPrintEntries_cheapCostPerQualityIsNotFlattened(t *testing.T) {
	entries := []summaryEntry{{
		Model: "tiny", Bucket: "30s", QualityScore: 87, CostPerQuality: 5.046e-07,
		TaskCostPerQuality: map[string]float64{"summarization": 6.922e-07},
	}}

	var buf bytes.Buffer
	printEntries(&buf, entries)

	assert.Contains(t, buf.String(), "5.046e-07¢")
	assert.Contains(t, buf.String(), "summarization=6.922e-07¢")
	assert.NotContains(t, buf.String(), "0.000001¢")
}

func writeSummaryTTFTReport(t *testing.T, dir, name string, report ttftSourceReport) {
	t.Helper()

	data, err := json.Marshal(report)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, name), data, 0o644))
}

func writeSummaryReport(t *testing.T, dir, name string, report sourceReport) {
	t.Helper()

	data, err := json.Marshal(report)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, name), data, 0o644))
}
