package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func entryModelNames(rows []summaryEntry) []string {
	names := make([]string, len(rows))
	for i, e := range rows {
		names[i] = e.Model
	}

	return names
}

func useCaseModelNames(models []useCaseModel) []string {
	names := make([]string, len(models))
	for i, m := range models {
		names[i] = m.Model
	}

	return names
}

func TestTaskTitle(t *testing.T) {
	tests := []struct{ task, want string }{
		{"summarization", "Summarization"},
		{"information_extraction", "Information Extraction"},
		{"text_correction", "Text Correction"},
		{"dictation", "Dictation"},
	}

	for _, tt := range tests {
		assert.Equal(t, tt.want, taskTitle(tt.task))
	}
}

func TestTasksIn(t *testing.T) {
	rows := []summaryEntry{
		{Model: "a", TaskQuality: map[string]int{"summarization": 90, "dictation": 80}},
		{Model: "b", TaskQuality: map[string]int{"text_correction": 70}},
		{Model: "c"},
	}

	assert.Equal(t, []string{"dictation", "summarization", "text_correction"}, tasksIn(rows))
}

func TestTasksIn_none(t *testing.T) {
	assert.Empty(t, tasksIn([]summaryEntry{{Model: "a"}}))
}

func TestTopModels(t *testing.T) {
	rows := []summaryEntry{
		{Model: "a", QualityScore: 90},
		{Model: "b", QualityScore: 70},
		{Model: "c", QualityScore: 80},
		{Model: "d", QualityScore: 60},
		{Model: "e", QualityScore: 75},
		{Model: "f", QualityScore: 85},
		{Model: "g", QualityScore: 72},
	}

	gate := func(e summaryEntry) bool { return e.QualityScore >= 70 }
	less := func(a, b summaryEntry) bool { return a.QualityScore > b.QualityScore }

	got := topModels(rows, gate, less)

	require.Len(t, got, maxModelsPerUseCase, "6 rows pass the gate; the cap must still win")
	assert.Equal(t, []string{"a", "f", "c", "e", "g"}, entryModelNames(got), "top 5 by score; b (lowest passing score) is cut by the cap")
}

func TestTopModels_noneEligible(t *testing.T) {
	rows := []summaryEntry{{Model: "a", QualityScore: 10}}
	gate := func(e summaryEntry) bool { return e.QualityScore >= 70 }

	assert.Nil(t, topModels(rows, gate, func(a, b summaryEntry) bool { return true }))
}

func TestToUseCaseModels(t *testing.T) {
	rows := []summaryEntry{{
		Model: "a", ReasoningEffort: "low", QualityScore: 90, CostPerQuality: 0.001, CostPerHourUSD: 0.25,
		Latency:     stats{P50: 1.5},
		TaskQuality: map[string]int{"summarization": 88},
	}}

	global := toUseCaseModels(rows, "")
	require.Len(t, global, 1)
	assert.Equal(t, useCaseModel{Model: "a", ReasoningEffort: "low", Rank: 1, QualityScore: 90, CostPerQuality: 0.001, CostPerHourUSD: 0.25, LatencyP50S: 1.5}, global[0])

	perTask := toUseCaseModels(rows, "summarization")
	assert.Equal(t, 88, perTask[0].TaskQualityScore)
}

func TestTaskSpecialistUseCase(t *testing.T) {
	rows := []summaryEntry{
		{Model: "great", QualityScore: 88, TaskQuality: map[string]int{"summarization": 95}, TaskCostPerQuality: map[string]float64{"summarization": 0.01}},
		{Model: "good-cheaper", QualityScore: 85, TaskQuality: map[string]int{"summarization": 92}, TaskCostPerQuality: map[string]float64{"summarization": 0.001}},
		{Model: "below-premium-floor", QualityScore: 85, TaskQuality: map[string]int{"summarization": 70}},
		{Model: "other-task-only", QualityScore: 85, TaskQuality: map[string]int{"dictation": 99}},
		{Model: "spiky-weak-overall", QualityScore: 50, TaskQuality: map[string]int{"summarization": 99}, TaskCostPerQuality: map[string]float64{"summarization": 0.001}},
	}

	uc, ok := taskSpecialistUseCase("30s", "summarization", rows)
	require.True(t, ok)

	assert.Equal(t, "30s", uc.Bucket)
	assert.Equal(t, "Summarization Specialist", uc.Title)
	assert.ElementsMatch(t, []string{"summarization", "premium", "high-accuracy"}, uc.Tags)
	assert.Equal(t, []string{"great", "good-cheaper"}, useCaseModelNames(uc.Models),
		"spiky-weak-overall has the single best task_quality (99) but its overall quality_score (50) fails overallQualityAcceptableNonRealtime, so it's rejected")
}

func TestTaskSpecialistUseCase_noneEligible(t *testing.T) {
	rows := []summaryEntry{{Model: "a", TaskQuality: map[string]int{"summarization": 50}}}

	_, ok := taskSpecialistUseCase("30s", "summarization", rows)
	assert.False(t, ok)
}

func TestBudgetTaskUseCase(t *testing.T) {
	rows := []summaryEntry{
		{Model: "cheap-good-enough", QualityScore: 80, TaskQuality: map[string]int{"dictation": 82}, TaskCostPerQuality: map[string]float64{"dictation": 0.0001}},
		{Model: "expensive-great", QualityScore: 90, TaskQuality: map[string]int{"dictation": 99}, TaskCostPerQuality: map[string]float64{"dictation": 0.05}},
		{Model: "cheapest-but-below-task-floor", QualityScore: 80, TaskQuality: map[string]int{"dictation": 75}, TaskCostPerQuality: map[string]float64{"dictation": 0.00001}},
		{Model: "cheapest-but-weak-overall", QualityScore: 40, TaskQuality: map[string]int{"dictation": 80}, TaskCostPerQuality: map[string]float64{"dictation": 0.000001}},
	}

	uc, ok := budgetTaskUseCase("30s", "dictation", rows)
	require.True(t, ok)

	assert.Equal(t, "Budget Dictation", uc.Title)
	assert.Equal(t, []string{"cheap-good-enough", "expensive-great"}, useCaseModelNames(uc.Models),
		"cheapest-but-below-task-floor clears the old 70 floor but not qualityFloorNonRealtime (80); cheapest-but-weak-overall clears the dictation floor but its weak overall quality_score (40) rejects it")
}

func TestRealtimeTaskUseCase(t *testing.T) {
	rows := []summaryEntry{
		{Model: "fast", QualityScore: 85, TaskQuality: map[string]int{"summarization": 80}, Latency: stats{N: 10, P50: 0.5}},
		{Model: "slow", QualityScore: 85, TaskQuality: map[string]int{"summarization": 90}, Latency: stats{N: 10, P50: 5}},
		{Model: "no-latency-data", QualityScore: 85, TaskQuality: map[string]int{"summarization": 95}},
		{Model: "fast-but-too-bad", QualityScore: 85, TaskQuality: map[string]int{"summarization": 40}, Latency: stats{N: 10, P50: 0.1}},
		{Model: "fastest-but-weak-overall", QualityScore: 40, TaskQuality: map[string]int{"summarization": 80}, Latency: stats{N: 10, P50: 0.05}},
	}

	uc, ok := realtimeTaskUseCase("30s", "summarization", rows)
	require.True(t, ok)

	assert.Equal(t, "Real-Time Summarization", uc.Title)
	assert.Equal(t, []string{"fast", "slow"}, useCaseModelNames(uc.Models),
		"fastest-but-weak-overall clears the task floor and is the fastest of all, but its weak overall quality_score (40) rejects it")
}

func TestFastestLatencyP50(t *testing.T) {
	rows := []summaryEntry{
		{Model: "a", QualityScore: 90, Latency: stats{N: 5, P50: 2}},
		{Model: "b", QualityScore: 90, Latency: stats{N: 5, P50: 0.5}},
		{Model: "c", QualityScore: 40, Latency: stats{N: 5, P50: 0.1}},
		{Model: "d", QualityScore: 90, Latency: stats{N: 0, P50: 0.01}},
	}

	gate := func(e summaryEntry) bool { return e.QualityScore >= 65 }

	fastest, ok := fastestLatencyP50(rows, gate)
	require.True(t, ok)
	assert.InDelta(t, 0.5, fastest, 1e-9, "c fails the gate and d has no latency data, so b's 0.5 wins over a's 2")
}

func TestFastestLatencyP50_noneEligible(t *testing.T) {
	rows := []summaryEntry{{Model: "a", QualityScore: 40, Latency: stats{N: 5, P50: 0.1}}}
	gate := func(e summaryEntry) bool { return e.QualityScore >= 65 }

	_, ok := fastestLatencyP50(rows, gate)
	assert.False(t, ok)
}

func TestOverallQualityAcceptable(t *testing.T) {
	tests := []struct {
		name string
		e    summaryEntry
		want bool
	}{
		{"clears the floor", summaryEntry{QualityScore: int(qualityFloorAcceptable)}, true},
		{"below the floor", summaryEntry{QualityScore: int(qualityFloorAcceptable) - 1}, false},
		{"no data (zero value)", summaryEntry{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, overallQualityAcceptable(tt.e))
		})
	}
}

func TestOverallQualityAcceptableNonRealtime(t *testing.T) {
	tests := []struct {
		name string
		e    summaryEntry
		want bool
	}{
		{"clears the floor", summaryEntry{QualityScore: int(qualityFloorNonRealtime)}, true},
		{"below the floor", summaryEntry{QualityScore: int(qualityFloorNonRealtime) - 1}, false},
		{"no data (zero value)", summaryEntry{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, overallQualityAcceptableNonRealtime(tt.e))
		})
	}
}

func TestNearRealtimeTaskUseCase(t *testing.T) {
	const fastestP50 = 1.0

	budget := fastestP50 * nearRealtimeLatencyMultiplier

	rows := []summaryEntry{
		// TaskQuality 88, above qualityFloorMedium (this archetype's own
		// floor), not just qualityFloorAcceptable.
		{Model: "fastest", QualityScore: 80, TaskQuality: map[string]int{"summarization": 88}, Latency: stats{N: 5, P50: fastestP50}},
		{Model: "small-step-big-gain", QualityScore: 85, TaskQuality: map[string]int{"summarization": 95}, Latency: stats{N: 5, P50: budget * 0.9}},
		{Model: "too-slow-despite-best-quality", QualityScore: 90, TaskQuality: map[string]int{"summarization": 99}, Latency: stats{N: 5, P50: budget * 1.1}},
		{Model: "below-floor-but-fastest-overall", QualityScore: 80, TaskQuality: map[string]int{"summarization": 50}, Latency: stats{N: 5, P50: 0.5}},
		{Model: "no-latency-data", QualityScore: 80, TaskQuality: map[string]int{"summarization": 90}},
		// Would reset the anchor to 0.05 (and even clear qualityFloorMedium)
		// if overallQualityAcceptable didn't reject it first.
		{Model: "very-fast-but-weak-overall", QualityScore: 30, TaskQuality: map[string]int{"summarization": 90}, Latency: stats{N: 5, P50: 0.05}},
	}

	uc, ok := nearRealtimeTaskUseCase("30s", "summarization", rows)
	require.True(t, ok)

	assert.Equal(t, "Near-Real-Time Summarization", uc.Title)
	assert.ElementsMatch(t, []string{"summarization", "near-real-time", "balanced"}, uc.Tags)
	assert.Equal(t, []string{"small-step-big-gain", "fastest"}, useCaseModelNames(uc.Models),
		"ranked by quality within nearRealtimeLatencyMultiplier of the fastest acceptable model's own p50 -- the too-slow model loses out despite scoring highest")
}

func TestNearRealtimeTaskUseCase_noneEligible(t *testing.T) {
	rows := []summaryEntry{{Model: "a", TaskQuality: map[string]int{"summarization": 40}, Latency: stats{N: 5, P50: 0.1}}}

	_, ok := nearRealtimeTaskUseCase("30s", "summarization", rows)
	assert.False(t, ok)
}

func TestTopOverallUseCase(t *testing.T) {
	rows := []summaryEntry{
		{Model: "best", QualityScore: 92, CostPerQuality: 0.01},
		{Model: "second", QualityScore: 91, CostPerQuality: 0.001},
		{Model: "below-premium-floor", QualityScore: 80, CostPerQuality: 0.0001},
	}

	uc, ok := topOverallUseCase("120s", rows)
	require.True(t, ok)

	assert.Equal(t, "Top Overall Performer", uc.Title)
	assert.Equal(t, []string{"best", "second"}, useCaseModelNames(uc.Models))
}

func TestBestValueUseCase(t *testing.T) {
	rows := []summaryEntry{
		{Model: "cheap", QualityScore: 80, CostPerQuality: 0.0001},
		{Model: "pricier", QualityScore: 95, CostPerQuality: 0.01},
		{Model: "cheapest-but-below-nonrealtime-floor", QualityScore: 75, CostPerQuality: 0.00001},
	}

	uc, ok := bestValueUseCase("120s", rows)
	require.True(t, ok)

	assert.Equal(t, "Best Value", uc.Title)
	assert.Equal(t, []string{"cheap", "pricier"}, useCaseModelNames(uc.Models),
		"cheapest-but-below-nonrealtime-floor clears qualityFloorAcceptable but not qualityFloorNonRealtime, so it's rejected")
}

func TestRealtimeVoiceAgentUseCase(t *testing.T) {
	rows := []summaryEntry{
		{Model: "fast", QualityScore: 80, Latency: stats{N: 5, P50: 0.4}},
		{Model: "slower", QualityScore: 90, Latency: stats{N: 5, P50: 2}},
		{Model: "fast-but-too-bad", QualityScore: 40, Latency: stats{N: 5, P50: 0.1}},
	}

	uc, ok := realtimeVoiceAgentUseCase("30s", rows)
	require.True(t, ok)

	assert.Equal(t, []string{"fast", "slower"}, useCaseModelNames(uc.Models))
}

func TestNearRealtimeVoiceAgentUseCase(t *testing.T) {
	const fastestP50 = 1.0

	budget := fastestP50 * nearRealtimeLatencyMultiplier

	rows := []summaryEntry{
		// QualityScore 85, not just above qualityFloorAcceptable, since this
		// archetype's own floor currently runs qualityFloorAcceptable+10.
		{Model: "fastest", QualityScore: 85, Latency: stats{N: 5, P50: fastestP50}},
		{Model: "small-step-big-gain", QualityScore: 95, Latency: stats{N: 5, P50: budget * 0.9}},
		{Model: "too-slow-despite-best-quality", QualityScore: 99, Latency: stats{N: 5, P50: budget * 1.1}},
	}

	uc, ok := nearRealtimeVoiceAgentUseCase("30s", rows)
	require.True(t, ok)

	assert.Equal(t, "Near-Real-Time Voice Agent", uc.Title)
	assert.Equal(t, []string{"small-step-big-gain", "fastest"}, useCaseModelNames(uc.Models))
}

func TestLongFormBatchUseCase(t *testing.T) {
	rows := []summaryEntry{
		{Model: "best-quality", QualityScore: 90, TokensPerSecond: 20},
		{Model: "tie-break-loses", QualityScore: 80, TokensPerSecond: 10},
		{Model: "tie-break-wins", QualityScore: 80, TokensPerSecond: 50},
		{Model: "too-bad", QualityScore: 40, TokensPerSecond: 100},
	}

	uc, ok := longFormBatchUseCase("3600s", rows)
	require.True(t, ok)

	assert.Equal(t, []string{"best-quality", "tie-break-wins", "tie-break-loses"}, useCaseModelNames(uc.Models))
}

// TestDeriveUseCases_bucketGating covers realtimeMaxBucketS/batchMinBucketS:
// the real-time archetypes must appear only for short buckets, the batch one
// only for long buckets, never both for the same bucket.
func TestDeriveUseCases_bucketGating(t *testing.T) {
	row := func(bucket string, p50 float64) summaryEntry {
		return summaryEntry{
			Model: "m", Bucket: bucket, QualityScore: 90, CostPerQuality: 0.001,
			Latency: stats{N: 1, P50: p50}, TokensPerSecond: 10,
			TaskQuality:        map[string]int{"summarization": 90},
			TaskCostPerQuality: map[string]float64{"summarization": 0.001},
		}
	}

	useCases := deriveUseCases([]summaryEntry{row("30s", 0.5), row("3600s", 30)})

	present := map[string]bool{}
	for _, uc := range useCases {
		present[uc.Bucket+"|"+uc.Title] = true
	}

	assert.True(t, present["30s|Real-Time Voice Agent"])
	assert.True(t, present["30s|Real-Time Summarization"])
	assert.True(t, present["30s|Near-Real-Time Voice Agent"])
	assert.True(t, present["30s|Near-Real-Time Summarization"])
	assert.False(t, present["3600s|Real-Time Voice Agent"])
	assert.False(t, present["3600s|Real-Time Summarization"])
	assert.False(t, present["3600s|Near-Real-Time Voice Agent"])
	assert.False(t, present["3600s|Near-Real-Time Summarization"])
	assert.True(t, present["3600s|Long-Form Call Analysis"])
	assert.False(t, present["30s|Long-Form Call Analysis"])

	require.NotEmpty(t, useCases)
	assert.Equal(t, "30s", useCases[0].Bucket, "buckets are emitted in ascending bucketSeconds order")
}

func TestDeriveUseCases_omitsArchetypesWithNoEligibleModels(t *testing.T) {
	entries := []summaryEntry{{Model: "weak", Bucket: "900s", QualityScore: 30, CostPerQuality: 0.5}}

	assert.Empty(t, deriveUseCases(entries), "nothing clears even the acceptable floor")
}

func TestDeriveUseCases_capsAtThreeModelsPerUseCase(t *testing.T) {
	var entries []summaryEntry
	for i := 0; i < 5; i++ {
		entries = append(entries, summaryEntry{
			Model: string(rune('a' + i)), Bucket: "120s",
			QualityScore: 90 - i, CostPerQuality: 0.001,
		})
	}

	for _, uc := range deriveUseCases(entries) {
		assert.LessOrEqual(t, len(uc.Models), maxModelsPerUseCase, uc.Title)
	}
}

func TestWriteUseCases(t *testing.T) {
	dir := t.TempDir()
	entries := []summaryEntry{
		{Model: "m", Bucket: "30s", QualityScore: 90, CostPerQuality: 0.001, Latency: stats{N: 1, P50: 0.5}},
	}

	require.NoError(t, writeUseCases(dir, entries))

	data, err := os.ReadFile(filepath.Join(dir, "use_cases.json"))
	require.NoError(t, err)

	var out useCasesReport
	require.NoError(t, json.Unmarshal(data, &out))
	assert.False(t, out.GeneratedAt.IsZero())
	assert.NotEmpty(t, out.UseCases)
	assert.True(t, bytes.HasSuffix(data, []byte("\n")), "must satisfy pre-commit's end-of-file-fixer without a second pass")
}
