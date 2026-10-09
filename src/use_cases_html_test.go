package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQualityClass(t *testing.T) {
	tests := []struct {
		name  string
		score int
		want  string
	}{
		{"at premium floor", int(qualityFloorPremium), "q-high"},
		{"above premium floor", 97, "q-high"},
		{"at acceptable floor", int(qualityFloorAcceptable), "q-mid"},
		{"between acceptable and premium", 80, "q-mid"},
		{"below acceptable floor", 40, "q-low"},
		{"zero (no data)", 0, "q-low"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, qualityClass(tt.score))
		})
	}
}

func TestFormatCostPerQuality(t *testing.T) {
	tests := []struct {
		name string
		v    float64
		want string
	}{
		{"no data", 0, "n/a"},
		{"typical value renders shortest round-trip form", 0.001162, "0.001162¢"},
		{"small value keeps every significant digit", 0.000023, "0.000023¢"},
		{"no trailing zero padding", 0.05, "0.05¢"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, formatCostPerQuality(tt.v))
		})
	}
}

func TestFormatLatencyP50(t *testing.T) {
	tests := []struct {
		name    string
		seconds float64
		want    string
	}{
		{"no data", 0, "n/a"},
		{"negative treated as no data", -1, "n/a"},
		{"sub-second renders as milliseconds", 0.561, "561 ms"},
		{"rounds to nearest millisecond", 0.5006, "501 ms"},
		{"one second and above renders in seconds", 2.226, "2.23 s"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, formatLatencyP50(tt.seconds))
		})
	}
}

func TestBucketDurationLabel(t *testing.T) {
	tests := []struct{ bucket, want string }{
		{"1s", "1 second"},
		{"30s", "30 seconds"},
		{"120s", "2 minutes"},
		{"900s", "15 minutes"},
		{"3600s", "1 hour"},
		{"7200s", "2 hours"},
		{"unparsable", "unparsable"},
	}

	for _, tt := range tests {
		t.Run(tt.bucket, func(t *testing.T) {
			assert.Equal(t, tt.want, bucketDurationLabel(tt.bucket))
		})
	}
}

func TestBucketKindLabel(t *testing.T) {
	tests := []struct{ bucket, want string }{
		{"30s", "real-time-adjacent"},
		{"120s", "real-time-adjacent"},
		{"300s", ""},
		{"900s", "batch / async"},
		{"3600s", "batch / async"},
		{"unparsable", ""},
	}

	for _, tt := range tests {
		t.Run(tt.bucket, func(t *testing.T) {
			assert.Equal(t, tt.want, bucketKindLabel(tt.bucket))
		})
	}
}

func TestBuildUseCasesPageView(t *testing.T) {
	report := useCasesReport{
		GeneratedAt: time.Date(2026, 9, 24, 21, 9, 32, 0, time.UTC),
		UseCases: []useCase{
			{
				Bucket: "3600s", Title: "Long-Form Call Analysis", Description: "d1",
				Tags: []string{"batch", "long-form"},
				Models: []useCaseModel{
					{Model: "model-a", Rank: 1, QualityScore: 92, CostPerQuality: 0.01, LatencyP50S: 5},
				},
			},
			{
				Bucket: "30s", Title: "Dictation Specialist", Description: "d2",
				Tags: []string{"dictation", "premium"},
				Models: []useCaseModel{
					{Model: "model-a", ReasoningEffort: "low", Rank: 1, QualityScore: 96, TaskQualityScore: 100, CostPerQuality: 0.001, LatencyP50S: 0.5},
					{Model: "model-b", ReasoningEffort: "default", Rank: 2, QualityScore: 90, TaskQualityScore: 95, CostPerQuality: 0.002, LatencyP50S: 0.6},
				},
			},
			{
				Bucket: "30s", Title: "Budget Dictation", Description: "d3",
				Tags: []string{"dictation", "cost-efficient"},
				Models: []useCaseModel{
					{Model: "model-c", Rank: 1, QualityScore: 78, TaskQualityScore: 81, CostPerQuality: 0.00002, LatencyP50S: 0.3},
				},
			},
		},
	}

	view := buildUseCasesPageView(report)

	require.Len(t, view.Buckets, 2, "one entry per distinct bucket")
	assert.Equal(t, "30s", view.Buckets[0].Bucket, "buckets are ordered ascending by duration, not first-seen order")
	assert.Equal(t, "3600s", view.Buckets[1].Bucket)
	assert.Equal(t, "30 seconds", view.Buckets[0].Label)
	assert.Equal(t, "real-time-adjacent", view.Buckets[0].Kind)
	assert.Equal(t, "batch / async", view.Buckets[1].Kind)

	assert.Equal(t, 3, view.TotalUseCases)
	assert.Equal(t, 3, view.TotalModels, "model-a appears twice across use cases but counts once")

	require.Len(t, view.Buckets[0].UseCases, 2, "both 30s use cases grouped together")
	dictationSpecialist := view.Buckets[0].UseCases[0]
	assert.Equal(t, "dictation premium", dictationSpecialist.TagsAttr)

	firstModel := dictationSpecialist.Models[0]
	assert.True(t, firstModel.HasTaskQuality)
	assert.Equal(t, "q-high", firstModel.QualityClass, "96 clears the premium floor")
	assert.Equal(t, "0.001¢", firstModel.CostLabel)
	assert.Equal(t, "500 ms", firstModel.LatencyLabel)
	assert.Equal(t, "low", firstModel.ReasoningEffortLabel)

	// Tag counts: "dictation" appears on 2 of the 3 use cases, every other
	// tag on exactly 1 -- frequency order first, alphabetical tiebreak
	// second.
	var tags []string
	for _, tc := range view.Tags {
		tags = append(tags, tc.Tag)
	}
	require.NotEmpty(t, tags)
	assert.Equal(t, "dictation", tags[0])

	dictationCount := 0
	for _, tc := range view.Tags {
		if tc.Tag == "dictation" {
			dictationCount = tc.Count
		}
	}
	assert.Equal(t, 2, dictationCount)
}

func TestRenderUseCasesHTML_escapesUntrustedText(t *testing.T) {
	report := useCasesReport{
		UseCases: []useCase{
			{
				Bucket: "30s", Title: `<script>alert(1)</script>`, Description: "d",
				Models: []useCaseModel{{Model: "m", Rank: 1, QualityScore: 90}},
			},
		},
	}

	var buf bytes.Buffer
	require.NoError(t, renderUseCasesHTML(&buf, report))

	out := buf.String()
	assert.NotContains(t, out, "<script>alert(1)</script>", "html/template must escape use-case text, not pass it through raw")
	assert.Contains(t, out, "&lt;script&gt;")
}

func TestRenderUseCasesHTML_omitsTaskQualityWhenAbsent(t *testing.T) {
	report := useCasesReport{
		UseCases: []useCase{
			{
				Bucket: "3600s", Title: "Best Value", Description: "d",
				Models: []useCaseModel{{Model: "m", Rank: 1, QualityScore: 90}},
			},
		},
	}

	var buf bytes.Buffer
	require.NoError(t, renderUseCasesHTML(&buf, report))

	// The page's static legend explains what "Task quality" means regardless
	// of the data, so assert on the per-model metric block's own markup
	// (its title tooltip is unique to it) rather than that phrase alone.
	assert.NotContains(t, buf.String(), `title="1-100 score on just this use case's own task"`,
		"a global use case (no per-task score) must not render an empty/zero task-quality metric")
}

// TestRenderUseCasesHTML_rendersReasoningEffort confirms the reasoning
// effort metric actually reaches the rendered page next to Latency p50, and
// that a model benchmarked without -reasoning-effort still shows an
// explicit "default" rather than an empty cell.
func TestRenderUseCasesHTML_rendersReasoningEffort(t *testing.T) {
	report := useCasesReport{
		UseCases: []useCase{
			{
				Bucket: "30s", Title: "t", Description: "d",
				Models: []useCaseModel{
					{Model: "m", ReasoningEffort: "low", Rank: 1, QualityScore: 90},
					{Model: "m2", ReasoningEffort: "default", Rank: 2, QualityScore: 85},
				},
			},
		},
	}

	var buf bytes.Buffer
	require.NoError(t, renderUseCasesHTML(&buf, report))

	output := buf.String()
	assert.Contains(t, output, "Reasoning effort")
	assert.Contains(t, output, ">low<")
	assert.Contains(t, output, ">default<")
}

// TestRenderUseCasesHTML_qualityBarWidth guards against html/template's CSS
// contextual autoescaping rewriting an untrusted-looking value inside
// style="width:...%" into its "ZgotmplZ" safety placeholder instead of the
// real number -- easy to miss by eye since the template still parses and
// executes without error either way.
func TestRenderUseCasesHTML_qualityBarWidth(t *testing.T) {
	report := useCasesReport{
		UseCases: []useCase{
			{
				Bucket: "30s", Title: "t", Description: "d",
				Models: []useCaseModel{{Model: "m", Rank: 1, QualityScore: 96}},
			},
		},
	}

	var buf bytes.Buffer
	require.NoError(t, renderUseCasesHTML(&buf, report))

	assert.Contains(t, buf.String(), `width:96%`)
	assert.NotContains(t, buf.String(), "ZgotmplZ")
}

func TestWriteUseCasesHTML(t *testing.T) {
	dir := t.TempDir()
	report := useCasesReport{
		GeneratedAt: time.Date(2026, 9, 24, 21, 9, 32, 0, time.UTC),
		UseCases: []useCase{
			{
				Bucket: "30s", Title: "Dictation Specialist", Description: "Highest-accuracy models.",
				Tags: []string{"dictation", "premium"},
				Models: []useCaseModel{
					{Model: "claude-sonnet-4-5-20250929", Rank: 1, QualityScore: 96, CostPerQuality: 0.001162, LatencyP50S: 2.226},
				},
			},
		},
	}

	encoded, err := json.MarshalIndent(report, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "use_cases.json"), encoded, 0o644))

	require.NoError(t, writeUseCasesHTML(dir))

	htmlBytes, err := os.ReadFile(filepath.Join(dir, "use_cases.html"))
	require.NoError(t, err)

	html := string(htmlBytes)
	assert.True(t, strings.HasPrefix(strings.TrimSpace(html), "<!doctype html>"))
	assert.Contains(t, html, "Dictation Specialist")
	assert.Contains(t, html, "claude-sonnet-4-5-20250929")
	assert.Contains(t, html, "0.001162¢")
}

func TestWriteUseCasesHTML_missingJSON(t *testing.T) {
	err := writeUseCasesHTML(t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "use_cases.json")
}
