package main

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRoundSignificant(t *testing.T) {
	tests := []struct {
		name   string
		val    float64
		digits int
		want   float64
	}{
		{"keeps four digits of a tiny value", 5.046271e-07, 4, 5.046e-07},
		{"rounds up the last kept digit", 0.000522875817, 4, 0.0005229},
		{"a value already short enough is unchanged", 0.00015, 4, 0.00015},
		{"large value rounds in the integer part", 123456, 4, 123500},
		{"one significant digit", 0.00047, 1, 0.0005},
		{"negative values round symmetrically", -5.046271e-07, 4, -5.046e-07},
		{"zero stays zero", 0, 4, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, roundSignificant(tt.val, tt.digits))
		})
	}

	t.Run("NaN and Inf pass through untouched", func(t *testing.T) {
		assert.True(t, math.IsNaN(roundSignificant(math.NaN(), 4)))
		assert.True(t, math.IsInf(roundSignificant(math.Inf(1), 4), 1))
	})
}

func TestComputeStats(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   Stats
	}{
		{
			name:   "empty input returns the zero value",
			values: nil,
			want:   Stats{},
		},
		{
			name:   "single value: every stat equals it",
			values: []float64{42},
			want:   Stats{N: 1, Min: 42, Max: 42, Mean: 42, P50: 42, P95: 42},
		},
		{
			name:   "five values, unsorted input",
			values: []float64{5, 1, 4, 2, 3},
			// sorted: [1,2,3,4,5]; nearest-rank: p50 -> ceil(0.5*5)-1=2 -> 3,
			// p95 -> ceil(0.95*5)-1=4 -> 5.
			want: Stats{N: 5, Min: 1, Max: 5, Mean: 3, P50: 3, P95: 5},
		},
		{
			name:   "ten values 1..10",
			values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			// p50 -> ceil(5)-1=4 -> 5, p95 -> ceil(9.5)-1=9 -> 10.
			want: Stats{N: 10, Min: 1, Max: 10, Mean: 5.5, P50: 5, P95: 10},
		},
		{
			name:   "seven values with duplicates",
			values: []float64{10, 10, 20, 20, 30, 30, 5},
			// sorted: [5,10,10,20,20,30,30]; p50 -> ceil(3.5)-1=3 -> 20,
			// p95 -> ceil(6.65)-1=6 -> 30. Mean is rounded to 3 decimal
			// places (see roundFloat) -- 125.0/7.0 = 17.857142857..., not
			// the raw division result.
			want: Stats{N: 7, Min: 5, Max: 30, Mean: 17.857, P50: 20, P95: 30},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeStats(tt.values)
			assert.Equal(t, tt.want.N, got.N)
			assert.InDelta(t, tt.want.Min, got.Min, 1e-9)
			assert.InDelta(t, tt.want.Max, got.Max, 1e-9)
			assert.InDelta(t, tt.want.Mean, got.Mean, 1e-9)
			assert.InDelta(t, tt.want.P50, got.P50, 1e-9)
			assert.InDelta(t, tt.want.P95, got.P95, 1e-9)
		})
	}
}

func TestComputeStats_doesNotMutateInput(t *testing.T) {
	values := []float64{3, 1, 2}
	_ = ComputeStats(values)
	assert.Equal(t, []float64{3, 1, 2}, values, "ComputeStats must sort a copy, not the caller's slice")
}

func TestSample_TokensPerSecond(t *testing.T) {
	tests := []struct {
		name   string
		sample Sample
		want   float64
	}{
		{
			name:   "normal case: 100 tokens over 2s",
			sample: Sample{CompletionTokens: 100, TotalLatency: 2 * time.Second},
			want:   50,
		},
		{
			name:   "failed sample scores zero regardless of token/latency fields",
			sample: Sample{CompletionTokens: 100, TotalLatency: 2 * time.Second, Err: errors.New("boom")},
			want:   0,
		},
		{
			name:   "zero completion tokens scores zero, not a divide artifact",
			sample: Sample{CompletionTokens: 0, TotalLatency: 2 * time.Second},
			want:   0,
		},
		{
			name:   "zero latency scores zero rather than +Inf",
			sample: Sample{CompletionTokens: 100, TotalLatency: 0},
			want:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, tt.sample.TokensPerSecond(), 1e-9)
		})
	}
}

// TestAggregate exercises grouping, per-cell statistics, error exclusion, and
// sort order together — the four things that actually have to cooperate for
// a report to be trustworthy. Table-driven per
// TestLLMGatewayFramework_processUserPrompt's structure (fields/args/assert),
// simplified because Aggregate has one dependency (a []Sample), not a
// service container's worth.
func TestAggregate(t *testing.T) {
	type args struct {
		samples []Sample
	}

	tests := []struct {
		name   string
		args   args
		assert func(t *testing.T, cells []Cell)
	}{
		{
			name: "groups by (model, task, bucket), excludes failures from stats, sorts by model then bucket duration",
			args: args{
				samples: []Sample{
					// model "b", bucket "120s": one success.
					{
						Model: "b", Task: "summarization", Bucket: "120s",
						TotalLatency: 2 * time.Second, TimeToFirstToken: 500 * time.Millisecond,
						CompletionTokens: 100,
						Judge:            map[string]float64{"faithfulness": 4, "coverage": 5},
					},
					// model "b", bucket "30s": one success.
					{
						Model: "b", Task: "summarization", Bucket: "30s",
						TotalLatency: 1 * time.Second, TimeToFirstToken: 200 * time.Millisecond,
						CompletionTokens: 20,
						Judge:            map[string]float64{"faithfulness": 3, "coverage": 3},
					},
					// model "a", bucket "120s": one success, one failure.
					{
						Model: "a", Task: "summarization", Bucket: "120s",
						TotalLatency: 4 * time.Second, TimeToFirstToken: 1 * time.Second,
						CompletionTokens: 40,
						Judge:            map[string]float64{"faithfulness": 2, "coverage": 2},
					},
					{
						Model: "a", Task: "summarization", Bucket: "120s",
						Err: errors.New("rate limited"),
					},
				},
			},
			assert: func(t *testing.T, cells []Cell) {
				require.Len(t, cells, 3, "one cell per distinct (model, task, bucket)")

				// Sort order: model "a" before "b"; within "b", bucket 30s
				// (duration 30) before 120s (duration 120) — lexical order
				// would get this backwards ("120s" < "30s").
				assert.Equal(t, "a", cells[0].Model)
				assert.Equal(t, "120s", cells[0].Bucket)
				assert.Equal(t, "b", cells[1].Model)
				assert.Equal(t, "30s", cells[1].Bucket)
				assert.Equal(t, "b", cells[2].Model)
				assert.Equal(t, "120s", cells[2].Bucket)

				// model "a"/120s: 2 attempts, 1 error, stats from the 1
				// success only.
				a120 := cells[0]
				assert.Equal(t, 2, a120.N)
				assert.Equal(t, 1, a120.Errors)
				require.Equal(t, 1, a120.Latency.N, "the failed sample must not contribute a latency data point")
				assert.InDelta(t, 4.0, a120.Latency.Mean, 1e-9)
				assert.InDelta(t, 2.0, a120.JudgeMeans["faithfulness"], 1e-9)

				// model "b"/120s: throughput = 100 tokens / 2s = 50 tok/s.
				b120 := cells[2]
				assert.Equal(t, 0, b120.Errors)
				assert.InDelta(t, 50.0, b120.Throughput.Mean, 1e-9)
				assert.InDelta(t, 0.5, b120.TTFT.Mean, 1e-9)
				assert.Equal(t, 1, b120.Judged)
			},
		},
		{
			name: "a schema-invalid sample is counted but still contributes its (worst-case) judge score",
			args: args{
				samples: []Sample{
					{
						Model: "m", Task: "information_extraction", Bucket: "30s",
						TotalLatency: time.Second, CompletionTokens: 10,
						SchemaInvalid: true,
						Judge:         map[string]float64{"groundedness": 1, "completeness": 1},
					},
					{
						Model: "m", Task: "information_extraction", Bucket: "30s",
						TotalLatency: time.Second, CompletionTokens: 10,
						Judge: map[string]float64{"groundedness": 5, "completeness": 5},
					},
				},
			},
			assert: func(t *testing.T, cells []Cell) {
				require.Len(t, cells, 1)
				cell := cells[0]
				assert.Equal(t, 1, cell.SchemaInvalid)
				assert.Equal(t, 2, cell.Judged)
				assert.InDelta(t, 3.0, cell.JudgeMeans["groundedness"], 1e-9, "worst-case score must still average in, not be excluded")
			},
		},
		{
			name: "no samples produces no cells",
			args: args{samples: nil},
			assert: func(t *testing.T, cells []Cell) {
				assert.Empty(t, cells)
			},
		},
		{
			name: "the same (model, task, bucket) at two different reasoning efforts produces two cells, never one blending both",
			args: args{
				samples: []Sample{
					{
						Model: "m", ReasoningEffort: "low", Task: "summarization", Bucket: "30s",
						TotalLatency: 1 * time.Second, CompletionTokens: 10,
					},
					{
						Model: "m", ReasoningEffort: "high", Task: "summarization", Bucket: "30s",
						TotalLatency: 9 * time.Second, CompletionTokens: 10,
					},
				},
			},
			assert: func(t *testing.T, cells []Cell) {
				require.Len(t, cells, 2, "reasoning effort must be part of the grouping key, not just the model")

				byEffort := map[string]Cell{cells[0].ReasoningEffort: cells[0], cells[1].ReasoningEffort: cells[1]}

				require.Contains(t, byEffort, "low")
				require.Contains(t, byEffort, "high")
				assert.InDelta(t, 1.0, byEffort["low"].Latency.Mean, 1e-9)
				assert.InDelta(t, 9.0, byEffort["high"].Latency.Mean, 1e-9, "a blended cell would average to 5.0 instead")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.assert(t, Aggregate(tt.args.samples))
		})
	}
}
