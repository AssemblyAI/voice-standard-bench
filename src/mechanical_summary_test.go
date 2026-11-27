package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMechanicalDictationSummaryIncludesConfiguredJudge(t *testing.T) {
	for _, judge := range []string{"M", "J1"} {
		t.Run(judge, func(t *testing.T) {
			writeFrameworkFixtures(t)
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				writeChatCompletion(w, "Hey, world.", 10, 5)
			}))
			defer server.Close()
			task, err := NewDictationTask(NewGateway(server.URL, "offline-fake-token", time.Second))
			require.NoError(t, err)
			runner := NewRunner[DictationRecord](task, RunConfig{
				Models: []ModelInfo{{ID: "M"}}, JudgeModels: []string{judge}, Concurrency: 1,
			})
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			samples, cells, err := runner.RunBenchmark(ctx)
			require.NoError(t, err)
			require.Len(t, samples, 1)
			assert.Equal(t, map[string]float64{"exact_match": 1}, samples[0].Judge)
			assert.Equal(t, int64(1), calls.Load(), "mechanical grading never calls a judge")
			dir := t.TempDir()
			require.NoError(t, WriteJSONReport(dir, samples, cells, judge))
			require.NoError(t, RunSummary(dir, dir, "", false))
			paths, err := filepath.Glob(filepath.Join(dir, "summary-*.json"))
			require.NoError(t, err)
			require.Len(t, paths, 1)
			data, err := os.ReadFile(paths[0])
			require.NoError(t, err)
			var report summaryReport
			require.NoError(t, json.Unmarshal(data, &report))
			require.Len(t, report.Combined, 1)
			assert.Equal(t, 100, report.Combined[0].QualityScore)
			assert.Equal(t, map[string]int{"dictation": 100}, report.Combined[0].TaskQuality)
		})
	}
}

func TestMechanicalExceptionDoesNotIncludeLLMSelfGrading(t *testing.T) {
	dir := t.TempDir()
	samples := []Sample{
		{Model: "M", Task: "summarization", Bucket: "30s", Judge: map[string]float64{"faithfulness": 5}},
		{Model: "M", Task: "text_correction", Bucket: "30s", Judge: map[string]float64{"improvement": 5}},
		{Model: "M", Task: "information_extraction", Bucket: "30s", Judge: map[string]float64{"faithfulness": 5}},
	}
	require.NoError(t, WriteJSONReport(dir, samples, Aggregate(samples), "M"))
	_, loaded, err := loadReports(dir)
	require.NoError(t, err)
	assert.Empty(t, loaded)
}
