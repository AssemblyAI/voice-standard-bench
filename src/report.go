package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
)

// PrintReport renders cells as a human-readable table, one section per task.
// Row order follows Aggregate's own sort (model, then bucket by ascending
// duration) rather than re-sorting here, so the two stay in sync by
// construction.
func PrintReport(w io.Writer, cells []Cell) {
	currentTask := ""

	for _, cell := range cells {
		if cell.Task != currentTask {
			currentTask = cell.Task
			fmt.Fprintf(w, "\n=== %s ===\n", currentTask)
			fmt.Fprintf(w, "%-32s %-10s %-8s %5s %5s  %-24s  %-24s  %-16s  %s\n",
				"MODEL", "EFFORT", "BUCKET", "N", "ERRS",
				"LATENCY s (p50/p95/max)", "TTFT s (p50/p95/max)", "TOK/S (p50/p95)", "QUALITY (mean)")
		}

		fmt.Fprintf(w, "%-32s %-10s %-8s %5d %5d  %-24s  %-24s  %-16s  %s\n",
			cell.Model, cell.ReasoningEffort, cell.Bucket, cell.N, cell.Errors,
			statTriple(cell.Latency),
			statTriple(cell.TTFT),
			statPair(cell.Throughput),
			qualitySummary(cell),
		)
	}
}

func statTriple(s Stats) string {
	if s.N == 0 {
		return "no data"
	}

	return fmt.Sprintf("%.2f / %.2f / %.2f", s.P50, s.P95, s.Max)
}

func statPair(s Stats) string {
	if s.N == 0 {
		return "no data"
	}

	return fmt.Sprintf("%.1f / %.1f", s.P50, s.P95)
}

func qualitySummary(cell Cell) string {
	if cell.Judged == 0 {
		return "not judged"
	}

	dims := make([]string, 0, len(cell.JudgeMeans))
	for dim := range cell.JudgeMeans {
		dims = append(dims, dim)
	}

	sort.Strings(dims)

	parts := make([]string, 0, len(dims)+1)
	for _, dim := range dims {
		parts = append(parts, fmt.Sprintf("%s=%.1f", dim, cell.JudgeMeans[dim]))
	}

	if cell.SchemaInvalid > 0 {
		parts = append(parts, fmt.Sprintf("invalid_schema=%d/%d", cell.SchemaInvalid, cell.N))
	}

	return strings.Join(parts, " ")
}

// jsonReport is the report file's top-level shape: raw per-sample
// measurements alongside the aggregated cells, so downstream analysis can
// either recompute its own statistics from Samples or read Cells directly —
// mirroring the manifest.json convention used by the Python half of
// voice_bench (../schema.py's write_jsonl / manifest pattern).
type jsonReport struct {
	Judge       string       `json:"judge"`
	GeneratedAt time.Time    `json:"generated_at"`
	Samples     []jsonSample `json:"samples"`
	Aggregates  []Cell       `json:"aggregates"`
}

type jsonSample struct {
	Model             string             `json:"model"`
	ReasoningEffort   string             `json:"reasoning_effort"`
	Task              string             `json:"task"`
	Bucket            string             `json:"bucket"`
	RecordID          string             `json:"record_id"`
	TotalLatencyS     float64            `json:"total_latency_s"`
	TimeToFirstTokenS float64            `json:"time_to_first_token_s"`
	TokensPerSecond   float64            `json:"tokens_per_second"`
	CompletionTokens  int                `json:"completion_tokens"`
	PromptTokens      int                `json:"prompt_tokens"`
	CostUSD           float64            `json:"cost_usd"`
	Judge             map[string]float64 `json:"judge,omitempty"`
	SchemaInvalid     bool               `json:"schema_invalid,omitempty"`
	Error             string             `json:"error,omitempty"`
}

// toJSONSample converts one Sample to its on-disk form (durations become
// float64 seconds, Err becomes a plain string) -- shared by WriteJSONReport
// and the append-canonical mode (canonical.go), which both serialize Samples back
// to this shape.
func toJSONSample(s Sample) jsonSample {
	js := jsonSample{
		Model:             s.Model,
		ReasoningEffort:   s.ReasoningEffort,
		Task:              s.Task,
		Bucket:            s.Bucket,
		RecordID:          s.RecordID,
		TotalLatencyS:     roundFloat(s.TotalLatency.Seconds(), 3),
		TimeToFirstTokenS: roundFloat(s.TimeToFirstToken.Seconds(), 3),
		TokensPerSecond:   s.TokensPerSecond(),
		CompletionTokens:  s.CompletionTokens,
		PromptTokens:      s.PromptTokens,
		CostUSD:           roundFloat(s.CostUSD, 6),
		Judge:             roundJudgeScores(s.Judge),
		SchemaInvalid:     s.SchemaInvalid,
	}

	if s.Err != nil {
		js.Error = s.Err.Error()
	}

	return js
}

// roundJudgeScores returns a copy of judge with every score rounded to 3
// decimal places -- the mechanical scores (correction.go's WERs,
// dictation.go's metrics) are true fractions or percentages, not the LLM
// rubric's whole-number Likert scores, so they carry real floating-point
// noise worth rounding away. Copies rather than mutating judge in place: judge is the live Sample
// still held by the caller (see runOne), and this function's only caller
// (toJSONSample) must not have a side effect on it.
func roundJudgeScores(judge map[string]float64) map[string]float64 {
	if judge == nil {
		return nil
	}

	rounded := make(map[string]float64, len(judge))
	for dim, score := range judge {
		rounded[dim] = roundFloat(score, 3)
	}

	return rounded
}

func toJSONSamples(samples []Sample) []jsonSample {
	out := make([]jsonSample, 0, len(samples))
	for _, s := range samples {
		out = append(out, toJSONSample(s))
	}

	return out
}

// WriteJSONReport writes samples and their aggregation to a fresh,
// uniquely-named subfolder under path, as indented JSON, and returns that
// subfolder's path -- a caller chaining into the canonical merge (see
// ReviewAndMergeIntoCanonical in canonical.go) needs it to know what it just
// staged without re-deriving or globbing for it.
func WriteJSONReport(path string, samples []Sample, cells []Cell, judgeModel string) (string, error) {
	report := jsonReport{
		Judge:       judgeModel,
		GeneratedAt: time.Now().UTC(),
		Aggregates:  cells,
		Samples:     toJSONSamples(samples),
	}

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "", err
	}

	now := time.Now().Format(time.DateOnly)
	dirPath := path + "/" + now + "-" + uuid.NewString()

	if err := os.MkdirAll(dirPath, 0o777); err != nil {
		return "", err
	}

	reportPath := filepath.Join(dirPath, fmt.Sprintf("report-%s.json", judgeModel))
	if err := os.WriteFile(reportPath, data, 0o644); err != nil {
		return "", err
	}

	return dirPath, nil
}
