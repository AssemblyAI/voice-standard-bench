// Time-to-first-token (TTFT) benchmark path.
//
// This is a deliberately separate flow from the rest of the benchmark
// (voicebench.go/gateway.go's RunCompletion/judge.go/metrics.go's Cell):
// RunCompletion calls the gateway synchronously, so its TimeToFirstToken is
// always just a copy of TotalLatency (see gateway.go) — it cannot answer
// "how long until the first token comes back" for real. This file measures
// that for real, via streaming (see RunCompletionStream in gateway.go), for
// the 30s/120s buckets only (TTFTBuckets) — TTFT is a real-time-adjacent
// responsiveness signal, not the broad transcript-size sweep the quality
// benchmark runs — and never runs a judge or a schema check: a TTFT sample
// only ever measures how fast the first token arrived, never what the model
// said. It has its own sample/cell/report types (TTFTSample, TTFTCell,
// PrintTTFTReport, WriteTTFTJSONReport) and its own output paths (see
// frameworkTTFTOutputPath/frameworkTTFTErrorsPath in framework_pass.go) so running it can never change
// a single number the standard benchmark produces.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/google/uuid"
)

// TTFTBuckets is the fixed set of audio-length buckets the TTFT benchmark
// measures. Unlike the standard benchmark's buckets (whatever's actually
// present in the loaded data — see dataset.go's Buckets), this is hardcoded:
// TTFT is specifically about short-call responsiveness, not a sweep across
// every transcript size.
var TTFTBuckets = []string{"30s", "120s"}

// filterGroupsByBucket returns the subset of groups whose Bucket appears in
// buckets, preserving order. Used to restrict the standard benchmark's
// already-loaded, already-sampled TaskGroups (see dataset.go) to TTFTBuckets
// without a second dataset load.
func filterGroupsByBucket(groups []TaskGroup, buckets []string) []TaskGroup {
	allowed := make(map[string]bool, len(buckets))
	for _, b := range buckets {
		allowed[b] = true
	}

	var filtered []TaskGroup

	for _, g := range groups {
		if allowed[g.Bucket] {
			filtered = append(filtered, g)
		}
	}

	return filtered
}

// TTFTSample is one streaming completion call made purely to measure
// time-to-first-token. Unlike Sample (metrics.go), it carries no judge score,
// no schema-validity flag, and no quality signal of any kind — judging is
// deliberately skipped for this path (see this file's package doc comment).
type TTFTSample struct {
	Model string
	// ReasoningEffort mirrors Sample.ReasoningEffort (metrics.go) -- see its
	// doc comment.
	ReasoningEffort  string
	Task             string
	Bucket           string
	RecordID         string
	TimeToFirstToken time.Duration
	TotalLatency     time.Duration
	CompletionTokens int
	PromptTokens     int
	// Err is set when the streaming call itself failed or was skipped
	// (context length exceeded, non-2xx, timeout, a stream that produced no
	// content, ...). Mirrors Sample.Err's contract: kept for error-rate
	// reporting, excluded from every TTFT/latency statistic.
	Err error
}

// runTTFTOne runs one streaming completion for one model against one
// transcript and measures TTFT. It mirrors runOne's pre-flight checks
// (taskInput, contextLengthExceeded, taskMaxTokens) exactly — same input
// selection, same context-window guard, same completion budget — but stops
// there: no schema validation, no judge call, no low-accuracy recording.
// Only FailureAPIError is possible on this path (there is no schema to be
// invalid and no judge to fail), recorded via the same recordFailure
// voicebench.go's runOne uses, into the SEPARATE FailureRecorder the caller
// passes (see frameworkTTFTErrorsPath).
func runTTFTOne(ctx context.Context, gw *Gateway, model ModelInfo, task string, spec prompts.Spec, record TranscriptRecord, failures *FailureRecorder) TTFTSample {
	sample := TTFTSample{Model: model.ID, Task: task, Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	input, err := taskInput(task, record)
	if err != nil {
		sample.Err = err
		return sample
	}

	if contextLengthExceeded(model, spec.Task, input) {
		sample.Err = fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		return sample
	}

	result, err := gw.RunCompletionStream(ctx, model, spec.Task, input, taskMaxTokens(task, input, model))
	if err != nil {
		sample.Err = err

		if failures != nil {
			recordFailure(failures, model.ID, model.ReasoningEffortLabel(), task, record.Bucket, record.ID, FailureAPIError, err, result.RawRequest, RawAPIError(err))
		}

		return sample
	}

	sample.TimeToFirstToken = result.TimeToFirstToken
	sample.TotalLatency = result.TotalLatency
	sample.CompletionTokens = result.CompletionTokens
	sample.PromptTokens = result.PromptTokens

	return sample
}

// runTTFTModel runs every group's records against one model, one goroutine
// per (group, record) bounded to concurrency simultaneous in-flight
// streaming requests — the same shape as runModel, minus judgeSem (there is
// no judge call on this path at all, so no shared judge concurrency to cap).
func runTTFTModel(ctx context.Context, gw *Gateway, model ModelInfo, groups []TaskGroup, concurrency int, failures *FailureRecorder) []TTFTSample {
	if concurrency < 1 {
		concurrency = 1
	}

	sem := make(chan struct{}, concurrency)

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		samples []TTFTSample
	)

	for _, group := range groups {
		spec, ok := prompts.For(group.Task)
		if !ok {
			log.Printf("ttft: model %s: unknown task %q, skipping", model.ID, group.Task)
			continue
		}

		for _, record := range group.Records {
			wg.Add(1)
			sem <- struct{}{}

			go func(task string, spec prompts.Spec, record TranscriptRecord) {
				defer wg.Done()
				defer func() { <-sem }()

				sample := runTTFTOne(ctx, gw, model, task, spec, record, failures)

				mu.Lock()
				samples = append(samples, sample)
				mu.Unlock()
			}(group.Task, spec, record)
		}
	}

	wg.Wait()

	return samples
}

// RunTTFTBenchmark runs runTTFTModel for every model in parallel — mirroring
// main()'s per-model fan-out for the standard benchmark — and pools every
// model's samples together. groups is expected to already be filtered to
// TTFTBuckets (see filterGroupsByBucket); this function doesn't filter
// itself, so it stays usable in tests against an arbitrary group list.
func RunTTFTBenchmark(ctx context.Context, gw *Gateway, models []ModelInfo, groups []TaskGroup, concurrency int, failures *FailureRecorder) []TTFTSample {
	var (
		mu       sync.Mutex
		samples  []TTFTSample
		modelsWG sync.WaitGroup
	)

	for _, model := range models {
		modelsWG.Add(1)

		go func(model ModelInfo) {
			defer modelsWG.Done()

			results := runTTFTModel(ctx, gw, model, groups, concurrency, failures)

			mu.Lock()
			samples = append(samples, results...)
			mu.Unlock()
		}(model)
	}

	modelsWG.Wait()

	return samples
}

// TTFTCell aggregates every TTFTSample for one (model, reasoning effort,
// task, bucket) combination — the TTFT-only counterpart to Cell (metrics.go),
// minus every quality/schema field: there is nothing to judge on this path.
type TTFTCell struct {
	Model string
	// ReasoningEffort mirrors Cell.ReasoningEffort (metrics.go).
	ReasoningEffort string
	Task            string
	Bucket          string

	N      int // total samples attempted, success or failure
	Errors int // samples where the streaming call itself failed or was skipped

	TTFT    Stats // seconds; the real measurement this whole path exists for
	Latency Stats // seconds; total stream wall-clock time, kept alongside TTFT for context
}

// AggregateTTFT groups samples into one TTFTCell per distinct (Model,
// ReasoningEffort, Task, Bucket) and computes every statistic in one pass,
// sorted the same way Aggregate sorts Cells (task, then model, then
// reasoning effort, then bucket by ascending duration).
func AggregateTTFT(samples []TTFTSample) []TTFTCell {
	type key struct{ model, reasoningEffort, task, bucket string }

	groups := map[key][]TTFTSample{}

	for _, s := range samples {
		k := key{s.Model, s.ReasoningEffort, s.Task, s.Bucket}
		groups[k] = append(groups[k], s)
	}

	cells := make([]TTFTCell, 0, len(groups))

	for k, group := range groups {
		cells = append(cells, buildTTFTCell(k.model, k.reasoningEffort, k.task, k.bucket, group))
	}

	sort.Slice(cells, func(i, j int) bool {
		if cells[i].Task != cells[j].Task {
			return cells[i].Task < cells[j].Task
		}

		if cells[i].Model != cells[j].Model {
			return cells[i].Model < cells[j].Model
		}

		if cells[i].ReasoningEffort != cells[j].ReasoningEffort {
			return cells[i].ReasoningEffort < cells[j].ReasoningEffort
		}

		si, oki := bucketSeconds(cells[i].Bucket)
		sj, okj := bucketSeconds(cells[j].Bucket)

		if oki && okj {
			return si < sj
		}

		return cells[i].Bucket < cells[j].Bucket
	})

	return cells
}

func buildTTFTCell(model, reasoningEffort, task, bucket string, group []TTFTSample) TTFTCell {
	cell := TTFTCell{Model: model, ReasoningEffort: reasoningEffort, Task: task, Bucket: bucket, N: len(group)}

	var ttfts, latencies []float64

	for _, s := range group {
		if s.Err != nil {
			cell.Errors++
			continue
		}

		ttfts = append(ttfts, s.TimeToFirstToken.Seconds())
		latencies = append(latencies, s.TotalLatency.Seconds())
	}

	cell.TTFT = ComputeStats(ttfts)
	cell.Latency = ComputeStats(latencies)

	return cell
}

// PrintTTFTReport renders cells as a human-readable table, mirroring
// PrintReport's layout (report.go) minus the throughput/quality columns,
// which don't exist on this path. statTriple is report.go's helper, reused
// as-is.
func PrintTTFTReport(w io.Writer, cells []TTFTCell) {
	currentTask := ""

	for _, cell := range cells {
		if cell.Task != currentTask {
			currentTask = cell.Task
			fmt.Fprintf(w, "\n=== ttft: %s ===\n", currentTask)
			fmt.Fprintf(w, "%-32s %-10s %-8s %5s %5s  %-24s  %-24s\n",
				"MODEL", "EFFORT", "BUCKET", "N", "ERRS",
				"TTFT s (p50/p95/max)", "LATENCY s (p50/p95/max)")
		}

		fmt.Fprintf(w, "%-32s %-10s %-8s %5d %5d  %-24s  %-24s\n",
			cell.Model, cell.ReasoningEffort, cell.Bucket, cell.N, cell.Errors,
			statTriple(cell.TTFT),
			statTriple(cell.Latency),
		)
	}
}

// ttftJSONReport is the TTFT report file's top-level shape, mirroring
// jsonReport (report.go) minus Judge (this path never judges).
type ttftJSONReport struct {
	GeneratedAt time.Time        `json:"generated_at"`
	Samples     []ttftJSONSample `json:"samples"`
	Aggregates  []TTFTCell       `json:"aggregates"`
}

type ttftJSONSample struct {
	Model             string  `json:"model"`
	ReasoningEffort   string  `json:"reasoning_effort"`
	Task              string  `json:"task"`
	Bucket            string  `json:"bucket"`
	RecordID          string  `json:"record_id"`
	TimeToFirstTokenS float64 `json:"time_to_first_token_s"`
	TotalLatencyS     float64 `json:"total_latency_s"`
	CompletionTokens  int     `json:"completion_tokens"`
	PromptTokens      int     `json:"prompt_tokens"`
	Error             string  `json:"error,omitempty"`
}

// toTTFTJSONSample mirrors report.go's toJSONSample for the TTFT path's
// sample type.
func toTTFTJSONSample(s TTFTSample) ttftJSONSample {
	js := ttftJSONSample{
		Model:             s.Model,
		ReasoningEffort:   s.ReasoningEffort,
		Task:              s.Task,
		Bucket:            s.Bucket,
		RecordID:          s.RecordID,
		TimeToFirstTokenS: roundFloat(s.TimeToFirstToken.Seconds(), 3),
		TotalLatencyS:     roundFloat(s.TotalLatency.Seconds(), 3),
		CompletionTokens:  s.CompletionTokens,
		PromptTokens:      s.PromptTokens,
	}

	if s.Err != nil {
		js.Error = s.Err.Error()
	}

	return js
}

func toTTFTJSONSamples(samples []TTFTSample) []ttftJSONSample {
	out := make([]ttftJSONSample, 0, len(samples))
	for _, s := range samples {
		out = append(out, toTTFTJSONSample(s))
	}

	return out
}

// WriteTTFTJSONReport writes samples and their aggregation to a fresh,
// uniquely-named subfolder under path — mirroring the layout NewRunDir and
// WriteJSONReport (report.go) give the standard pass — as "ttft-report.json"
// (no judge-model suffix: there is no judge on this path, so no name to
// disambiguate by) -- and returns that subfolder's path, mirroring NewRunDir's
// own return for the same reason (see its doc comment). The summary tool
// (summary.go) globs for exactly this filename pattern.
//
// The subfolder is created with os.MkdirAll: the TTFT output directory (see
// frameworkTTFTOutputPath in framework_pass.go) doesn't necessarily exist yet
// the first time a TTFT pass runs, and this path has no equivalent of the
// standard pass's already-checked-in "results/" convention to lean on.
func WriteTTFTJSONReport(path string, samples []TTFTSample, cells []TTFTCell) (string, error) {
	report := ttftJSONReport{
		GeneratedAt: time.Now().UTC(),
		Aggregates:  cells,
		Samples:     toTTFTJSONSamples(samples),
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

	if err := os.WriteFile(filepath.Join(dirPath, "ttft-report.json"), data, 0o644); err != nil {
		return "", err
	}

	return dirPath, nil
}
