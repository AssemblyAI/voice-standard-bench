package main

import (
	"math"
	"sort"
	"strconv"
	"time"
)

// Sample is one completed (or failed) task-completion call: the raw
// measurements a single request produced, before any aggregation.
type Sample struct {
	Model string
	// ReasoningEffort is ModelInfo.ReasoningEffortLabel's value for the
	// model that produced this sample -- the actual -reasoning-effort value
	// when set and supported, or "default" otherwise (see
	// reasoningEffortDefaultLabel in gateway.go) -- never blank. Model and
	// ReasoningEffort together are the unique key Aggregate groups by, so
	// the same base model benchmarked at two different efforts produces two
	// distinct Cells instead of one blending both.
	ReasoningEffort  string
	Task             string
	Bucket           string
	RecordID         string
	TotalLatency     time.Duration
	TimeToFirstToken time.Duration
	CompletionTokens int
	PromptTokens     int
	// CostUSD is this call's actual USD cost, computed from PromptTokens/
	// CompletionTokens against the candidate model's global pricing (see
	// ModelInfo.Cost) -- set by runRecord (runner_internal.go), not RunRecord
	// itself, since RunRecord returns before Runner merges in the judged
	// result. Zero (not "no data") for an errored or never-judged sample.
	CostUSD float64
	// JudgeModel is the judge that produced Judge (and SchemaInvalid): a pass
	// with several judges yields one Sample per judge for each task call. Empty
	// for a TTFT sample, which is never judged.
	JudgeModel string
	// Judge holds this sample's quality scores, keyed by rubric dimension
	// name (see prompts.Spec.Rubric). Empty if judging was skipped or failed
	// independently of the task call itself.
	Judge map[string]float64
	// SchemaInvalid is set for information_extraction when the candidate's
	// output failed ValidateExtraction. It is tracked separately from Judge
	// rather than only folded into a low score, so a model that reliably
	// breaks the schema is visible as a distinct failure mode in the report
	// (see Cell.SchemaInvalid), not indistinguishable from one that follows
	// the schema but extracts poorly.
	SchemaInvalid bool
	// Err is set when the task call itself failed or was skipped (context
	// length exceeded, non-2xx, timeout, ...). A failed sample is still kept
	// — for error-rate reporting — but excluded from every latency and
	// quality statistic: a duration or score from a call that didn't
	// complete isn't a measurement of anything.
	Err error
}

// TokensPerSecond is completion tokens over total wall-clock latency — the
// whole-request throughput a caller actually experiences. Since RunCompletion
// calls the gateway synchronously (see gateway.go), there is no decode-only
// alternative to compute this against: TimeToFirstToken always equals
// TotalLatency for a synchronous response, so "subtract TTFT first" would be
// a no-op, not a different, more precise number.
func (s Sample) TokensPerSecond() float64 {
	if s.Err != nil || s.CompletionTokens <= 0 || s.TotalLatency <= 0 {
		return 0
	}

	return roundFloat(float64(s.CompletionTokens)/s.TotalLatency.Seconds(), 3)
}

// roundFloat rounds val to precision decimal places. Every float that ends
// up in a report (JSON or CSV) goes through this -- raw durations-as-seconds
// and ratios otherwise carry 9+ digits of floating-point noise that's not
// real precision, just how the arithmetic landed.
func roundFloat(val float64, precision uint) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}

// roundSignificant rounds val to digits SIGNIFICANT figures rather than a
// fixed number of decimal places: roundFloat(5.046e-07, 6) keeps one
// significant digit (1e-06) and would collapse 5.0e-07 and 8.5e-07 onto the
// same value, where roundSignificant(5.046e-07, 4) keeps 5.046e-07. It exists
// for values whose magnitude spans orders of magnitude -- see
// costPerQualityDigits in summary.go. Formatting and re-parsing through
// strconv rounds in decimal, so the result prints as exactly the short value
// it represents rather than carrying the float noise of a multiply-round-
// divide. Zero, NaN, and +/-Inf are returned unchanged.
func roundSignificant(val float64, digits int) float64 {
	if val == 0 || math.IsNaN(val) || math.IsInf(val, 0) {
		return val
	}

	rounded, err := strconv.ParseFloat(strconv.FormatFloat(val, 'g', digits, 64), 64)
	if err != nil {
		return val
	}

	return rounded
}

// Stats summarizes one metric across a set of samples, in the samples' own
// unit (seconds for latency/TTFT, tokens/second for throughput). Percentiles
// use the nearest-rank method — sort, then index by ceil(p*n) — rather than
// interpolation: simpler to reason about and to test, and the difference
// from interpolated percentiles is noise at the sample sizes this tool runs
// (tens of requests per model/task/bucket cell, not millions).
type Stats struct {
	N    int
	Min  float64
	Max  float64
	Mean float64
	P50  float64
	P95  float64
}

// ComputeStats returns the zero Stats{} for an empty input. Callers must
// check N == 0 before trusting the rest of the fields — the zero value's
// Min/Max/Mean/P50/P95 are all 0, which is indistinguishable from a real
// all-zero measurement if N is ignored.
func ComputeStats(values []float64) Stats {
	if len(values) == 0 {
		return Stats{}
	}

	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)

	sum := 0.0
	for _, v := range sorted {
		sum += v
	}

	return Stats{
		N:    len(sorted),
		Min:  roundFloat(sorted[0], 3),
		Max:  roundFloat(sorted[len(sorted)-1], 3),
		Mean: roundFloat(sum/float64(len(sorted)), 3),
		P50:  roundFloat(percentile(sorted, 0.50), 3),
		P95:  roundFloat(percentile(sorted, 0.95), 3),
	}
}

// percentile assumes sorted is already sorted ascending.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 1 {
		return sorted[0]
	}

	rank := int(math.Ceil(p*float64(len(sorted)))) - 1

	if rank < 0 {
		rank = 0
	}

	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}

	return sorted[rank]
}

// Cell aggregates every sample for one (model, reasoning effort, task,
// bucket) combination — the report's unit of comparison, so latency and
// quality can be read side by side per audio-length size.
type Cell struct {
	Model string
	// ReasoningEffort mirrors Sample.ReasoningEffort -- see its doc comment.
	ReasoningEffort string
	Task            string
	Bucket          string
	// JudgeModel mirrors Sample.JudgeModel: samples scored by different judges
	// never share a Cell. Not serialized -- a report file is already one judge's.
	JudgeModel string `json:"-"`

	N      int // total samples attempted, success or failure
	Errors int // samples where the task call itself failed or was skipped

	// SchemaInvalid counts information_extraction samples whose output
	// failed ValidateExtraction — a distinct failure mode from Errors (the
	// task call itself succeeded; what it produced didn't parse). Always 0
	// for the other two tasks.
	SchemaInvalid int

	Latency Stats // seconds
	// TTFT is always identical to Latency: RunCompletion calls the gateway
	// synchronously, so there is no earlier moment to measure a first token
	// against (see gateway.go). Kept as its own field for JSON/report schema
	// stability rather than removed — see CompletionResult.TimeToFirstToken.
	TTFT       Stats // seconds
	Throughput Stats // completion tokens / second

	// JudgeMeans is the mean score per rubric dimension across successfully
	// judged samples. A dimension absent here was never scored for this cell
	// (every sample failed the task call, or every judge call itself
	// failed) — report.go must treat a missing key as "no data", not zero.
	JudgeMeans map[string]float64
	Judged     int // samples with at least one judge score recorded
}

// Aggregate groups samples into one Cell per distinct (Model, ReasoningEffort,
// Task, Bucket) and computes every statistic in one pass. The returned slice
// is sorted by (Task, Model, ReasoningEffort, Bucket-by-duration) so a report
// can print it directly without re-sorting.
func Aggregate(samples []Sample) []Cell {
	type key struct{ model, reasoningEffort, task, bucket, judge string }

	groups := map[key][]Sample{}

	for _, s := range samples {
		k := key{s.Model, s.ReasoningEffort, s.Task, s.Bucket, s.JudgeModel}
		groups[k] = append(groups[k], s)
	}

	cells := make([]Cell, 0, len(groups))

	for k, group := range groups {
		cell := buildCell(k.model, k.reasoningEffort, k.task, k.bucket, group)
		cell.JudgeModel = k.judge
		cells = append(cells, cell)
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

		if cells[i].JudgeModel != cells[j].JudgeModel {
			return cells[i].JudgeModel < cells[j].JudgeModel
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

func buildCell(model, reasoningEffort, task, bucket string, group []Sample) Cell {
	cell := Cell{Model: model, ReasoningEffort: reasoningEffort, Task: task, Bucket: bucket, N: len(group), JudgeMeans: map[string]float64{}}

	var latencies, ttfts, throughputs []float64
	judgeSums := map[string]float64{}
	judgeCounts := map[string]int{}

	for _, s := range group {
		if s.Err != nil {
			cell.Errors++
			continue
		}

		latencies = append(latencies, s.TotalLatency.Seconds())
		ttfts = append(ttfts, s.TimeToFirstToken.Seconds())

		if tps := s.TokensPerSecond(); tps > 0 {
			throughputs = append(throughputs, tps)
		}

		if s.SchemaInvalid {
			cell.SchemaInvalid++
		}

		if len(s.Judge) > 0 {
			cell.Judged++

			for dim, score := range s.Judge {
				judgeSums[dim] += score
				judgeCounts[dim]++
			}
		}
	}

	cell.Latency = ComputeStats(latencies)
	cell.TTFT = ComputeStats(ttfts)
	cell.Throughput = ComputeStats(throughputs)

	for dim, sum := range judgeSums {
		cell.JudgeMeans[dim] = roundFloat(sum/float64(judgeCounts[dim]), 3)
	}

	return cell
}
