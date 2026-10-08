// Summarize mode (RunSummary below): reads every benchmark report JSON
// (see report.go) in a results directory and summarizes
// them two ways: by judge model, and combined across every judge. Each row
// is a (model, reasoning effort, bucket) triple, collapsed across every
// task, with a single 1-100 quality_score distilling every judge dimension
// into one number, a latency breakdown (min/max/mean/p50/p95), and mean
// tokens/second. reasoning_effort is "default" for a row whose model was
// benchmarked without -reasoning-effort (or that didn't support it) rather
// than blank, so the same model run at two different efforts always lands
// as two distinct rows, never one blending both (see
// ModelInfo.ReasoningEffortLabel in gateway.go). Passing a
// TTFT results directory merges a real time-to-first-token latency
// breakdown (see ttft.go) into every row that shares its (model, reasoning
// effort, bucket) — blank for a bucket TTFT's separate streaming-only path
// didn't measure.
//
// Every row ALSO carries a task_quality breakdown alongside quality_score:
// the same 1-100 score, but computed from only that task's own judged
// samples within the row, one column per task actually present in the data
// (see summarize and, for the CSV, taskColumns) — so filtering the one
// summary-*.csv/.json for a (model, bucket) shows both the combined score
// across every task and each task's own score side by side, with no
// separate files to cross-reference.
//
// Every row also carries total_cost_usd/cost_per_quality alongside
// quality_score. total_cost_usd is the real USD spent on the judged samples
// behind that score (tokens actually used times the candidate's global
// per-token pricing, captured when the model list was pulled — see
// ModelInfo.Cost in gateway.go). cost_per_quality is the MEAN cost of one
// judged sample, in cents, divided by the score, never the total: a total scales with
// how many samples a row happens to hold (a different -max-samples-per-bucket
// between runs, or a model that errored on half its calls), which would make
// a row with fewer samples look cheaper for no reason a reader could act on.
// Models can then be compared on quality per dollar, not quality alone.
// task_cost_usd/task_cost_per_quality mirror task_quality: the same
// breakdown restricted to one task's own samples (its own mean, its own
// score).
package main

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

type sourceReport struct {
	Judge   string         `json:"judge"`
	Samples []sourceSample `json:"samples"`
}

type sourceSample struct {
	Model string `json:"model"`
	// ReasoningEffort mirrors Sample.ReasoningEffort (metrics.go) -- "default"
	// when -reasoning-effort wasn't set or the model didn't support it,
	// never blank. Model+ReasoningEffort is the key summarize groups by,
	// alongside Bucket.
	ReasoningEffort string `json:"reasoning_effort"`
	Bucket          string `json:"bucket"`
	// Task is read by summarize to build each entry's TaskQuality
	// breakdown — see summarize's doc comment. It does NOT change what
	// (model, bucket) a sample groups into: every task's samples still land
	// in the same row; Task determines how quality_score is weighted (each
	// task equally, see summarize) and adds a per-task view of quality
	// alongside that combined one.
	Task            string  `json:"task"`
	TotalLatencyS   float64 `json:"total_latency_s"`
	TokensPerSecond float64 `json:"tokens_per_second"`
	// CostUSD is this sample's actual USD cost (report.go's jsonSample.CostUSD
	// -- tokens actually used times the candidate's global per-token pricing,
	// captured when the model list was pulled). Averaged into CostPerQuality
	// below over the same population QualityScore itself pools.
	CostUSD float64            `json:"cost_usd"`
	Judge   map[string]float64 `json:"judge"`
	Error   string             `json:"error"`
}

type scale struct{ lo, hi float64 }

// dimensionRange gives the raw [lo, hi] scale of every score that counts
// toward quality, hi always being the best score. Every LLM-graded dimension
// (faithfulness, coverage, conciseness, groundedness, completeness,
// improvement, and text_correction's reference-based meaning_similarity) is a
// 1-5 Likert scale. word_error_rate (dictation.go) is absent because LOWER is
// better, so normalize inverts it separately below instead of listing it
// here — dictation's other four mechanical scores (exact_match, a plain 0/1,
// and punctuation_accuracy/capitalization_accuracy/formatting_command_accuracy,
// already stored as 0-100 percentages so a raw report reads directly as
// "87.5% accurate") DO fit the plain range shape and are listed here like
// every LLM dimension.
//
// A score recorded in a report but absent from here is informational and
// never reaches quality_score: text_correction's truth_wer/input_wer are word-
// level context for reading meaning_similarity (a do-nothing input's near-zero
// WER would otherwise be averaged in as free quality), and older reports carry
// scores text_correction no longer scores -- accuracy (an order-insensitive
// word overlap that rated a no-op 0.955-0.997), edit_ratio (distance from the
// input, which a no-op minimizes), and error_reduction (a word-for-word
// comparison with a truth that keeps the fillers the prompt says to remove).
// Leaving those out is what stops an old report's credit for them inflating
// -- or its penalty deflating -- its text_correction quality.
var dimensionRange = map[string]scale{
	"faithfulness":                {1, 5},
	"coverage":                    {1, 5},
	"conciseness":                 {1, 5},
	"groundedness":                {1, 5},
	"completeness":                {1, 5},
	"improvement":                 {1, 5},
	"meaning_similarity":          {1, 5},
	"exact_match":                 {0, 1},
	"punctuation_accuracy":        {0, 100},
	"capitalization_accuracy":     {0, 100},
	"formatting_command_accuracy": {0, 100},
}

// normalize maps a raw judge score to [0, 1], 1 being the best possible
// result on that dimension. An unrecognized name reports ok=false so it's
// excluded rather than silently mis-scaled.
func normalize(dimension string, value float64) (score float64, ok bool) {
	if dimension == "word_error_rate" {
		// A 0-100 percentage, like the other three mechanical dictation
		// scores, but LOWER is better — a shape mismatch with the plain
		// dimensionRange scale, so it gets a special-cased inversion
		// rather than a {0, 100} entry above (which
		// would score a WORSE transcript as higher quality). clamp guards
		// against word_error_rate exceeding 100 (a real, expected outcome
		// when a candidate inserts far more words than the truth text has —
		// see wordErrorRate's doc comment in dictation.go), which would
		// otherwise push this below 0 and drag a quality_score negative.
		return clamp(1-value/100, 0, 1), true
	}

	r, ok := dimensionRange[dimension]
	if !ok {
		return 0, false
	}

	return (value - r.lo) / (r.hi - r.lo), true
}

// sampleQuality averages one sample's judge dimensions, each normalized to
// [0, 1], into a single per-sample score. Averaging within the sample
// first — before summarize averages across samples — keeps every sample
// weighted equally regardless of how many rubric dimensions its task
// carries (text_correction has three, information_extraction two).
func sampleQuality(judge map[string]float64) (score float64, ok bool) {
	var sum float64

	var n int

	for dimension, value := range judge {
		if normalized, ok := normalize(dimension, value); ok {
			sum += normalized
			n++
		}
	}

	if n == 0 {
		return 0, false
	}

	return sum / float64(n), true
}

// stats mirrors metrics.go's Stats: N==0 means no sample contributed and
// every other field is meaningless, not a real zero measurement. It carries
// its own JSON tags for this mode's summary.json output, which Stats itself
// doesn't need.
type stats struct {
	N    int     `json:"n"`
	Min  float64 `json:"min"`
	Max  float64 `json:"max"`
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P95  float64 `json:"p95"`
}

// computeStats duplicates metrics.go's ComputeStats exactly (same
// nearest-rank method, via the shared percentile helper) rather than
// reusing it directly: it returns this file's own JSON-tagged stats, not
// metrics.go's Stats.
func computeStats(values []float64) stats {
	if len(values) == 0 {
		return stats{}
	}

	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)

	var sum float64
	for _, v := range sorted {
		sum += v
	}

	return stats{
		N:    len(sorted),
		Min:  roundFloat(sorted[0], 3),
		Max:  roundFloat(sorted[len(sorted)-1], 3),
		Mean: roundFloat(sum/float64(len(sorted)), 3),
		P50:  roundFloat(percentile(sorted, 0.50), 3),
		P95:  roundFloat(percentile(sorted, 0.95), 3),
	}
}

// costPerQualityDigits is how many SIGNIFICANT digits CostPerQuality and
// TaskCostPerQuality keep, instead of roundFloat's fixed decimal places: a mean
// cost per judged call over a 1-100 score is routinely a fraction of a
// hundred-thousandth of a cent (5e-05 for a small model on a 30s transcript), where
// six decimal places keep a single significant digit, collapse distinct
// models onto one value, and can round all the way to 0 -- which every
// consumer reads as "no cost data" (see csvFloat, costPerQualityString).
const costPerQualityDigits = 4

// centsPerDollar converts the USD costSum into the cents cost_per_quality is
// reported in.
const centsPerDollar = 100

// meanPerQualityPoint is costSum's mean over n judged samples, per point of
// score, in CENTS: the one place cost_per_quality's arithmetic lives, shared by the
// combined rollup and every per-task one so the two can never diverge. n and
// score are always >= 1 at every call site (a group only reaches it with a
// judged sample, and the 1 + fraction*99 formula never scores below 1).
func meanPerQualityPoint(costSum float64, n, score int) float64 {
	return costSum * centsPerDollar / float64(n) / float64(score)
}

// formatCents renders a per-call cents figure with costPerQualityDigits
// significant digits -- "5.046e-05¢", "0.09865¢" -- rather than a fixed number
// of decimal places, which would print every cheap row as "0.000001¢".
func formatCents(v float64) string {
	return strconv.FormatFloat(v, 'g', costPerQualityDigits, 64) + "¢"
}

type summaryEntry struct {
	Model string `json:"model"`
	// ReasoningEffort mirrors sourceSample.ReasoningEffort -- see its doc
	// comment. Part of summarize's grouping key alongside Model and Bucket.
	ReasoningEffort string `json:"reasoning_effort"`
	Bucket          string `json:"bucket"`
	QualityScore    int    `json:"quality_score,omitempty"`
	// TotalCostUSD sums CostUSD (sourceSample) across every sample pooled
	// into QualityScore. It is informational only -- a total scales with how
	// many samples the row holds, so CostPerQuality is built from the mean
	// instead, never from this.
	TotalCostUSD float64 `json:"total_cost_usd,omitempty"`
	// CostPerQuality is the MEAN CostUSD of one judged sample (TotalCostUSD
	// over the number of samples pooled into QualityScore) divided by
	// QualityScore: USD per call, per 1-100 quality point. Averaging first
	// keeps it comparable across models whose rows hold different sample
	// counts, and across the same bucket/task since quality_score is always
	// on that same normalized scale. Rounded to costPerQualityDigits
	// significant digits, not decimal places -- see there.
	CostPerQuality float64 `json:"cost_per_quality,omitempty"`
	Latency        stats   `json:"latency"`
	// ErrorCount is how many samples in this (model, reasoning effort,
	// bucket) group carried a non-empty Error -- the same population
	// excluded from Latency/TokensPerSecond/QualityScore above, counted here
	// instead of silently dropped so a row with a high failure rate is
	// visible even though its other numbers still look fine.
	ErrorCount      int     `json:"error_count,omitempty"`
	TokensPerSecond float64 `json:"tokens_per_second,omitempty"`
	// TTFT is the time-to-first-token latency breakdown from the SEPARATE
	// streaming-only benchmark path (see ttft.go), merged in by model+
	// bucket (see attachTTFT) after summarize/summarizeByJudge build this
	// entry from the standard benchmark's samples alone — those two never
	// share a sample, so TTFT is always attached as a second pass. Left at
	// its zero value (stats{}, N==0) for a bucket the TTFT run never
	// measured (see ttft.go's TTFTBuckets) or when no TTFT results dir
	// was passed at all, per the usual "N==0 means no data" convention.
	TTFT stats `json:"ttft"`
	// TaskQuality is QualityScore's per-task breakdown for this SAME (model,
	// bucket) row: one 1-100 score per task that contributed a judged
	// sample, computed from that task's samples within the group alone (see
	// summarize) — so summarization's score here is never diluted by
	// text_correction's or information_extraction's. The row's combined
	// QualityScore is the equal-weight mean of these per-task scores. A task with no judged samples
	// in this row has no key here at all (nil/absent when the row has no
	// per-task data yet, e.g. no samples carry a Task),
	// never a misleading 0.
	TaskQuality map[string]int `json:"task_quality,omitempty"`
	// TaskCostUSD/TaskCostPerQuality mirror TotalCostUSD/CostPerQuality,
	// restricted to one task's own samples -- the same relationship
	// TaskQuality has to QualityScore (TaskCostPerQuality is that task's own
	// mean cost per judged sample over its own score). Keyed identically to
	// TaskQuality (a task absent from TaskQuality is absent from these too).
	TaskCostUSD        map[string]float64 `json:"task_cost_usd,omitempty"`
	TaskCostPerQuality map[string]float64 `json:"task_cost_per_quality,omitempty"`
}

// summarize groups samples by (model, reasoning effort, bucket) — across
// every task — and
// reduces each group to a quality_score (1-100, 100 best), a latency
// breakdown in seconds (min/max/mean/p50/p95), mean tokens/second, a
// cost_per_quality (see below), and a task_quality breakdown (see
// summaryEntry.TaskQuality): the SAME row also carries, alongside
// quality_score, one more 1-100 score per task computed from only that
// task's own samples within the group — so a reader gets both the
// combined-across-every-task number and each task's own number in the same
// place, without a second file to open. A sample with a non-empty Error
// never completed its task call, so its latency/throughput fields are
// meaningless zeros and it's excluded from both; it naturally carries no
// Judge either, so it's excluded from quality (both the combined score and
// its own task's score) the same way. A group with no contributing samples
// is left at its zero value — stats{} for Latency (self-describing via
// N==0, same convention as metrics.go's Stats) and an omitted field for
// QualityScore/TokensPerSecond/TaskQuality (see summaryEntry) — rather than
// a misleading 0.
//
// QualityScore weights every task equally: it is the mean of each task's own
// mean sample quality, not a mean over all samples, so a task with many
// samples doesn't outweigh one with few. A task with no judged samples in
// the group (not run at all, or every call failed) is excluded from that
// mean rather than counted as 0.
//
// CostUSD is accumulated over the exact same population as quality (inside
// the sampleQuality ok branch below), not every non-errored sample: cost_per_
// quality relates the dollars those SAME calls cost to the score they
// produced, so its numerator and denominator must share one population.
// The numerator is the MEAN cost of that population (costSum / qualityN, and
// per task taskCostSum / taskQualityN), not the sum: a sum grows with the
// number of samples in the group, so a model with fewer judged samples --
// because its run used a smaller -max-samples-per-bucket, or because more of
// its calls errored -- would look cheaper than an identical model with more.
// QualityScore is never 0 when qualityN > 0 (the 1 + fraction*99 formula's
// range is [1,100]), so dividing by it is never a divide-by-zero.
func summarize(samples []sourceSample) []summaryEntry {
	type key struct{ model, reasoningEffort, bucket string }

	groups := map[key][]sourceSample{}

	for _, s := range samples {
		k := key{s.Model, s.ReasoningEffort, s.Bucket}
		groups[k] = append(groups[k], s)
	}

	entries := make([]summaryEntry, 0, len(groups))

	for k, group := range groups {
		entry := summaryEntry{Model: k.model, ReasoningEffort: k.reasoningEffort, Bucket: k.bucket}

		var latencies []float64

		var throughputSum, costSum float64

		var throughputN, qualityN, errorCount int

		taskQualitySum := map[string]float64{}
		taskQualityN := map[string]int{}
		taskCostSum := map[string]float64{}

		for _, s := range group {
			if s.Error == "" {
				latencies = append(latencies, s.TotalLatencyS)
				throughputSum += s.TokensPerSecond
				throughputN++
			} else {
				errorCount++
			}

			if q, ok := sampleQuality(s.Judge); ok {
				qualityN++
				costSum += s.CostUSD

				taskQualitySum[s.Task] += q
				taskQualityN[s.Task]++
				taskCostSum[s.Task] += s.CostUSD
			}
		}

		entry.Latency = computeStats(latencies)
		entry.ErrorCount = errorCount

		if throughputN > 0 {
			entry.TokensPerSecond = roundFloat(throughputSum/float64(throughputN), 3)
		}

		if qualityN > 0 {
			// Every task that contributed a judged sample counts equally,
			// regardless of how many samples it ran; a task with none is
			// absent from taskQualityN and so never enters the mean.
			var taskMeanSum float64

			for task, n := range taskQualityN {
				taskMeanSum += taskQualitySum[task] / float64(n)
			}

			entry.QualityScore = int(math.Round(1 + (taskMeanSum/float64(len(taskQualityN)))*99))
			entry.TotalCostUSD = roundFloat(costSum, 6)
			entry.CostPerQuality = roundSignificant(meanPerQualityPoint(costSum, qualityN, entry.QualityScore), costPerQualityDigits)
		}

		if len(taskQualityN) > 0 {
			entry.TaskQuality = make(map[string]int, len(taskQualityN))
			entry.TaskCostUSD = make(map[string]float64, len(taskQualityN))
			entry.TaskCostPerQuality = make(map[string]float64, len(taskQualityN))

			for task, n := range taskQualityN {
				taskScore := int(math.Round(1 + (taskQualitySum[task]/float64(n))*99))
				entry.TaskQuality[task] = taskScore
				entry.TaskCostUSD[task] = roundFloat(taskCostSum[task], 6)
				entry.TaskCostPerQuality[task] = roundSignificant(meanPerQualityPoint(taskCostSum[task], n, taskScore), costPerQualityDigits)
			}
		}

		entries = append(entries, entry)
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Model != entries[j].Model {
			return entries[i].Model < entries[j].Model
		}

		if entries[i].ReasoningEffort != entries[j].ReasoningEffort {
			return entries[i].ReasoningEffort < entries[j].ReasoningEffort
		}

		si, oki := bucketSeconds(entries[i].Bucket)
		sj, okj := bucketSeconds(entries[j].Bucket)

		if oki && okj {
			return si < sj
		}

		return entries[i].Bucket < entries[j].Bucket
	})

	return entries
}

// reportFilenamePattern matches report.go's WriteJSONReport naming
// (report-<RFC3339>.json), optionally followed by a hand- or externally
// -added "-<judge>" suffix, e.g. "report-2026-08-27T18:06:45-06:00-claude
// -opus-5.json". Capture group 1 is the judge suffix, empty when absent.
var reportFilenamePattern = regexp.MustCompile(`^report-\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})(?:-(.+))?\.json$`)

// unknownJudge labels a report that names its judge nowhere this tool can
// find it — neither the JSON body nor the filename. It's kept as its own
// group rather than guessed at (e.g. defaulting to voicebench.go's
// -judge-model default): a wrong guess would silently misattribute real
// data to a judge that never produced it.
const unknownJudge = "unknown"

// judgeFor determines which model judged report, preferring its own "judge"
// field — added after these tools were first written, so authoritative when
// present — and falling back to a "-<judge>" filename suffix for older
// files written before that field existed.
func judgeFor(path string, report sourceReport) string {
	if report.Judge != "" {
		return report.Judge
	}

	if m := reportFilenamePattern.FindStringSubmatch(filepath.Base(path)); m != nil && m[1] != "" {
		return m[1]
	}

	return unknownJudge
}

// judgedSample pairs a sample with the judge model that scored it, so
// summarizeByJudge can group by judge and summarize can still pool
// everything together (see RunSummary) without either needing to know about
// the other.
type judgedSample struct {
	judge string
	sourceSample
}

// loadReports reads every report-*.json file in dir. A sample whose model
// equals its own report's judge is dropped: the benchmark never lets a
// judge grade itself (see the self-preference-bias note on -judge-model in
// voicebench.go), so such a sample can only come from a stale or hand-edited
// file, and keeping it would silently reintroduce exactly the bias that
// exclusion exists to avoid.
func loadReports(dir string) (names []string, samples []judgedSample, err error) {
	paths, err := filepath.Glob(filepath.Join(dir, "report-*.json"))
	if err != nil {
		return nil, nil, fmt.Errorf("invalid results dir %q: %w", dir, err)
	}

	if len(paths) == 0 {
		return nil, nil, fmt.Errorf("no report-*.json files found in %q", dir)
	}

	sort.Strings(paths)

	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, nil, err
		}

		var report sourceReport
		if err := json.Unmarshal(data, &report); err != nil {
			return nil, nil, fmt.Errorf("parsing %s: %w", path, err)
		}

		judge := judgeFor(path, report)
		names = append(names, filepath.Base(path))

		for _, s := range report.Samples {
			if s.Model == judge {
				continue
			}

			samples = append(samples, judgedSample{judge: judge, sourceSample: s})
		}
	}

	return names, samples, nil
}

// ttftSourceReport/ttftSourceSample are the slice of ttft.go's
// ttft-report.json this mode reads: no Judge field anywhere, because the
// TTFT path (see ttft.go's package doc comment) never judges — it exists
// purely to measure real time-to-first-token via streaming, which
// report.go's synchronous path can't produce (TimeToFirstToken there is
// always a copy of TotalLatency).
type ttftSourceReport struct {
	Samples []ttftSourceSample `json:"samples"`
}

type ttftSourceSample struct {
	Model             string  `json:"model"`
	ReasoningEffort   string  `json:"reasoning_effort"`
	Bucket            string  `json:"bucket"`
	TimeToFirstTokenS float64 `json:"time_to_first_token_s"`
	Error             string  `json:"error"`
}

// ttftReportFilenamePattern matches ttft.go's WriteTTFTJSONReport naming
// (always exactly "ttft-report.json", no timestamp or judge suffix — there's
// only ever one file per run directory, unlike report.go's report-*.json
// which can carry a judge suffix).
var ttftReportFilenamePattern = "ttft-report*.json"

// loadTTFTReports reads every ttft-report*.json file in dir — the SEPARATE
// output directory the TTFT pass stages into (see frameworkTTFTOutputPath in
// framework_pass.go), never mixed with the standard pass's report-*.json
// files even if both happen to live under the same parent, matching how
// ttft.go's output/errors directories never share a directory with the
// standard benchmark's.
func loadTTFTReports(dir string) (names []string, samples []ttftSourceSample, err error) {
	paths, err := filepath.Glob(filepath.Join(dir, ttftReportFilenamePattern))
	if err != nil {
		return nil, nil, fmt.Errorf("invalid ttft results dir %q: %w", dir, err)
	}

	if len(paths) == 0 {
		return nil, nil, fmt.Errorf("no %s files found in %q", ttftReportFilenamePattern, dir)
	}

	sort.Strings(paths)

	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, nil, err
		}

		var report ttftSourceReport
		if err := json.Unmarshal(data, &report); err != nil {
			return nil, nil, fmt.Errorf("parsing %s: %w", path, err)
		}

		names = append(names, filepath.Base(path))
		samples = append(samples, report.Samples...)
	}

	return names, samples, nil
}

// modelBucketKey identifies one (model, reasoning effort, bucket) triple —
// the join key between the standard benchmark's summaryEntry rows and the
// TTFT path's samples, which share no sample with each other but do share
// this key space (see attachTTFT).
type modelBucketKey struct{ model, reasoningEffort, bucket string }

// ttftLatencyByModelBucket groups ttftSourceSamples by (model, reasoning
// effort, bucket) and reduces each group to a TTFT latency breakdown in
// seconds (min/max/mean/p50/p95), the same error-exclusion summarize itself
// uses. The result is a lookup table, not a report shape of its own:
// attachTTFT merges it directly into the standard summaryEntry rows rather
// than this mode carrying a second, parallel set of rows to print or export.
func ttftLatencyByModelBucket(samples []ttftSourceSample) map[modelBucketKey]stats {
	groups := map[modelBucketKey][]float64{}

	for _, s := range samples {
		if s.Error != "" {
			continue
		}

		k := modelBucketKey{s.Model, s.ReasoningEffort, s.Bucket}
		groups[k] = append(groups[k], s.TimeToFirstTokenS)
	}

	byModelBucket := make(map[modelBucketKey]stats, len(groups))

	for k, values := range groups {
		byModelBucket[k] = computeStats(values)
	}

	return byModelBucket
}

// attachTTFT sets each entry's TTFT field by (model, reasoning effort,
// bucket) lookup into ttft, mutating entries in place. A key absent from
// ttft — because the TTFT run never measured that bucket (see ttft.go's
// TTFTBuckets) or because no TTFT results dir was passed at all
// (ttft is nil, and a nil map reads as always-missing) — is left at its zero
// value, which is exactly what it already was: attachTTFT only ever narrows
// "no data" into "no data for THIS row", never invents one where summarize/
// summarizeByJudge didn't already leave the field blank.
func attachTTFT(entries []summaryEntry, ttft map[modelBucketKey]stats) {
	for i := range entries {
		entries[i].TTFT = ttft[modelBucketKey{entries[i].Model, entries[i].ReasoningEffort, entries[i].Bucket}]
	}
}

type judgeSummary struct {
	Judge   string         `json:"judge"`
	Summary []summaryEntry `json:"summary"`
}

// summarizeByJudge partitions samples by their judge and runs summarize
// independently within each partition, so e.g. two separate runs against
// the same judge model pool together while different judges stay apart.
func summarizeByJudge(samples []judgedSample) []judgeSummary {
	byJudge := map[string][]sourceSample{}

	for _, s := range samples {
		byJudge[s.judge] = append(byJudge[s.judge], s.sourceSample)
	}

	summaries := make([]judgeSummary, 0, len(byJudge))

	for judge, group := range byJudge {
		summaries = append(summaries, judgeSummary{Judge: judge, Summary: summarize(group)})
	}

	sort.Slice(summaries, func(i, j int) bool { return summaries[i].Judge < summaries[j].Judge })

	return summaries
}

type summaryReport struct {
	GeneratedAt time.Time      `json:"generated_at"`
	Sources     []string       `json:"sources"`
	ByJudge     []judgeSummary `json:"by_judge"`
	// Combined pools every judge's samples into one summarize call per
	// (model, bucket) — the "combination of all of the judges" score,
	// alongside (not instead of) each judge's own breakdown in ByJudge.
	Combined []summaryEntry `json:"combined"`
	// TTFTSources names the ttft-report*.json files every entry's TTFT
	// field (see summaryEntry) was merged in from — omitted (nil) when
	// no TTFT results dir was passed, in which case every entry's
	// TTFT is left at its zero value.
	TTFTSources []string `json:"ttft_sources,omitempty"`
}

// printTable renders out as a human-readable table, one section per judge
// plus a final combined section, mirroring report.go's PrintReport. Each
// row includes its TTFT column regardless of section — TTFT is attached
// per (model, bucket) (see attachTTFT), not scoped to a judge, so there is
// no separate ttft section the way there briefly was.
func printTable(w io.Writer, out summaryReport) {
	for _, js := range out.ByJudge {
		fmt.Fprintf(w, "\n=== judge: %s ===\n", js.Judge)
		printEntries(w, js.Summary)
	}

	fmt.Fprintf(w, "\n=== combined (all judges) ===\n")
	printEntries(w, out.Combined)
}

func printEntries(w io.Writer, entries []summaryEntry) {
	fmt.Fprintf(w, "%-28s %-10s %-8s %7s  %10s  %-52s  %8s  %-52s  %s  %s\n",
		"MODEL", "EFFORT", "BUCKET", "QUALITY", "¢/QUALITY", "LATENCY s", "TOK/S", "TTFT s", "TASK QUALITY", "TASK ¢/QUALITY")

	for _, e := range entries {
		fmt.Fprintf(w, "%-28s %-10s %-8s %7s  %10s  %-52s  %8s  %-52s  %s  %s\n",
			e.Model, e.ReasoningEffort, e.Bucket, qualityString(e), costPerQualityString(e), latencyString(e.Latency), throughputString(e), latencyString(e.TTFT), taskQualityString(e), taskCostPerQualityString(e))
	}
}

func qualityString(e summaryEntry) string {
	if e.QualityScore == 0 {
		return "n/a"
	}

	return strconv.Itoa(e.QualityScore)
}

// costPerQualityString renders e.CostPerQuality, gated on QualityScore
// rather than CostPerQuality itself: a 0¢ row with real judged samples
// (a genuinely free/unpriced model) must still print, not read as "n/a".
func costPerQualityString(e summaryEntry) string {
	if e.QualityScore == 0 {
		return "n/a"
	}

	return formatCents(e.CostPerQuality)
}

// taskQualityString renders e.TaskQuality as "task=score" pairs, sorted by
// task name for a stable, diffable order — the printed-table counterpart to
// taskColumns/csvRow's CSV columns for the same data.
func taskQualityString(e summaryEntry) string {
	if len(e.TaskQuality) == 0 {
		return "n/a"
	}

	tasks := make([]string, 0, len(e.TaskQuality))
	for task := range e.TaskQuality {
		tasks = append(tasks, task)
	}

	sort.Strings(tasks)

	parts := make([]string, 0, len(tasks))
	for _, task := range tasks {
		parts = append(parts, fmt.Sprintf("%s=%d", task, e.TaskQuality[task]))
	}

	return strings.Join(parts, " ")
}

// taskCostPerQualityString mirrors taskQualityString for e.TaskCostPerQuality.
func taskCostPerQualityString(e summaryEntry) string {
	if len(e.TaskCostPerQuality) == 0 {
		return "n/a"
	}

	tasks := make([]string, 0, len(e.TaskCostPerQuality))
	for task := range e.TaskCostPerQuality {
		tasks = append(tasks, task)
	}

	sort.Strings(tasks)

	parts := make([]string, 0, len(tasks))
	for _, task := range tasks {
		parts = append(parts, fmt.Sprintf("%s=%s", task, formatCents(e.TaskCostPerQuality[task])))
	}

	return strings.Join(parts, " ")
}

func throughputString(e summaryEntry) string {
	if e.TokensPerSecond == 0 {
		return "n/a"
	}

	return fmt.Sprintf("%.1f", e.TokensPerSecond)
}

func latencyString(s stats) string {
	if s.N == 0 {
		return "no data"
	}

	return fmt.Sprintf("min %.2f / mean %.2f / p50 %.2f / p95 %.2f / max %.2f", s.Min, s.Mean, s.P50, s.P95, s.Max)
}

// csvHeader is the FIXED prefix every row shares. writeCSV appends
// one quality_<task> column per task actually present in out (see
// taskColumns) after this — the only part of the header/row shape that
// varies with the input, since which tasks were benchmarked isn't known
// ahead of time.
var csvHeader = []string{
	"judge", "model", "reasoning_effort", "bucket", "quality_score", "total_cost_usd", "cost_per_quality",
	// error_count sits right after latency_n -- how many of that same
	// group's samples carried a non-empty Error, excluded from every stat
	// alongside it (see summaryEntry.ErrorCount). Unlike the blank-for-zero
	// convention every other numeric column here follows, error_count is
	// ALWAYS rendered as a literal integer, including 0 -- see csvRow.
	"latency_n", "error_count", "latency_min", "latency_mean", "latency_p50", "latency_p95", "latency_max",
	"tokens_per_second",
	// ttft_* mirrors latency_*'s shape but for the SEPARATE streaming-only
	// benchmark path (see ttft.go), merged in by (model, reasoning effort,
	// bucket) — see attachTTFT. Blank on every row for a bucket the TTFT run
	// never measured (see ttft.go's TTFTBuckets) or when
	// no TTFT results dir was passed at all — never "0", per
	// csvRow's usual convention.
	"ttft_n", "ttft_min", "ttft_mean", "ttft_p50", "ttft_p95", "ttft_max",
}

// combinedJudgeLabel tags Combined's rows in the CSV output, the flat
// format's stand-in for the JSON output's separate top-level field.
const combinedJudgeLabel = "combined"

// taskColumns returns every task name appearing in any entry's TaskQuality
// map across out — ByJudge and Combined alike — sorted for a stable column
// order. Discovered from the data itself rather than hardcoded (this mode
// deliberately doesn't import prompts/ — see computeStats' doc comment on
// why it keeps its own stats type instead), so a task the benchmark adds
// later needs no code change here to get its own column.
func taskColumns(out summaryReport) []string {
	seen := map[string]bool{}

	collect := func(entries []summaryEntry) {
		for _, e := range entries {
			for task := range e.TaskQuality {
				seen[task] = true
			}
		}
	}

	for _, js := range out.ByJudge {
		collect(js.Summary)
	}

	collect(out.Combined)

	tasks := make([]string, 0, len(seen))
	for task := range seen {
		tasks = append(tasks, task)
	}

	sort.Strings(tasks)

	return tasks
}

// taskQualityHeaders renders tasks as their "quality_<task>" CSV column
// names, in the same order csvRow appends their values in.
func taskQualityHeaders(tasks []string) []string {
	headers := make([]string, len(tasks))

	for i, task := range tasks {
		headers[i] = "quality_" + task
	}

	return headers
}

// taskCostHeaders renders tasks as "<prefix>_<task>" CSV column names —
// shared by the two per-task cost blocks (cost_usd_<task> and
// cost_per_quality_<task>), the same way taskQualityHeaders does for
// quality_<task>.
func taskCostHeaders(prefix string, tasks []string) []string {
	headers := make([]string, len(tasks))

	for i, task := range tasks {
		headers[i] = prefix + "_" + task
	}

	return headers
}

// writeCSV flattens out into one row per (judge, model, bucket) — including
// a combinedJudgeLabel block for Combined — so the same data ByJudge/
// Combined split in the JSON output can be pivoted or filtered in a
// spreadsheet. A cell is left blank rather than "0" wherever the
// corresponding JSON field would be omitted (see summaryEntry/stats): a
// blank reads as "no data" to spreadsheet tools and is excluded from
// AVERAGE/SUM, where a literal 0 would silently corrupt either. tasks is
// computed once up front (see taskColumns) so every row, by-judge or
// combined, gets the exact same trailing quality_<task>/cost_usd_<task>/
// cost_per_quality_<task> columns in the exact same order.
func writeCSV(path string, out summaryReport) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)

	tasks := taskColumns(out)

	header := append(append([]string{}, csvHeader...), taskQualityHeaders(tasks)...)
	header = append(header, taskCostHeaders("cost_usd", tasks)...)
	header = append(header, taskCostHeaders("cost_per_quality", tasks)...)

	if err := w.Write(header); err != nil {
		return err
	}

	for _, js := range out.ByJudge {
		for _, e := range js.Summary {
			if err := w.Write(csvRow(js.Judge, e, tasks)); err != nil {
				return err
			}
		}
	}

	for _, e := range out.Combined {
		if err := w.Write(csvRow(combinedJudgeLabel, e, tasks)); err != nil {
			return err
		}
	}

	w.Flush()

	return w.Error()
}

func csvRow(judge string, e summaryEntry, tasks []string) []string {
	row := []string{judge, e.Model, e.ReasoningEffort, e.Bucket, csvInt(e.QualityScore), csvFloat(e.TotalCostUSD), csvFloat(e.CostPerQuality)}

	latencyCells := statsCSVCells(e.Latency)
	// error_count, unlike every csvInt/statsCSVCells cell around it, is never
	// blanked for a zero value: every group summarize builds has at least one
	// sample (see summarize), so error_count is always a real, fully-defined
	// count from 0..N -- never the "no data collected for this row" case the
	// blank convention exists to signal elsewhere. Blanking 0 here would mix
	// blank cells with digit strings in the same column, which is exactly
	// what makes a CSV reader (pandas, Excel) infer the column as float64
	// (NaN-coerced) instead of a clean int.
	row = append(row, latencyCells[0], strconv.Itoa(e.ErrorCount))
	row = append(row, latencyCells[1:]...)
	row = append(row, csvFloat(e.TokensPerSecond))
	row = append(row, statsCSVCells(e.TTFT)...)

	for _, task := range tasks {
		if score, ok := e.TaskQuality[task]; ok {
			row = append(row, strconv.Itoa(score))
		} else {
			row = append(row, "")
		}
	}

	for _, task := range tasks {
		row = append(row, csvTaskCost(e.TaskQuality, e.TaskCostUSD, task))
	}

	for _, task := range tasks {
		row = append(row, csvTaskCost(e.TaskQuality, e.TaskCostPerQuality, task))
	}

	return row
}

// csvTaskCost renders costs[task], blank if task has no TaskQuality entry —
// i.e. was never judged in this row — so a task absent from the quality
// breakdown reads as "no data" here too rather than a stray 0.
func csvTaskCost(quality map[string]int, costs map[string]float64, task string) string {
	if _, ok := quality[task]; !ok {
		return ""
	}

	return csvFloat(costs[task])
}

// statsCSVCells renders one stats block as its six CSV cells (n, min, mean,
// p50, p95, max), blank across the board for s.N == 0 rather than "0" —
// shared by csvRow's latency_* and ttft_* blocks, which are the same shape.
func statsCSVCells(s stats) []string {
	if s.N == 0 {
		return []string{"", "", "", "", "", ""}
	}

	return []string{
		strconv.Itoa(s.N),
		csvFloat(s.Min), csvFloat(s.Mean), csvFloat(s.P50), csvFloat(s.P95), csvFloat(s.Max),
	}
}

func csvInt(n int) string {
	if n == 0 {
		return ""
	}

	return strconv.Itoa(n)
}

func csvFloat(v float64) string {
	if v == 0 {
		return ""
	}

	return strconv.FormatFloat(v, 'f', -1, 64)
}

// RunSummary reads every
// report-*.json under resultsDir, summarizes it by judge and combined (see
// the package doc comment above), prints the resulting table to stdout, and
// writes it out as JSON+CSV into outputDir (resultsDir when outputDir is
// empty) — "summary.json"/"summary.csv" when isCanonical, otherwise a fresh
// "summary-<RFC3339>.json"/".csv" pair. ttftResultsDir is optional; when
// non-empty it merges a TTFT latency breakdown into every row — see
// attachTTFT. When isCanonical, it also derives use_cases.json alongside them
// (see use_cases.go) and renders that into the standalone use_cases.html
// report (see use_cases_html.go) — isCanonical is always the canonical version dir
// (the only caller passing true is refreshCanonicalSummary in canonical.go, with
// outputDir==resultsDir==versionDir), so this is exactly "runs with every
// benchmark" without also cluttering a one-off staged-review dir.
func RunSummary(resultsDir, outputDir, ttftResultsDir string, isCanonical bool) error {
	if strings.TrimSpace(resultsDir) == "" {
		return errors.New("results dir is required")
	}

	names, samples, err := loadReports(resultsDir)
	if err != nil {
		return err
	}

	pooled := make([]sourceSample, len(samples))
	for i, s := range samples {
		pooled[i] = s.sourceSample
	}

	out := summaryReport{
		GeneratedAt: time.Now().UTC(),
		Sources:     names,
		ByJudge:     summarizeByJudge(samples),
		Combined:    summarize(pooled),
	}

	if strings.TrimSpace(ttftResultsDir) != "" {
		ttftNames, ttftSamples, err := loadTTFTReports(ttftResultsDir)
		if err != nil {
			return fmt.Errorf("ttft results dir: %w", err)
		}

		out.TTFTSources = ttftNames
		byModelBucket := ttftLatencyByModelBucket(ttftSamples)

		for i := range out.ByJudge {
			attachTTFT(out.ByJudge[i].Summary, byModelBucket)
		}

		attachTTFT(out.Combined, byModelBucket)
	}

	printTable(os.Stdout, out)

	encoded, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}

	dir := outputDir
	if dir == "" {
		dir = resultsDir
	}

	var stem string

	if isCanonical {
		stem = filepath.Join(dir, "summary")
	} else {
		stem = filepath.Join(dir, "summary-"+time.Now().Format(time.RFC3339))
	}

	if err := os.WriteFile(stem+".json", encoded, 0o644); err != nil {
		return err
	}

	fmt.Println("wrote", stem+".json")

	if err := writeCSV(stem+".csv", out); err != nil {
		return err
	}

	fmt.Println("wrote", stem+".csv")

	if isCanonical {
		if err := writeUseCases(dir, out.Combined); err != nil {
			return err
		}

		if err := writeUseCasesHTML(dir); err != nil {
			return err
		}
	}

	return nil
}
