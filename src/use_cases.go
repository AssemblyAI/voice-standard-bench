// Post-summary use-case derivation (see RunSummary in summary.go): every
// time RunSummary refreshes the canonical result set, it also mechanically
// derives a short list of named use cases -- "Budget Summarization",
// "Real-Time Voice Agent", "Near-Real-Time Voice Agent", etc. -- from that
// same run's Combined rows, one set per audio-length bucket, and writes them
// to use_cases.json alongside summary.json/summary.csv in the canonical
// version dir (see writeUseCases).
//
// Every use case is one of two shapes, both built by topModels below:
//   - per-task: ranks a single task's TaskQuality/TaskCostPerQuality
//     (dynamically discovered per bucket -- see tasksIn -- the same
//     "discovered, not hardcoded" convention taskColumns uses).
//   - global: ranks a row's own combined QualityScore/CostPerQuality/Latency
//     across every task at once.
//
// Both gate eligibility on a minimum quality bar before cost/latency/
// throughput ever get to rank anyone -- so "cheapest" or "fastest" never
// surfaces a model that's merely cheap or fast because it's bad. A per-task
// archetype also requires overallQualityAcceptable(e): good task_quality
// alone isn't enough if the model's combined quality_score is weak.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	// qualityFloorPremium/qualityFloorAcceptable are the two quality bars use
	// cases gate on -- Premium for "best regardless of cost", Acceptable for
	// "good enough that cost/latency become the deciding factor". Both are
	// fixed points on quality_score's stable 1-100 scale, not per-bucket
	// percentiles, so a use case's meaning doesn't drift as the model roster
	// changes.
	qualityFloorPremium    = 85.0
	qualityFloorMedium     = 85.0
	qualityFloorAcceptable = 70.0
	// qualityFloorNonRealtime is the overall quality_score floor for every
	// archetype that isn't realtime/near-realtime (see
	// overallQualityAcceptableNonRealtime) -- higher than
	// qualityFloorAcceptable since these archetypes aren't already trading
	// quality for latency.
	qualityFloorNonRealtime = 80.0
	// realtimeMaxBucketS/batchMinBucketS gate which buckets a latency- or
	// throughput-flavored use case even makes sense for. realtimeMaxBucketS
	// mirrors ttft.go's own 30s/120s "real-time-adjacent" buckets (see
	// README.md's Time-to-first-token section).
	realtimeMaxBucketS = 120
	batchMinBucketS    = 900
	// nearRealtimeLatencyMultiplier bounds the "near-real-time" archetypes'
	// latency budget relative to the fastest acceptable-quality model in the
	// same bucket (see fastestLatencyP50) -- a step up in latency in exchange
	// for ranking by quality, rather than speed, within that budget. 2x
	// rather than something tighter like 1.5x because real benchmark data
	// clusters into latency tiers (a handful of sub-second models, then a
	// jump to the next tier): under 1.5x the budget usually falls short of
	// that next tier entirely, so the archetype degenerates into a singleton
	// re-ranking realtimeTaskUseCase/realtimeVoiceAgentUseCase's own top
	// pick rather than surfacing a real trade-off. Still "a small step up"
	// in absolute terms for these buckets -- e.g. 0.3s -> 0.6s -- since only
	// realtimeMaxBucketS buckets ever reach this archetype at all.
	nearRealtimeLatencyMultiplier = 2.0
	// maxModelsPerUseCase caps how many models back any one use case.
	maxModelsPerUseCase = 5
)

type useCaseModel struct {
	Model string `json:"model"`
	// ReasoningEffort mirrors summaryEntry.ReasoningEffort (summary.go) --
	// "default" when the model was benchmarked without -reasoning-effort or
	// didn't support it, never blank.
	ReasoningEffort string  `json:"reasoning_effort"`
	Rank            int     `json:"rank"`
	QualityScore    int     `json:"quality_score,omitempty"`
	CostPerQuality  float64 `json:"cost_per_quality,omitempty"`
	LatencyP50S     float64 `json:"latency_p50_s,omitempty"`
	// TaskQualityScore is set only for a per-task use case -- see
	// toUseCaseModels.
	TaskQualityScore int `json:"task_quality_score,omitempty"`
}

type useCase struct {
	Bucket      string         `json:"bucket"`
	Title       string         `json:"title"`
	Description string         `json:"description"`
	Tags        []string       `json:"tags"`
	Models      []useCaseModel `json:"models"`
}

type useCasesReport struct {
	GeneratedAt time.Time `json:"generated_at"`
	UseCases    []useCase `json:"use_cases"`
}

// isNemotron reports whether model is a Nemotron variant. Nemotron has shown to have huge latency
// variation across a number of providers. A simple hello world ping can take 500ms or 20000ms, and is thus
// unreliable to rank against.
func isNemotron(model string) bool {
	return strings.Contains(strings.ToLower(model), "nemotron")
}

// topModels returns up to maxModelsPerUseCase rows passing gate, best first
// by less -- or nil if none pass. Shared by every archetype function below so
// each one only has to state its own gate/ranking, not the cap/sort logic.
func topModels(rows []summaryEntry, gate func(summaryEntry) bool, less func(a, b summaryEntry) bool) []summaryEntry {
	var eligible []summaryEntry

	for _, e := range rows {
		if !isNemotron(e.Model) && gate(e) {
			eligible = append(eligible, e)
		}
	}

	sort.Slice(eligible, func(i, j int) bool { return less(eligible[i], eligible[j]) })

	if len(eligible) > maxModelsPerUseCase {
		eligible = eligible[:maxModelsPerUseCase]
	}

	return eligible
}

// fastestLatencyP50 returns the lowest Latency.P50 among rows passing gate --
// the anchor the nearRealtime* archetypes build their own, slightly looser,
// latency budget from (see nearRealtimeTaskUseCase/
// nearRealtimeVoiceAgentUseCase). ok is false if no row in rows both passes
// gate and carries latency data.
func fastestLatencyP50(rows []summaryEntry, gate func(summaryEntry) bool) (fastest float64, ok bool) {
	for _, e := range rows {
		if isNemotron(e.Model) || !gate(e) || e.Latency.N == 0 {
			continue
		}

		if !ok || e.Latency.P50 < fastest {
			fastest = e.Latency.P50
			ok = true
		}
	}

	return fastest, ok
}

// overallQualityAcceptable reports whether e's combined quality_score clears
// qualityFloorAcceptable, independent of any one task's own score. Every
// per-task archetype below requires this too, so a model that's spiky on
// one task but weak everywhere else is never recommended for it.
func overallQualityAcceptable(e summaryEntry) bool {
	return e.QualityScore > 0 && float64(e.QualityScore) >= qualityFloorAcceptable
}

// overallQualityAcceptableNonRealtime is overallQualityAcceptable's higher
// bar (qualityFloorNonRealtime) for archetypes that aren't realtime/
// near-realtime.
func overallQualityAcceptableNonRealtime(e summaryEntry) bool {
	return e.QualityScore > 0 && float64(e.QualityScore) >= qualityFloorNonRealtime
}

// toUseCaseModels converts already-ranked rows into their rank-1..N output
// shape. task is "" for a global use case (TaskQualityScore left at zero/
// omitted); otherwise each model's TaskQualityScore is filled from its own
// TaskQuality[task].
func toUseCaseModels(rows []summaryEntry, task string) []useCaseModel {
	out := make([]useCaseModel, len(rows))

	for i, e := range rows {
		m := useCaseModel{
			Model:           e.Model,
			ReasoningEffort: e.ReasoningEffort,
			Rank:            i + 1,
			QualityScore:    e.QualityScore,
			CostPerQuality:  e.CostPerQuality,
			LatencyP50S:     e.Latency.P50,
		}

		if task != "" {
			m.TaskQualityScore = e.TaskQuality[task]
		}

		out[i] = m
	}

	return out
}

// taskTitle renders a snake_case task name ("information_extraction") as a
// short title ("Information Extraction").
func taskTitle(task string) string {
	words := strings.Split(task, "_")
	for i, w := range words {
		if w != "" {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}

	return strings.Join(words, " ")
}

// tasksIn returns every task name with at least one judged sample among
// rows, sorted -- rows is already scoped to one bucket, so this is "which
// tasks does this bucket actually have data for" (dictation, for example,
// never covers the 900s/3600s buckets -- see dictation.go).
func tasksIn(rows []summaryEntry) []string {
	seen := map[string]bool{}

	for _, e := range rows {
		for task := range e.TaskQuality {
			seen[task] = true
		}
	}

	tasks := make([]string, 0, len(seen))
	for task := range seen {
		tasks = append(tasks, task)
	}

	sort.Strings(tasks)

	return tasks
}

// taskSpecialistUseCase: highest task_quality for task, regardless of cost,
// among models that also clear overallQualityAcceptableNonRealtime -- a
// model spiky on this one task but weak overall doesn't get to headline it.
func taskSpecialistUseCase(bucket, task string, rows []summaryEntry) (useCase, bool) {
	gate := func(e summaryEntry) bool {
		q, ok := e.TaskQuality[task]
		return ok && float64(q) >= qualityFloorPremium && overallQualityAcceptableNonRealtime(e)
	}
	less := func(a, b summaryEntry) bool {
		if a.TaskQuality[task] != b.TaskQuality[task] {
			return a.TaskQuality[task] > b.TaskQuality[task]
		}

		return a.TaskCostPerQuality[task] < b.TaskCostPerQuality[task]
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	title := taskTitle(task)

	return useCase{
		Bucket:      bucket,
		Title:       title + " Specialist",
		Description: fmt.Sprintf("Highest-accuracy models for %s on %s audio, regardless of cost — pick these when correctness matters more than price.", title, bucket),
		Tags:        []string{task, "premium", "high-accuracy"},
		Models:      toUseCaseModels(top, task),
	}, true
}

// budgetTaskUseCase: cheapest cost_per_quality for task among models that
// clear qualityFloorNonRealtime on both task_quality and
// overallQualityAcceptableNonRealtime.
func budgetTaskUseCase(bucket, task string, rows []summaryEntry) (useCase, bool) {
	gate := func(e summaryEntry) bool {
		q, ok := e.TaskQuality[task]
		return ok && float64(q) >= qualityFloorNonRealtime && overallQualityAcceptableNonRealtime(e)
	}
	less := func(a, b summaryEntry) bool {
		if a.TaskCostPerQuality[task] != b.TaskCostPerQuality[task] {
			return a.TaskCostPerQuality[task] < b.TaskCostPerQuality[task]
		}

		return a.TaskQuality[task] > b.TaskQuality[task]
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	title := taskTitle(task)

	return useCase{
		Bucket:      bucket,
		Title:       "Budget " + title,
		Description: fmt.Sprintf("Cheapest %s options on %s audio per quality point earned, at solid-not-flashy accuracy — pick these for high-volume runs.", title, bucket),
		Tags:        []string{task, "cost-efficient", "high-volume"},
		Models:      toUseCaseModels(top, task),
	}, true
}

// realtimeTaskUseCase: lowest overall p50 latency among models that still
// clear an acceptable task_quality floor and overallQualityAcceptable.
// Latency is the row's combined figure -- summaryEntry carries no per-task
// latency -- so this pairs "fast overall" with "good at task", not a
// task-isolated latency measurement. Only called for buckets at or under
// realtimeMaxBucketS.
func realtimeTaskUseCase(bucket, task string, rows []summaryEntry) (useCase, bool) {
	gate := func(e summaryEntry) bool {
		q, ok := e.TaskQuality[task]
		return ok && float64(q) >= qualityFloorAcceptable && e.Latency.N > 0 && overallQualityAcceptable(e)
	}
	less := func(a, b summaryEntry) bool {
		if a.Latency.P50 != b.Latency.P50 {
			return a.Latency.P50 < b.Latency.P50
		}

		return a.TaskQuality[task] > b.TaskQuality[task]
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	title := taskTitle(task)

	return useCase{
		Bucket:      bucket,
		Title:       "Real-Time " + title,
		Description: fmt.Sprintf("Fastest p50 response time on %s audio while still handling %s well — pick these for live, interactive workflows.", bucket, title),
		Tags:        []string{task, "real-time", "low-latency"},
		Models:      toUseCaseModels(top, task),
	}, true
}

// nearRealtimeTaskUseCase: like realtimeTaskUseCase, but trades a small step
// up in latency for quality -- eligible rows are capped at
// nearRealtimeLatencyMultiplier times the fastest acceptable-quality model's
// own p50 (rather than ranked by latency outright), then ranked by
// task_quality descending within that budget, so the pick is the best a
// nearly-as-fast model can do rather than the fastest model, full stop.
// acceptable also requires overallQualityAcceptable, so a model that's only
// fast because it's weak overall never anchors (or wins) the budget. Only
// called for buckets at or under realtimeMaxBucketS.
func nearRealtimeTaskUseCase(bucket, task string, rows []summaryEntry) (useCase, bool) {
	acceptable := func(e summaryEntry) bool {
		q, ok := e.TaskQuality[task]
		return ok && float64(q) >= qualityFloorMedium && overallQualityAcceptable(e)
	}

	fastest, ok := fastestLatencyP50(rows, acceptable)
	if !ok {
		return useCase{}, false
	}

	budget := fastest * nearRealtimeLatencyMultiplier

	gate := func(e summaryEntry) bool { return acceptable(e) && e.Latency.N > 0 && e.Latency.P50 <= budget }
	less := func(a, b summaryEntry) bool {
		if a.TaskQuality[task] != b.TaskQuality[task] {
			return a.TaskQuality[task] > b.TaskQuality[task]
		}

		return a.Latency.P50 < b.Latency.P50
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	title := taskTitle(task)

	return useCase{
		Bucket:      bucket,
		Title:       "Near-Real-Time " + title,
		Description: fmt.Sprintf("Best %s quality on %s audio within a small step up from the fastest p50 response time — trades a little latency for noticeably better results in interactive workflows.", title, bucket),
		Tags:        []string{task, "near-real-time", "balanced"},
		Models:      toUseCaseModels(top, task),
	}, true
}

// topOverallUseCase: highest combined quality_score, regardless of cost.
func topOverallUseCase(bucket string, rows []summaryEntry) (useCase, bool) {
	gate := func(e summaryEntry) bool { return e.QualityScore > 0 && float64(e.QualityScore) >= qualityFloorPremium }
	less := func(a, b summaryEntry) bool {
		if a.QualityScore != b.QualityScore {
			return a.QualityScore > b.QualityScore
		}

		return a.CostPerQuality < b.CostPerQuality
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	return useCase{
		Bucket:      bucket,
		Title:       "Top Overall Performer",
		Description: fmt.Sprintf("Best combined quality_score across every task on %s audio — the safe default when the task mix isn't known ahead of time.", bucket),
		Tags:        []string{"overall", "premium", "high-accuracy"},
		Models:      toUseCaseModels(top, ""),
	}, true
}

// bestValueUseCase: lowest combined cost_per_quality among models that still
// clear an acceptable quality floor.
func bestValueUseCase(bucket string, rows []summaryEntry) (useCase, bool) {
	gate := overallQualityAcceptableNonRealtime
	less := func(a, b summaryEntry) bool {
		if a.CostPerQuality != b.CostPerQuality {
			return a.CostPerQuality < b.CostPerQuality
		}

		return a.QualityScore > b.QualityScore
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	return useCase{
		Bucket:      bucket,
		Title:       "Best Value",
		Description: fmt.Sprintf("Lowest cost_per_quality on %s audio among models that still clear a decent quality bar — the cheapest defensible choice.", bucket),
		Tags:        []string{"overall", "cost-efficient", "value"},
		Models:      toUseCaseModels(top, ""),
	}, true
}

// realtimeVoiceAgentUseCase: lowest overall p50 latency among models that
// still clear an acceptable combined quality floor. Only called for buckets
// at or under realtimeMaxBucketS.
func realtimeVoiceAgentUseCase(bucket string, rows []summaryEntry) (useCase, bool) {
	gate := func(e summaryEntry) bool { return overallQualityAcceptable(e) && e.Latency.N > 0 }
	less := func(a, b summaryEntry) bool {
		if a.Latency.P50 != b.Latency.P50 {
			return a.Latency.P50 < b.Latency.P50
		}

		return a.QualityScore > b.QualityScore
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	return useCase{
		Bucket:      bucket,
		Title:       "Real-Time Voice Agent",
		Description: fmt.Sprintf("Fastest p50 latency on %s audio at acceptable quality across every task — for live voice-agent turns where response time is the binding constraint.", bucket),
		Tags:        []string{"real-time", "voice-agent", "low-latency"},
		Models:      toUseCaseModels(top, ""),
	}, true
}

// nearRealtimeVoiceAgentUseCase: like realtimeVoiceAgentUseCase, but trades a
// small step up in latency for quality -- see nearRealtimeTaskUseCase, whose
// combined-across-tasks counterpart this is. Only called for buckets at or
// under realtimeMaxBucketS.
func nearRealtimeVoiceAgentUseCase(bucket string, rows []summaryEntry) (useCase, bool) {
	acceptable := func(e summaryEntry) bool {
		return e.QualityScore > 0 && float64(e.QualityScore) >= qualityFloorMedium
	}

	fastest, ok := fastestLatencyP50(rows, acceptable)
	if !ok {
		return useCase{}, false
	}

	budget := fastest * nearRealtimeLatencyMultiplier

	gate := func(e summaryEntry) bool { return acceptable(e) && e.Latency.N > 0 && e.Latency.P50 <= budget }
	less := func(a, b summaryEntry) bool {
		if a.QualityScore != b.QualityScore {
			return a.QualityScore > b.QualityScore
		}

		return a.Latency.P50 < b.Latency.P50
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	return useCase{
		Bucket:      bucket,
		Title:       "Near-Real-Time Voice Agent",
		Description: fmt.Sprintf("Best overall quality on %s audio within a small step up from the fastest p50 latency — trades a little response time for noticeably better results in voice-agent turns.", bucket),
		Tags:        []string{"near-real-time", "voice-agent", "balanced"},
		Models:      toUseCaseModels(top, ""),
	}, true
}

// longFormBatchUseCase: highest combined quality_score among models that
// clear an acceptable floor, tie-broken by throughput -- wall-clock latency
// matters less than getting a long, dense transcript right. Only called for
// buckets at or over batchMinBucketS.
func longFormBatchUseCase(bucket string, rows []summaryEntry) (useCase, bool) {
	gate := overallQualityAcceptableNonRealtime
	less := func(a, b summaryEntry) bool {
		if a.QualityScore != b.QualityScore {
			return a.QualityScore > b.QualityScore
		}

		return a.TokensPerSecond > b.TokensPerSecond
	}

	top := topModels(rows, gate, less)
	if len(top) == 0 {
		return useCase{}, false
	}

	return useCase{
		Bucket:      bucket,
		Title:       "Long-Form Call Analysis",
		Description: fmt.Sprintf("Best quality on %s calls, where wall-clock latency matters less than getting long, dense transcripts right — suited to async/batch processing.", bucket),
		Tags:        []string{"batch", "long-form", "async"},
		Models:      toUseCaseModels(top, ""),
	}, true
}

// deriveUseCases groups entries (summaryReport.Combined, the pooled-across-
// judges view -- the "overall" score the rest of this file ranks on) by
// bucket and runs every archetype above within each one. An archetype that
// finds no eligible model in a bucket contributes nothing, so sparse data
// (e.g. a task no model has been judged on yet) produces fewer use cases
// rather than empty or misleading ones.
func deriveUseCases(entries []summaryEntry) []useCase {
	byBucket := map[string][]summaryEntry{}
	for _, e := range entries {
		byBucket[e.Bucket] = append(byBucket[e.Bucket], e)
	}

	buckets := make([]string, 0, len(byBucket))
	for b := range byBucket {
		buckets = append(buckets, b)
	}

	sort.Slice(buckets, func(i, j int) bool {
		si, oki := bucketSeconds(buckets[i])
		sj, okj := bucketSeconds(buckets[j])

		if oki && okj {
			return si < sj
		}

		return buckets[i] < buckets[j]
	})

	var out []useCase

	for _, bucket := range buckets {
		rows := byBucket[bucket]
		seconds, hasSeconds := bucketSeconds(bucket)
		realtime := hasSeconds && seconds <= realtimeMaxBucketS
		batch := hasSeconds && seconds >= batchMinBucketS

		for _, task := range tasksIn(rows) {
			if uc, ok := taskSpecialistUseCase(bucket, task, rows); ok {
				out = append(out, uc)
			}

			if uc, ok := budgetTaskUseCase(bucket, task, rows); ok {
				out = append(out, uc)
			}

			if realtime {
				if uc, ok := realtimeTaskUseCase(bucket, task, rows); ok {
					out = append(out, uc)
				}

				if uc, ok := nearRealtimeTaskUseCase(bucket, task, rows); ok {
					out = append(out, uc)
				}
			}
		}

		if uc, ok := topOverallUseCase(bucket, rows); ok {
			out = append(out, uc)
		}

		if uc, ok := bestValueUseCase(bucket, rows); ok {
			out = append(out, uc)
		}

		if realtime {
			if uc, ok := realtimeVoiceAgentUseCase(bucket, rows); ok {
				out = append(out, uc)
			}

			if uc, ok := nearRealtimeVoiceAgentUseCase(bucket, rows); ok {
				out = append(out, uc)
			}
		}

		if batch {
			if uc, ok := longFormBatchUseCase(bucket, rows); ok {
				out = append(out, uc)
			}
		}
	}

	return out
}

// writeUseCases runs deriveUseCases over entries and writes use_cases.json
// into dir, mirroring RunSummary's own MarshalIndent-then-WriteFile shape
// for summary.json -- plus a trailing newline MarshalIndent doesn't add,
// so the file satisfies pre-commit's end-of-file-fixer on the first write
// rather than needing a second pass to fix it up.
func writeUseCases(dir string, entries []summaryEntry) error {
	out := useCasesReport{
		GeneratedAt: time.Now().UTC(),
		UseCases:    deriveUseCases(entries),
	}

	encoded, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}

	path := filepath.Join(dir, "use_cases.json")
	if err := os.WriteFile(path, append(encoded, '\n'), 0o644); err != nil {
		return err
	}

	fmt.Println("wrote", path)

	return nil
}
