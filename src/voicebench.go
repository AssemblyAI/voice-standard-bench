// Command benchmark measures llm_gateway models against real AssemblyAI
// call-transcript data through the Task[R]/Runner framework (framework.go):
// for every -tasks entry with a framework port (see frameworkTaskConstructors
// in framework_pass.go), it runs the standard judged pass and a separate
// streaming time-to-first-token pass, then merges both into the
// -canonical-version result set.
//
// -replay re-runs only the judge step over an earlier run's stashed candidate
// outputs, without calling the candidate models again -- see replay.go.
//
// -refresh-canonical-summary skips all of that and just recomputes
// -canonical-version's summary.json/summary.csv/use_cases.json from its
// already-staged files -- see RunRefreshCanonicalSummary in canonical.go.
//
// See README.md for methodology and prompts/README.md for the task and judge
// prompts themselves. Run with -h for flag documentation.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

func main() {
	startTime := time.Now()
	gatewayURL := flag.String("gateway-url", "https://llm-gateway.assemblyai.com", "llm_gateway base URL")
	tasksFlag := flag.String("tasks", "information_extraction,text_correction,summarization", "comma-separated tasks to run")
	modelsFlag := flag.String("models", "", "comma-separated model IDs to benchmark (default: every model returned by GET /v1/models, minus -judge-model)")
	bucketsFlag := flag.String("buckets", "", "comma-separated audio-length buckets to benchmark, e.g. 30s,120s (default: every bucket present in the loaded dataset); the canonical merge only updates the bucket(s) actually run, leaving every other bucket's already-merged data untouched")
	judgeModel := flag.String("judge-model", "claude-opus-5-5,gpt-6.1-sol", "fixed model used to grade quality, or several comma-separated to run every judge over each model output (each task call is made once and judged by all of them, one report-<judge>.json per judge); judges are excluded from the default candidate set to avoid self-grading")
	maxSamples := flag.Int("max-samples-per-bucket", 10, "max transcripts to benchmark per audio-length bucket per task, sampled deterministically by record id (<=0 means every loaded record)")
	concurrency := flag.Int("concurrency", 5, "max concurrent in-flight requests PER MODEL — see the rate-limit note on NewGateway in gateway.go before raising this")
	timeout := flag.Duration("timeout", 500*time.Second, "per-request timeout, for both task and judge completions")
	reasoningEffort := flag.String("reasoning-effort", "", "reasoning_effort to request from every candidate model (never the judge model) whose GET /v1/models entry advertises support for it via supported_parameters; empty means don't set it at all, and a model that doesn't advertise support logs a warning and is left unset rather than sent the parameter anyway")
	canonicalVersion := flag.String("canonical-version", DefaultCanonicalVersion, "benchmark schema version to append/update in the canonical result set (folder name uses dashes, e.g. 0.1.2 -> 0-1-2)")
	approveMerge := flag.Bool("approve-merge", false, "skip the interactive merge confirmation and merge the staged results into canonical immediately -- for scripted/CI use once a run is already trusted")
	replay := flag.String("replay", "", "name of an earlier run (its directory under results/, as logged at the end of that run -- a path to it works too) to replay: every model call that run already made and stashed (both the standard and the TTFT pass, under task_runs/) is reused instead of called again, and only the judging runs live; any (model, record) it didn't stash runs live as usual. Refused before anything is called if the run stashed nothing, if none of the requested models (at the requested -reasoning-effort) and tasks was stashed, or if a requested task's prompt has changed since")
	skipJudge := flag.Bool("skip-judge", false, "gather and stash the candidates' outputs (both passes) but skip all judging -- the LLM judge and dictation's mechanical scoring alike -- and skip the canonical merge, which unjudged samples must not overwrite; judge the run later with -replay")
	refreshSummary := flag.Bool("refresh-canonical-summary", false, "recompute -canonical-version's summary.json/summary.csv/use_cases.json from its already-staged report-*.json/ttft-report*.json and exit -- no benchmark run, gateway, or ASSEMBLY_AI_TOKEN needed")
	flag.Parse()

	if *refreshSummary {
		canonicalRoot := frameworkCanonicalRoot
		if err := RunRefreshCanonicalSummary(canonicalRoot, *canonicalVersion); err != nil {
			fatal(err)
		}

		log.Printf("refreshed canonical summary for version %s", *canonicalVersion)

		return
	}

	runFrameworkBenchmarkPass(context.Background(), frameworkPassConfig{
		GatewayURL:          *gatewayURL,
		Timeout:             *timeout,
		ModelsFlag:          *modelsFlag,
		TasksFlag:           *tasksFlag,
		BucketsFlag:         *bucketsFlag,
		JudgeModels:         parseCSV(*judgeModel),
		ReasoningEffort:     *reasoningEffort,
		MaxSamplesPerBucket: *maxSamples,
		Concurrency:         *concurrency,
		CanonicalVersion:    *canonicalVersion,
		ApproveMerge:        *approveMerge,
		SkipJudge:           *skipJudge,
		Replay:              *replay,
	})

	log.Printf("finished framework run, total time: %f\n", time.Since(startTime).Seconds())
}

// taskInput returns the text sent to the model as its user message. Every
// task but text_correction operates on the transcript as recorded
// (record.Text). text_correction instead corrects a SIMULATED corrupted
// transcript (record.MissedText) derived from the original (record.Text), so
// it is the corrupted version that must be sent as input; sending record.Text
// would hand the model the uncorrupted original.
//
// A handful of text_correction records have no MissedText (the simulation
// didn't cover them — see TranscriptRecord's doc comment); those are reported
// as a failed sample rather than silently falling back to record.Text, which
// would score as trivially perfect for having done nothing.
func taskInput(task string, record TranscriptRecord) (string, error) {
	if task != prompts.TextCorrection {
		return record.Text, nil
	}

	if record.MissedText == "" {
		return "", errors.New("record has no missed_text to correct")
	}

	return record.MissedText, nil
}

// recordFailure writes one FailureRecord to failures, logging rather than
// failing the sample if the write itself errors — a benchmark run that's
// otherwise succeeding shouldn't abort over a disk-full or permissions
// problem writing its OWN diagnostic output. Takes bucket/recordID directly
// rather than a whole TranscriptRecord — the only two fields this function
// actually reads — so callers with a different record shape (dictation.go's
// DictationRecord, which has no TranscriptRecord to hand it) can use it too.
func recordFailure(failures *FailureRecorder, model, reasoningEffort, task, bucket, recordID, reason string, cause error, request, response string) {
	err := failures.Record(FailureRecord{
		Model:           model,
		ReasoningEffort: reasoningEffort,
		Task:            task,
		Bucket:          bucket,
		RecordID:        recordID,
		Reason:          reason,
		Err:             cause,
		Request:         request,
		Response:        response,
	})

	if err != nil {
		log.Printf("model %s task %s record %s: recording failure: %v", model, task, recordID, err)
	}
}

func worstRubric(rubric []string) map[string]float64 {
	scores := make(map[string]float64, len(rubric))
	for _, dim := range rubric {
		scores[dim] = 1
	}

	return scores
}

// estimatedTokens approximates token count from character count using the
// common ~4-characters-per-token heuristic for English text. It exists only
// to pre-flight-guard against sending a doomed request to a model whose
// context window can't hold the transcript — not to size billing or
// max_tokens precisely — so the crude approximation is an acceptable
// trade-off against pulling in a real tokenizer dependency for a guard rail.
func estimatedTokens(s string) int {
	return len(s)/4 + 1
}

// contextLengthExceeded reports whether transcript plus systemPrompt would
// likely exceed model's advertised context window. model.ContextLength <= 0
// (not reported by GET /v1/models for every model) skips the check rather
// than blocking every request against that model — absence of the field
// isn't evidence the model has no room.
func contextLengthExceeded(model ModelInfo, systemPrompt, transcript string) bool {
	if model.ContextLength <= 0 {
		return false
	}

	return estimatedTokens(systemPrompt)+estimatedTokens(transcript) > model.ContextLength
}

// taskMaxTokens sizes the completion budget per task. Correction's and
// dictation's output is roughly the same length as its input — cleanup or
// reformatting, not compression — so their budget scales with the input;
// summarization and extraction produce short, roughly fixed-size output
// regardless of transcript length. Every budget is clamped to the model's own
// advertised max completion tokens when GET /v1/models reported one.
func taskMaxTokens(task string, transcript string, model ModelInfo) int {
	var want int

	switch task {
	case prompts.TextCorrection, prompts.Dictation:
		want = estimatedTokens(transcript) + estimatedTokens(transcript)/5 + 64 + 15000 // input-sized, plus 20% headroom, plus thinking tokens
	case prompts.InformationExtraction:
		want = 15000
	default: // Summarization
		want = 15000
	}

	if model.MaxCompletionTokens > 0 && want > model.MaxCompletionTokens {
		want = model.MaxCompletionTokens
	}

	return want
}

func parseTasks(csv string) ([]string, error) {
	tasks := parseCSV(csv)

	if len(tasks) == 0 {
		return nil, errors.New("-tasks must name at least one task")
	}

	for _, task := range tasks {
		if _, ok := prompts.For(task); !ok {
			return nil, fmt.Errorf("unknown task %q (valid: %s)", task, strings.Join(prompts.All, ", "))
		}
	}

	return tasks, nil
}

// filterModels returns the models in all whose ID appears in the
// comma-separated allowlist. explicit reports whether allowlist was
// non-empty at all — an empty allowlist returns every model in all, and
// callers use explicit to tell "user asked for everything" apart from "user
// didn't ask for anything in particular" (see the judge-exclusion logic in
// runFrameworkBenchmarkPass).
func filterModels(all []ModelInfo, allowlist string) (filtered []ModelInfo, explicit bool) {
	ids := parseCSV(allowlist)

	if len(ids) == 0 {
		return all, false
	}

	wanted := make(map[string]bool, len(ids))
	for _, id := range ids {
		wanted[id] = true
	}

	for _, m := range all {
		if wanted[m.ID] {
			filtered = append(filtered, m)
		}
	}

	return filtered, true
}

func removeModel(models []ModelInfo, id string) []ModelInfo {
	filtered := make([]ModelInfo, 0, len(models))

	for _, m := range models {
		if m.ID != id {
			filtered = append(filtered, m)
		}
	}

	return filtered
}

func containsModel(models []ModelInfo, id string) bool {
	for _, m := range models {
		if m.ID == id {
			return true
		}
	}

	return false
}

// unstableModelSubstrings names model families excluded from the default
// (no -models flag) candidate set because they have shown very inconsistent
// stability across runs -- e.g. wildly variable latency or a high error
// rate -- which skews benchmark results rather than reflecting genuine model
// performance. Passing -models explicitly still allows benchmarking them.
var unstableModelSubstrings = []string{"gpt-oss", "nemotron"}

// isUnstableModel reports whether id belongs to one of the model families in
// unstableModelSubstrings.
func isUnstableModel(id string) bool {
	lower := strings.ToLower(id)

	for _, substr := range unstableModelSubstrings {
		if strings.Contains(lower, substr) {
			return true
		}
	}

	return false
}

// removeUnstableModels drops every model in models matching
// unstableModelSubstrings, returning the filtered slice and the IDs removed.
func removeUnstableModels(models []ModelInfo) (filtered []ModelInfo, removed []string) {
	for _, m := range models {
		if isUnstableModel(m.ID) {
			removed = append(removed, m.ID)
			continue
		}

		filtered = append(filtered, m)
	}

	return filtered, removed
}

// applyReasoningEffort sets ReasoningEffort on every model in models that
// advertises support for it (ModelInfo.SupportsParameter(reasoningEffortParam),
// backed by GET /v1/models' supported_parameters), leaving every other
// model untouched and logging a warning naming it instead — never silently
// sending a provider a parameter it never claimed to support. Called once,
// against the already-resolved candidate list (see runFrameworkBenchmarkPass),
// not per request, so an unsupported model is warned about once per run
// rather than once per sample.
func applyReasoningEffort(models []ModelInfo, effort string) []ModelInfo {
	out := make([]ModelInfo, len(models))

	for i, m := range models {
		if m.SupportsParameter(reasoningEffortParam) {
			value := effort
			m.ReasoningEffort = &value
		} else {
			log.Printf("framework: model %s does not advertise support for %s (supported params: %v) — ignoring -reasoning-effort for this model", m.ID, reasoningEffortParam, m.SupportedParameters)
		}

		out[i] = m
	}

	return out
}

func parseCSV(s string) []string {
	var out []string

	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}

	return out
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
