// The benchmark run (see voicebench.go): runs the tasks named in -tasks
// that have a framework port, through the Task[R]/Runner framework.
// Resolves its own gateway/model list.
//
// Every invocation always runs the standard judged pass AND the TTFT pass,
// stages both, then reviews and merges them into -canonical-version via
// ReviewAndMergeIntoCanonical (see canonical.go). There is no flag to opt out of
// the merge step.
//
// Both passes stash every successful model call's output as it comes back (see
// TaskRunStore), in a staging directory named for the run -- the same name under
// results/ and results_ttft/ -- and -replay takes that name to re-run only the
// judging over an earlier run's outputs (see replay.go).
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

// frameworkPassConfig bundles the CLI flags a benchmark run takes.
type frameworkPassConfig struct {
	GatewayURL string
	APIKey     string
	Timeout    time.Duration
	ModelsFlag string
	TasksFlag  string
	// BucketsFlag, when non-empty, restricts every requested task to just
	// these comma-separated audio-length buckets -- parsed once in
	// runFrameworkBenchmarkPass (see its buckets local) and passed to both
	// the standard and TTFT passes as RunConfig.Buckets. Empty means every
	// bucket the loaded dataset has, mirroring ModelsFlag's own "empty means
	// everything" convention (see filterModels in voicebench.go).
	BucketsFlag string
	JudgeModel  string
	// ReasoningEffort, when non-empty, is applied to every candidate model
	// that advertises support for it -- see applyReasoningEffort
	// (voicebench.go). Never applied to JudgeModel, which is passed to
	// RunJudge by bare ID, not as a ModelInfo (see judge.go's RunJudge).
	ReasoningEffort string

	MaxSamplesPerBucket int
	Concurrency         int

	// Canonical-merge target, mirroring -canonical-version/-approve-merge. Every
	// path a run stages or records into is otherwise fixed (see the
	// framework* constants below) -- no flag for those.
	CanonicalVersion string
	ApproveMerge     bool

	// Debug mirrors -debug: tasks print extra diagnostics as they run (see
	// RunConfig.Debug). It never changes what is measured, recorded, or merged.
	Debug bool

	// SkipJudge mirrors -skip-judge: the standard pass gathers and stashes the
	// task outputs but judges nothing (see RunConfig.SkipJudge), and the run ends
	// before the canonical merge -- unjudged samples merged in would replace
	// canonical's judged rows for the same (model, task, bucket). The TTFT pass
	// still runs: it never judges anyway.
	SkipJudge bool

	// Replay mirrors -replay: the name of an earlier run (see replayRunName for
	// the forms it takes) whose stashed task runs this run reuses instead of
	// calling the candidate models again -- both passes' -- so that only the
	// judge step is paid for. A (model, record) the earlier run didn't stash
	// runs live. Empty replays nothing.
	Replay string
}

// frameworkOutputPath, frameworkErrorsPath, frameworkCanonicalRoot,
// frameworkTTFTOutputPath, and frameworkTTFTErrorsPath are where a run
// always stages/records every pass and merges its canonical result set -- there
// is no flag to override any of them. Every one of these (and normalizedRoot,
// where the datasets are read from) is resolved relative to the working
// directory and lives one level up from it (the repo root, when run from
// src/), so tests can read/write them under the parent of their
// own t.Chdir'd temp root.
const (
	normalizedRoot          = "../normalized"
	frameworkOutputPath     = "../results"
	frameworkErrorsPath     = "../errors"
	frameworkCanonicalRoot  = "../results/canonical"
	frameworkTTFTOutputPath = "../results_ttft"
	frameworkTTFTErrorsPath = "../errors_ttft"
)

// frameworkRunner lets a Runner[R] run without its type parameter --
// RunBenchmark/RunTTFTBenchmark don't mention R, so every Runner[R]
// satisfies this.
type frameworkRunner interface {
	Name() string
	RunBenchmark(ctx context.Context) ([]Sample, []Cell, error)
	RunTTFTBenchmark(ctx context.Context) ([]Sample, []Cell, error)
}

// frameworkTaskConstructors maps each framework-ported task's name to its
// constructor. Register new tasks here.
var frameworkTaskConstructors = map[string]func(*Gateway, RunConfig) (frameworkRunner, error){
	prompts.Dictation: func(gw *Gateway, cfg RunConfig) (frameworkRunner, error) {
		task, err := NewDictationTask(gw)
		if err != nil {
			return nil, err
		}

		return NewRunner[DictationRecord](task, cfg), nil
	},
	prompts.Summarization: func(gw *Gateway, cfg RunConfig) (frameworkRunner, error) {
		task, err := NewSummarizationTask(gw)
		if err != nil {
			return nil, err
		}

		return NewRunner[TranscriptRecord](task, cfg), nil
	},
	prompts.TextCorrection: func(gw *Gateway, cfg RunConfig) (frameworkRunner, error) {
		task, err := NewTextCorrectionTask(gw, cfg.Debug)
		if err != nil {
			return nil, err
		}

		return NewRunner[TranscriptRecord](task, cfg), nil
	},
	prompts.InformationExtraction: func(gw *Gateway, cfg RunConfig) (frameworkRunner, error) {
		task, err := NewInformationExtractionTask(gw)
		if err != nil {
			return nil, err
		}

		return NewRunner[TranscriptRecord](task, cfg), nil
	},
}

// frameworkTasks builds the requested tasks that have a framework port,
// skipping (and logging) any that don't.
func frameworkTasks(gw *Gateway, cfg RunConfig, requested []string) ([]frameworkRunner, error) {
	var runners []frameworkRunner

	for _, name := range requested {
		newTask, ok := frameworkTaskConstructors[name]
		if !ok {
			log.Printf("framework: task %s has no framework port yet, skipping", name)
			continue
		}

		runner, err := newTask(gw, cfg)
		if err != nil {
			return nil, fmt.Errorf("constructing task %s: %w", name, err)
		}

		runners = append(runners, runner)
	}

	if len(runners) == 0 {
		return nil, errors.New("no requested tasks have a framework port")
	}

	return runners, nil
}

// runFrameworkBenchmarkPass implements the benchmark run: resolves the
// gateway/model list once, runs the standard pass then the TTFT pass --
// sequentially, never concurrently, so streaming and synchronous requests
// never compound against the gateway's per-model rate limit -- and merges both staged results into
// canonical, unconditionally (see the package doc comment above).
func runFrameworkBenchmarkPass(ctx context.Context, cfg frameworkPassConfig) {
	requestedTasks, err := parseTasks(cfg.TasksFlag)
	if err != nil {
		fatal(fmt.Errorf("framework: %w", err))
	}

	// Resolved before anything is called -- the gateway included: a replay that
	// can't be done faithfully is refused here, not discovered halfway through
	// (see openReplaySource).
	var replay *replaySource

	if cfg.Replay != "" {
		replay, err = openReplaySource(cfg.Replay, requestedTasks)
		if err != nil {
			fatal(fmt.Errorf("framework: %w", err))
		}
	}

	key := cfg.APIKey
	if key == "" {
		key = os.Getenv("ASSEMBLY_AI_TOKEN")
	}

	if key == "" {
		fatal(errors.New("framework: no API key: set the ASSEMBLY_AI_TOKEN environment variable (GET /v1/models needs none, but running completions does)"))
	}

	gw := NewGateway(cfg.GatewayURL, key, cfg.Timeout)

	log.Printf("framework: fetching available models from %s/v1/models", cfg.GatewayURL)

	allModels, err := gw.ListModels(ctx)
	if err != nil {
		fatal(fmt.Errorf("framework: listing models: %w", err))
	}

	models, explicit := filterModels(allModels, cfg.ModelsFlag)

	if !explicit {
		// mirrors main()'s judge-exclusion logic
		before := len(models)
		models = removeModel(models, cfg.JudgeModel)

		if len(models) < before {
			log.Printf("framework: excluding judge model %s from the default candidate set (pass -models to include it explicitly)", cfg.JudgeModel)
		}

		// gpt-oss and nemotron models are skipped by default due to very
		// inconsistent stability (see unstableModelSubstrings) -- pass
		// -models to include them explicitly.
		var removedUnstable []string
		models, removedUnstable = removeUnstableModels(models)

		if len(removedUnstable) > 0 {
			log.Printf("framework: excluding unstable model(s) %s from the default candidate set (pass -models to include them explicitly)", strings.Join(removedUnstable, ", "))
		}
	} else if containsModel(models, cfg.JudgeModel) {
		log.Printf("framework: warning: judge model %s is also in -models — it will grade its own output on every task it runs", cfg.JudgeModel)
	}

	if cfg.ReasoningEffort != "" {
		models = applyReasoningEffort(models, cfg.ReasoningEffort)
	}

	if len(models) == 0 {
		fatal(errors.New("framework: no models to benchmark — check -models against GET /v1/models"))
	}

	buckets := parseCSV(cfg.BucketsFlag)
	if len(buckets) > 0 {
		log.Printf("framework: restricting to bucket(s): %s", strings.Join(buckets, ", "))
	}

	if replay != nil {
		if err := replay.checkSelection(models, requestedTasks); err != nil {
			fatal(fmt.Errorf("framework: %w", err))
		}
	}

	if cfg.Debug && !slices.Contains(requestedTasks, prompts.TextCorrection) {
		log.Printf("framework: -debug only prints for %s, which isn't in -tasks -- nothing extra will be printed", prompts.TextCorrection)
	}

	reportDir := runFrameworkStandardPass(ctx, gw, models, requestedTasks, buckets, cfg, replay)
	ttftDir := runFrameworkTTFTPass(ctx, gw, models, requestedTasks, buckets, cfg, filepath.Base(reportDir), replay)

	if cfg.SkipJudge {
		log.Printf("framework: -skip-judge: nothing was judged, so nothing is merged into canonical; judge this run later with -replay %s (staged under %s and %s)", filepath.Base(reportDir), reportDir, ttftDir)

		return
	}

	canonicalVersion := cfg.CanonicalVersion
	if strings.TrimSpace(canonicalVersion) == "" {
		canonicalVersion = DefaultCanonicalVersion
	}

	canonicalRoot := frameworkCanonicalRoot
	if err := ReviewAndMergeIntoCanonical(canonicalRoot, canonicalVersion, reportDir, ttftDir, cfg.ApproveMerge); err != nil {
		fatal(fmt.Errorf("framework: %w", err))
	}
}

// runFrameworkStandardPass runs the standard, judged pass for every
// requestedTasks task with a framework port, prints the report, and stages
// it under frameworkOutputPath -- the framework counterpart to
// voicebench.go's runStandardBenchmarkPass. buckets, when non-empty,
// restricts every task to just those audio-length buckets (see
// RunConfig.Buckets) -- so a re-run scoped to one bucket only produces a
// report covering that bucket, which RunAppendCanonical's bucket-aware merge
// (canonical.go) then uses to update just that bucket's rows in canonical.
//
// The staged run directory is created before the first model is called, not
// when the report is written, because the pass writes into it as it goes:
// every successful task run's output is stashed under its task_runs/ (see
// TaskRunStore) the moment it returns. A pass that dies before it has a
// report -- or whose judge calls fail -- therefore still leaves every output
// it had paid for.
//
// replay, when non-nil, is an earlier run whose stashed task runs stand in for
// the RunRecord calls it made (see RunConfig.Replay): those are not called again,
// only judged. The pass still stashes them into its own directory, so every run's
// directory is complete in itself and can be replayed in turn.
func runFrameworkStandardPass(ctx context.Context, gw *Gateway, models []ModelInfo, requestedTasks []string, buckets []string, cfg frameworkPassConfig, replay *replaySource) (reportDir string) {
	failures, err := NewFailureRecorder(frameworkErrorsPath)
	if err != nil {
		fatal(fmt.Errorf("framework: setting up failure recorder: %w", err))
	}

	log.Printf("framework: recording failures under %s", failures.RunDir())

	reportDir, err = NewRunDir(frameworkOutputPath)
	if err != nil {
		fatal(fmt.Errorf("framework: creating run directory under %s: %w", frameworkOutputPath, err))
	}

	taskRuns := NewTaskRunStore(reportDir)

	log.Printf("framework: stashing each task run's output under %s", taskRuns.Dir())

	tasks, err := frameworkTasks(gw, RunConfig{
		Models:              models,
		JudgeModel:          cfg.JudgeModel,
		MaxSamplesPerBucket: cfg.MaxSamplesPerBucket,
		Concurrency:         cfg.Concurrency,
		Buckets:             buckets,
		Failures:            failures,
		TaskRuns:            taskRuns,
		Replay:              replay.standardRuns(),
		SkipJudge:           cfg.SkipJudge,
		Debug:               cfg.Debug,
	}, requestedTasks)
	if err != nil {
		fatal(fmt.Errorf("framework: %w", err))
	}

	log.Printf("framework: benchmarking %d model(s) on %d task(s)", len(models), len(tasks))

	var (
		samples []Sample
		cells   []Cell
	)

	for _, runner := range tasks {
		taskSamples, taskCells, err := runner.RunBenchmark(ctx)
		if err != nil {
			fatal(fmt.Errorf("framework: running task %s: %w", runner.Name(), err))
		}

		samples = append(samples, taskSamples...)
		cells = append(cells, taskCells...)
	}

	PrintReport(os.Stdout, cells)

	log.Printf("framework: recorded %d failure(s) under %s", failures.Count(), failures.RunDir())
	log.Printf("framework: stashed %d task run output(s) under %s", taskRuns.Count(), taskRuns.Dir())
	logReplay("", replay.standardRuns(), samples)

	if err := WriteJSONReport(reportDir, samples, cells, cfg.JudgeModel); err != nil {
		fatal(fmt.Errorf("framework: writing results: %w", err))
	}

	log.Printf("framework: wrote full results to %s", reportDir)

	return reportDir
}

// runFrameworkTTFTPass is runFrameworkStandardPass's streaming counterpart --
// the framework counterpart to voicebench.go's runTTFTBenchmarkPass -- same
// models/tasks, a separate FailureRecorder under frameworkTTFTErrorsPath,
// staged as ttft-report.json under frameworkTTFTOutputPath. The framework's
// Task.RunTTFT already returns a Sample/Cell (unlike the original
// TaskGroup-based pipeline's dedicated TTFTSample/TTFTCell), so results are
// converted via sampleToTTFTSample/cellToTTFTCell before staging -- keeping
// the on-disk shape identical to what RunAppendCanonical/RunSummary already
// merge/summarize separately from a judged report (see ttft.go's package doc
// comment). buckets is threaded through exactly like runFrameworkStandardPass's
// own -- see its doc comment -- and composes with TTFTBuckets inside
// Runner.RunTTFT (framework.go), so a buckets selection outside 30s/120s
// simply produces no TTFT samples at all rather than an error.
//
// Its staging directory is named runName, the name the standard pass gave its
// own, so one name finds everything a run staged (see NewNamedRunDir) -- and is
// made up front, like the standard pass's, because the pass stashes each
// streaming call's output into it as it returns. replay is used as in
// runFrameworkStandardPass, for RunTTFT.
func runFrameworkTTFTPass(ctx context.Context, gw *Gateway, models []ModelInfo, requestedTasks []string, buckets []string, cfg frameworkPassConfig, runName string, replay *replaySource) (ttftDir string) {
	failures, err := NewFailureRecorder(frameworkTTFTErrorsPath)
	if err != nil {
		fatal(fmt.Errorf("framework: setting up ttft failure recorder: %w", err))
	}

	log.Printf("framework: ttft: recording failures under %s", failures.RunDir())

	ttftDir, err = NewNamedRunDir(frameworkTTFTOutputPath, runName)
	if err != nil {
		fatal(fmt.Errorf("framework: creating ttft run directory %s under %s: %w", runName, frameworkTTFTOutputPath, err))
	}

	taskRuns := NewTaskRunStore(ttftDir)

	log.Printf("framework: ttft: stashing each task run's output under %s", taskRuns.Dir())

	tasks, err := frameworkTasks(gw, RunConfig{
		Models:              models,
		JudgeModel:          cfg.JudgeModel,
		MaxSamplesPerBucket: cfg.MaxSamplesPerBucket,
		Concurrency:         cfg.Concurrency,
		Buckets:             buckets,
		Failures:            failures,
		TaskRuns:            taskRuns,
		Replay:              replay.ttftRuns(),
		Debug:               cfg.Debug,
	}, requestedTasks)
	if err != nil {
		fatal(fmt.Errorf("framework: %w", err))
	}

	log.Printf("framework: ttft: benchmarking %d model(s) via streaming on %d task(s)", len(models), len(tasks))

	var (
		samples []Sample
		cells   []Cell
	)

	for _, runner := range tasks {
		taskSamples, taskCells, err := runner.RunTTFTBenchmark(ctx)
		if err != nil {
			fatal(fmt.Errorf("framework: ttft: running task %s: %w", runner.Name(), err))
		}

		samples = append(samples, taskSamples...)
		cells = append(cells, taskCells...)
	}

	ttftCells := cellsToTTFTCells(cells)
	PrintTTFTReport(os.Stdout, ttftCells)

	log.Printf("framework: ttft: recorded %d failure(s) under %s", failures.Count(), failures.RunDir())
	log.Printf("framework: ttft: stashed %d task run output(s) under %s", taskRuns.Count(), taskRuns.Dir())
	logReplay("ttft: ", replay.ttftRuns(), samples)

	if err := WriteTTFTJSONReport(ttftDir, samplesToTTFTSamples(samples), ttftCells); err != nil {
		fatal(fmt.Errorf("framework: writing ttft results: %w", err))
	}

	log.Printf("framework: ttft: wrote results to %s", ttftDir)

	return ttftDir
}

// sampleToTTFTSample and cellToTTFTCell adapt the framework's Sample/Cell to
// ttft.go's dedicated TTFTSample/TTFTCell shape. Judge/SchemaInvalid/
// Throughput/JudgeMeans have no TTFT counterpart and are dropped:
// Task.RunTTFT never populates the first two, and the TTFT report schema
// never carried the latter two.
func sampleToTTFTSample(s Sample) TTFTSample {
	return TTFTSample{
		Model:            s.Model,
		ReasoningEffort:  s.ReasoningEffort,
		Task:             s.Task,
		Bucket:           s.Bucket,
		RecordID:         s.RecordID,
		TimeToFirstToken: s.TimeToFirstToken,
		TotalLatency:     s.TotalLatency,
		CompletionTokens: s.CompletionTokens,
		PromptTokens:     s.PromptTokens,
		Err:              s.Err,
	}
}

func samplesToTTFTSamples(samples []Sample) []TTFTSample {
	out := make([]TTFTSample, 0, len(samples))
	for _, s := range samples {
		out = append(out, sampleToTTFTSample(s))
	}

	return out
}

func cellToTTFTCell(c Cell) TTFTCell {
	return TTFTCell{
		Model:           c.Model,
		ReasoningEffort: c.ReasoningEffort,
		Task:            c.Task,
		Bucket:          c.Bucket,
		N:               c.N,
		Errors:          c.Errors,
		TTFT:            c.TTFT,
		Latency:         c.Latency,
	}
}

func cellsToTTFTCells(cells []Cell) []TTFTCell {
	out := make([]TTFTCell, 0, len(cells))
	for _, c := range cells {
		out = append(out, cellToTTFTCell(c))
	}

	return out
}
