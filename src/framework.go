package main

import (
	"context"
	"fmt"

	"github.com/AssemblyAI/voice-standard-bench/internal/concurrency"
)

// Bucketed is the minimal capability Runner needs from a Task's record type:
// its own audio-length bucket, generically, without Runner needing to know
// the concrete record shape (TranscriptRecord, DictationRecord, ...). Every
// record type used with NewRunner implements it -- see e.g.
// TranscriptRecord.RecordBucket (dataset.go) and
// DictationRecord.RecordBucket (dictation.go). Named RecordBucket rather
// than Bucket to avoid colliding with those types' own Bucket field (Go
// disallows a method and a field sharing one name).
type Bucketed interface {
	RecordBucket() string
}

// Record is what Runner needs from a Task's record type: its bucket (Bucketed)
// and its ID, which together with the model and task name identify one task run
// -- the key a stashed result is found by when a run replays an earlier one (see
// RunConfig.Replay). Every record type used with NewRunner implements it -- see
// TranscriptRecord.RecordID (dataset.go) and DictationRecord.RecordID
// (dictation.go). Named RecordID rather than ID for the same reason as
// RecordBucket: those types have an ID field of their own.
type Record interface {
	Bucketed
	RecordID() string
}

// Task is a grouping of functionality that is needed to run an complete benchmark
// it has methods for all the consistent steps needed in order to run a benchmark
// so that we can organize code coherently, even if the core details of each
// task is different
// R generic will be the type of dataset record this task uses. Each record will contain
// which bucket it belongs too
type Task[R any] interface {
	// Name is the name of the task for bookkeeping.
	Name() string

	// LoadDataset contains all of the logic needed to load the dataset
	// needed in order to run the task. It will contain any relative paths,
	// file loading logic, loading from an external source like s3, etc.
	// It will reference libs as needed, to complete the loading.
	LoadDataset() ([]R, error)

	// SampleDataset will act on the data structure where we have stored the data
	// from LoadDataset, and then sample the data deterministically to be a subset
	// of the data up to maxSamples per bucket of data.
	SampleDataset(records []R, maxSamples int) []R

	// RunRecord executes the task's completion call for one record against
	// model. It always returns an identified Sample -- Model/ReasoningEffort/
	// Task/Bucket/RecordID populated (the model's ID and ReasoningEffortLabel,
	// the task's Name, the record's RecordBucket and RecordID -- the very values
	// a Runner looks a stashed run up by when it replays an earlier one, see
	// RunConfig.Replay), plus latency/token measurements on success or Err on
	// failure -- alongside the raw CompletionResult a following RunJudge call
	// needs. Returning a real Sample here, rather than only a CompletionResult,
	// is what lets a Runner report a failed record's error rate without having
	// to build that Sample itself. It records any failures to the
	// FailureRecorder.
	RunRecord(ctx context.Context, model ModelInfo, record R, failures *FailureRecorder) (Sample, CompletionResult, error)

	// RunTTFT is RunRecord's streaming counterpart, used purely to measure a
	// real time-to-first-token: the same record, but the underlying call
	// uses streaming completions instead of a synchronous one. Like
	// RunRecord it returns both an identified Sample and the raw
	// CompletionResult -- but a TTFT sample is never judged (its Judge is
	// always left empty/nil), so unlike RunRecord's output, nothing is scored
	// off of RunTTFT's CompletionResult; a Runner stashes it, as it does
	// RunRecord's, so a later run can replay the call (see RunConfig.Replay).
	// It records any failures to the FailureRecorder.
	RunTTFT(ctx context.Context, model ModelInfo, record R, failures *FailureRecorder) (Sample, CompletionResult, error)

	// RunJudge scores a candidate's output against record -- either via an
	// LLM call to judgeModel, or, for a task with a real answer key, entirely
	// mechanically (some tasks may not rely on an LLM to judge the score, and
	// thus judgeModel is ignored). It is only ever called after a successful
	// RunRecord for the same (model, record), so it may assume result
	// reflects a real completion. It records any failures -- a failed judge
	// call, or a mechanical score worth flagging on its own -- to the
	// FailureRecorder.
	//
	// Only the returned Sample's Judge and SchemaInvalid fields are used: a
	// Runner merges them onto the Sample RunRecord already produced rather
	// than replacing it outright, so an implementation need not (and should
	// not bother to) reconstruct Model/Task/Bucket/RecordID or copy result's
	// latency/token fields onto what it returns here.
	RunJudge(ctx context.Context, model ModelInfo, record R, judgeModel string, result CompletionResult, failures *FailureRecorder) (Sample, error)

	// DimensionRanges returns the dimension ranges for this task to be used in calculating metrics
	// and in the summary
	DimensionRanges() map[string]scale
}

// RunConfig bundles everything a Runner needs beyond the Task itself: which
// candidate models to benchmark, the judge model to grade against, how many
// records to sample per bucket, how much concurrency to allow, and where (if
// anywhere) to record failures.
type RunConfig struct {
	// Models are the candidate models under test.
	Models []ModelInfo
	// JudgeModel grades every candidate's output. It should not also appear
	// in Models (see the self-grading note on -judge-model in the original
	// voicebench.go), but Runner does not enforce that itself -- excluding it
	// is a caller/CLI-wiring concern, not a framework one.
	JudgeModel string
	// MaxSamplesPerBucket caps how many records SampleDataset draws per
	// bucket; <= 0 means "use every loaded record" (see SampleDataset's own
	// doc comment on Task).
	MaxSamplesPerBucket int
	// Concurrency bounds two SEPARATE things, both sized the same: how many
	// records run at once PER MODEL (see Runner's doc comment), and how many
	// judge calls run at once TOTAL, shared across every model. Values below
	// 1 are normalized to 1 by NewRunner.
	Concurrency int
	// Buckets, when non-empty, restricts LoadAndSample to just these
	// audio-length buckets (matched against R.RecordBucket() -- see Bucketed)
	// -- e.g. ["3600s"] to benchmark only the hour-long calls. Empty means no
	// filter at all: every bucket the loaded dataset has, the same "empty
	// means everything" convention -models/filterModels uses (voicebench.go).
	Buckets []string
	// Failures records failed samples for later inspection; nil disables
	// failure recording entirely, matching FailureRecorder's own contract.
	Failures *FailureRecorder
	// TaskRuns, when non-nil, receives every successful task call's result -- the
	// candidate's output -- the moment it comes back (RunRecord's, in Run;
	// RunTTFT's, in RunTTFT), before RunJudge sees it, so the judge step can be
	// repeated later against those same outputs without paying for the candidate
	// calls again (see TaskRunStore). nil disables it, matching Failures. A pass
	// is given a store of its own: the standard pass's and the TTFT pass's
	// outputs are different measurements and never share one.
	TaskRuns *TaskRunStore
	// Replay, when non-nil, is an earlier run's stash, consulted before every
	// RunRecord (in Run) and RunTTFT (in RunTTFT) call: a (model, record) whose
	// result is in it is not called at all -- its stashed Sample and
	// CompletionResult stand in for the call's -- and Run goes on to judge that
	// result live, as it would a fresh one. One that is NOT in it (the call
	// failed, or never ran: another model, a bigger sample) runs live as usual,
	// so a replay also finishes a run that was interrupted. A replayed result is
	// stashed to TaskRuns like any other, which keeps every run's directory
	// complete and lets a replay be replayed in turn. nil replays nothing. Like
	// TaskRuns, a pass is given the earlier run's matching store: what the
	// standard pass replays is not what the TTFT pass does.
	Replay *TaskRunStore
	// SkipJudge, when true, makes Run stop after the task call and its stash:
	// RunJudge is never called, so every sample comes back unjudged (for every
	// task, the mechanically scored ones included). It is for gathering outputs
	// now and judging them later, by replaying the run (see Replay).
	SkipJudge bool
	// Debug is the -debug flag: a task that has extra diagnostics to offer
	// prints them as it runs (today only text_correction does -- see
	// correction_debug.go). It never changes what is measured, recorded, or
	// merged.
	Debug bool
}

// Runner drives a Task[R] through the full benchmark lifecycle: load,
// sample, run, judge, and aggregate. It carries no task-specific logic of
// its own -- everything it does is generic over R, so the same Runner works
// for any Task, one type argument bound per call site (e.g.
// NewRunner[TranscriptRecord] for summarization/extraction/correction
// alongside a separate NewRunner[DictationRecord] for dictation).
//
// Concurrency is bounded two ways, mirroring why the original benchmark
// needed two separate caps (see gateway.go's NewGateway and voicebench.go's
// runStandardBenchmarkPass): RunConfig.Concurrency limits how many records
// run at once PER MODEL -- each candidate has its own per-account-per-model
// rate limit, so models don't compound against each other -- and a SECOND
// cap of the same size, SHARED across every model's goroutines, limits how
// many judge calls are in flight at once, since every candidate's samples
// grade against the very same judge model, which does not get that
// per-candidate isolation.
//
// RunConfig.Replay lets a Runner skip the task calls an earlier run already made
// and judge what that run stashed instead -- see callOrReplay
// (runner_internal.go), the one place the decision is made.
type Runner[R Record] struct {
	task   Task[R]
	config RunConfig
}

// NewRunner returns a Runner for task, configured by config. Concurrency
// values below 1 are normalized to 1 -- a concurrency.JobLimiter constructed
// with a max of 0 would block every job forever (see runner_internal.go).
func NewRunner[R Record](task Task[R], config RunConfig) *Runner[R] {
	if config.Concurrency < 1 {
		config.Concurrency = 1
	}

	return &Runner[R]{task: task, config: config}
}

// Name returns the underlying Task's name.
func (r *Runner[R]) Name() string {
	return r.task.Name()
}

// LoadAndSample loads task's full dataset, restricts it to config.Buckets
// (when set -- see RunConfig.Buckets), and samples what's left down to
// config.MaxSamplesPerBucket, ready to hand to Run and/or RunTTFT. Split out
// from those two methods so a caller running both passes (see
// RunBenchmark/RunTTFTBenchmark) loads and samples the dataset once and
// reuses the same records for each, rather than re-reading the dataset
// twice. Filtering to config.Buckets BEFORE sampling, rather than after,
// means a bucket the caller didn't ask for is dropped outright -- it never
// occupies any of MaxSamplesPerBucket's budget, and RunTTFT's own further
// restriction to TTFTBuckets composes on top of whatever this leaves.
func (r *Runner[R]) LoadAndSample() ([]R, error) {
	records, err := r.task.LoadDataset()
	if err != nil {
		return nil, fmt.Errorf("task %s: loading dataset: %w", r.task.Name(), err)
	}

	if len(r.config.Buckets) > 0 {
		records = filterBucketed(records, r.config.Buckets)
	}

	return r.task.SampleDataset(records, r.config.MaxSamplesPerBucket), nil
}

// Run executes the standard, judged pass over records: every model against
// every record, bounded per model by config.Concurrency, with each
// successful RunRecord call immediately followed by a RunJudge call bounded
// by the separate, shared judge concurrency described on Runner. A
// RunRecord failure short-circuits straight to its own Sample -- RunJudge is
// never called for a record whose task call didn't succeed, since there is
// nothing to grade. A success is stashed to config.TaskRuns (when set) first,
// so a judge call that fails, or a run that dies, never loses an output that
// was already paid for. With config.Replay set, a (model, record) the earlier
// run already stashed skips RunRecord altogether and goes straight to the judge.
//
// Run must never run concurrently with RunTTFT against the same models:
// mixing streaming and synchronous latency numbers for the same candidate at
// the same time corrupts both measurements (see RunTTFT's doc comment).
func (r *Runner[R]) Run(ctx context.Context, records []R) []Sample {
	judgeLimiter := concurrency.NewJobLimiter(r.config.Concurrency)

	return r.runAll(ctx, records, func(ctx context.Context, model ModelInfo, record R) Sample {
		return r.runRecord(ctx, model, record, judgeLimiter)
	})
}

// RunTTFT executes the streaming, unjudged pass over records, purely to
// measure a real time-to-first-token per (model, record) -- see RunTTFT on
// the Task interface. It shares Run's per-model concurrency shape but has no
// judge step, so it needs no judge concurrency cap. Nothing downstream of a
// TTFT pass needs the raw candidate output, only its Sample, so the Task's
// CompletionResult goes no further than config.TaskRuns, where it is stashed so
// a later run can replay the call (config.Replay) rather than stream again. As
// in Run, a (model, record) the earlier run's stash holds is not called at all.
//
// records is first restricted to TTFTBuckets (see ttft.go's package doc
// comment on why TTFT only ever measures 30s/120s, the same restriction the
// original TaskGroup-based pipeline applies via filterGroupsByBucket) --
// every other bucket is dropped before a single streaming call is made, so a
// caller need not pre-filter records itself. This narrows further still
// when records already came from a config.Buckets-restricted LoadAndSample
// (e.g. -buckets 3600s composes with TTFTBuckets down to no records at all,
// since TTFT never measures anything past 120s).
//
// Keep a TTFT pass's samples in their own []Sample / Aggregate call, never
// pooled with a Run pass's samples for the same model: a synchronous call's
// latency and a streaming call's latency are not the same measurement, and
// averaging them together corrupts both -- see RunBenchmark/RunTTFTBenchmark,
// which keep the two passes' Aggregate calls entirely separate for exactly
// this reason.
func (r *Runner[R]) RunTTFT(ctx context.Context, records []R) []Sample {
	records = filterBucketed(records, TTFTBuckets)

	return r.runAll(ctx, records, func(ctx context.Context, model ModelInfo, record R) Sample {
		return r.runTTFTRecord(ctx, model, record)
	})
}

// filterBucketed returns the subset of records whose own RecordBucket falls
// within buckets, preserving order. Shared by LoadAndSample (RunConfig.Buckets,
// a caller-chosen allowlist) and RunTTFT (the fixed TTFTBuckets) -- both are
// "keep only these buckets" filters over a Bucketed record type, just with
// different sources for buckets. Also RunTTFT's generic counterpart to
// ttft.go's filterGroupsByBucket, which does the same job for the original
// TaskGroup-based pipeline.
func filterBucketed[R Bucketed](records []R, buckets []string) []R {
	allowed := make(map[string]bool, len(buckets))
	for _, b := range buckets {
		allowed[b] = true
	}

	filtered := make([]R, 0, len(records))

	for _, rec := range records {
		if allowed[rec.RecordBucket()] {
			filtered = append(filtered, rec)
		}
	}

	return filtered
}

// RunBenchmark is the one-call entry point for the standard pipeline: load,
// sample, run, judge, and aggregate every resulting Sample into per (model,
// task, bucket) Cells via Aggregate (metrics.go).
func (r *Runner[R]) RunBenchmark(ctx context.Context) ([]Sample, []Cell, error) {
	records, err := r.LoadAndSample()
	if err != nil {
		return nil, nil, err
	}

	samples := r.Run(ctx, records)

	return samples, Aggregate(samples), nil
}

// RunTTFTBenchmark is RunBenchmark's TTFT counterpart: load, sample, run the
// streaming pass, and aggregate. RunTTFT itself restricts the streaming pass
// to TTFTBuckets, so the records LoadAndSample returns here span every
// bucket the standard benchmark does, same as RunBenchmark -- only RunTTFT
// narrows them. See RunTTFT's doc comment for why this must not run
// concurrently with RunBenchmark against the same models.
func (r *Runner[R]) RunTTFTBenchmark(ctx context.Context) ([]Sample, []Cell, error) {
	records, err := r.LoadAndSample()
	if err != nil {
		return nil, nil, err
	}

	samples := r.RunTTFT(ctx, records)

	return samples, Aggregate(samples), nil
}
