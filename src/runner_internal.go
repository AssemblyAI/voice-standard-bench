// Runner's concurrency mechanics: the per-model/per-record fan-out every
// pass (Run, RunTTFT) shares, and the plumbing that lets a RunJudge call
// nest a SHARED, cross-model concurrency gate inside a goroutine that is
// already running inside its own per-model gate. See framework.go for the
// public Runner API these functions implement.
package main

import (
	"context"
	"log"
	"sync"

	"github.com/AssemblyAI/voice-standard-bench/internal/concurrency"
)

// runAll fans work out across every model in config.Models, fully in
// parallel with each other (candidate models don't share a rate-limit
// budget -- see Runner's doc comment), and within each model, across every
// record bounded by a fresh per-model concurrency.Limiter sized to
// config.Concurrency. perRecord does whatever the calling pass (Run or
// RunTTFT) needs for one (model, record) pair and returns its Sample;
// runAll only owns the fan-out, the concurrency bound, and thread-safe
// collection -- the one shape every pass shares.
func (r *Runner[R]) runAll(ctx context.Context, records []R, perRecord func(ctx context.Context, model ModelInfo, record R) Sample) []Sample {
	collect, samples := newSampleCollector()

	var modelsWG sync.WaitGroup

	for _, model := range r.config.Models {
		modelsWG.Add(1)

		go func(model ModelInfo) {
			defer modelsWG.Done()

			recordLimiter := concurrency.NewJobLimiter(r.config.Concurrency)

			for _, record := range records {
				record := record

				recordLimiter.Run(func() error {
					collect(perRecord(ctx, model, record))
					return nil
				})
			}

			recordLimiter.WaitForAll()
		}(model)
	}

	modelsWG.Wait()

	return samples()
}

// runRecord runs one (model, record) pair through RunRecord -- or takes the
// result an earlier run stashed for it, when config.Replay holds one (see
// callOrReplay) -- and, on success, immediately through RunJudge, bridging the
// RunJudge call through judgeLimiter, a concurrency gate SHARED across every
// model's goroutines (see Runner's doc comment), even though runRecord itself is
// already running inside one model's own, separate per-record concurrency slot
// (see runAll). Also stamps Sample.CostUSD from model's pricing here, the one
// place both model and RunRecord's real token usage are in scope -- and stashes
// each successful result to config.TaskRuns, ahead of the judge call: every
// task's RunRecord passes through here, so that is done once, for all of them,
// rather than in each task.
func (r *Runner[R]) runRecord(ctx context.Context, model ModelInfo, record R, judgeLimiter concurrency.Limiter) Sample {
	sample, result, err := r.callOrReplay(ctx, model, record, r.task.RunRecord)
	if err != nil {
		return sample
	}

	sample.CostUSD = model.Cost(sample.PromptTokens, sample.CompletionTokens)

	r.stashTaskRun(sample, result)

	if r.config.SkipJudge {
		return sample
	}

	judged, err := runSync(judgeLimiter, func() (Sample, error) {
		return r.task.RunJudge(ctx, model, record, r.config.JudgeModel, result, r.config.Failures)
	})
	if err != nil {
		// The task call succeeded; only the quality score is missing.
		// Latency/token measurements from RunRecord are still valid
		// measurements and stay in `sample`.
		return sample
	}

	// Merge RunJudge's quality-scoring fields onto RunRecord's sample rather
	// than replacing it outright: RunJudge is only responsible for Judge and
	// SchemaInvalid (see its doc comment on the Task interface), so a Task
	// implementation never has to reconstruct Model/Task/Bucket/RecordID or
	// copy result's latency/token fields back onto its returned Sample just
	// to avoid silently zeroing them here.
	sample.Judge = judged.Judge
	sample.SchemaInvalid = judged.SchemaInvalid

	return sample
}

// runTTFTRecord is runRecord's streaming counterpart: one (model, record) pair
// through RunTTFT -- or the stashed result of an earlier run's, when
// config.Replay holds it -- and nothing after, since a TTFT sample is never
// judged. Its result is stashed like a standard one, so a later run can replay
// it. A failed call's Sample comes back as it is, Err and all, and is never
// stashed.
func (r *Runner[R]) runTTFTRecord(ctx context.Context, model ModelInfo, record R) Sample {
	sample, result, err := r.callOrReplay(ctx, model, record, r.task.RunTTFT)
	if err == nil {
		r.stashTaskRun(sample, result)
	}

	return sample
}

// callOrReplay makes one task call for (model, record) -- call is the Task's
// RunRecord or its RunTTFT -- unless config.Replay already holds that run's
// result, in which case the stashed Sample and CompletionResult are returned in
// the call's place and the model is never asked. It is the one place a replay is
// decided, for both passes.
//
// The run is looked up by the identity every task puts in the Sample it returns
// -- the model's ID and reasoning effort label, the task's name, the record's
// bucket and ID -- which is the identity TaskRunStore.Save files it under, so a
// run saved by one pass is found by the next (a Task that filled its Sample's
// identity in differently would save runs a replay could never find, and every
// one of them would run live).
//
// A run that isn't in the stash is not an error: the call runs live, exactly as
// it would without a Replay -- that's what lets a replay add a model or finish an
// interrupted run. Neither is a stashed file that can't be trusted (unreadable,
// or holding some other run -- see TaskRunStore.Load): that is logged, and the
// call runs live too. The one thing not to do with a stash that can't be trusted
// is use it.
func (r *Runner[R]) callOrReplay(ctx context.Context, model ModelInfo, record R, call func(context.Context, ModelInfo, R, *FailureRecorder) (Sample, CompletionResult, error)) (Sample, CompletionResult, error) {
	if r.config.Replay != nil {
		sample, result, ok, err := r.config.Replay.Load(model.ID, model.ReasoningEffortLabel(), r.task.Name(), record.RecordBucket(), record.RecordID())

		switch {
		case err != nil:
			log.Printf("model %s task %s record %s: not replaying its stashed task run, calling it live instead: %v", model.ID, r.task.Name(), record.RecordID(), err)
		case ok:
			return sample, result, nil
		}
	}

	return call(ctx, model, record, r.config.Failures)
}

// stashTaskRun saves one successful task run to config.TaskRuns, if there is
// one. A failed write is logged, not returned: a benchmark that is otherwise
// succeeding shouldn't abort, or drop a sample it has already paid for, over
// trouble writing what is only a copy of its own output -- the same call
// recordFailure makes for its own diagnostic files.
func (r *Runner[R]) stashTaskRun(sample Sample, result CompletionResult) {
	if r.config.TaskRuns == nil {
		return
	}

	if err := r.config.TaskRuns.Save(sample, result); err != nil {
		log.Printf("model %s task %s record %s: stashing task run: %v", sample.Model, sample.Task, sample.RecordID, err)
	}
}

// newSampleCollector returns a collect function safe to call concurrently
// from many goroutines, and a samples function returning everything
// collected so far -- the shared-mutex-protected-slice pattern runAll needs
// since every model's goroutines, and every record's job within them, feed
// into the same final []Sample.
func newSampleCollector() (collect func(Sample), samples func() []Sample) {
	var (
		mu  sync.Mutex
		all []Sample
	)

	collect = func(s Sample) {
		mu.Lock()
		defer mu.Unlock()
		all = append(all, s)
	}

	samples = func() []Sample {
		mu.Lock()
		defer mu.Unlock()
		return append([]Sample(nil), all...)
	}

	return collect, samples
}

// runSync submits fn to limiter and blocks until fn itself has finished,
// returning fn's result directly. concurrency.Limiter.Run is fire-and-forget
// -- it launches fn in a goroutine and returns immediately, only ever
// waitable in aggregate via WaitForAll -- which is the right shape for
// runAll's outer per-record fan-out, but not for nesting a SHARED gate
// (judgeLimiter) inside a caller that is itself already running inside its
// own per-model slot and needs this specific call's result before it can
// return. The result travels over a buffered channel rather than a captured
// variable so there is no shared mutable state between the two goroutines to
// reason about beyond the channel handoff itself.
func runSync[T any](limiter concurrency.Limiter, fn func() (T, error)) (T, error) {
	type outcome struct {
		result T
		err    error
	}

	out := make(chan outcome, 1)

	limiter.Run(func() error {
		result, err := fn()
		out <- outcome{result, err}
		return err
	})

	o := <-out

	return o.result, o.err
}
