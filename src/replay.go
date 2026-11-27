// Replaying an earlier run (-replay): re-running a benchmark's judge step over
// the candidate outputs an earlier run already collected, without calling the
// candidate models again.
//
// This file is the part that lives above Runner: finding the earlier run's
// stash from its name, refusing -- before a single model is called -- one that
// couldn't be replayed faithfully, and saying afterwards how much of each pass
// came out of it. The part inside Runner, which actually skips a task call for
// a stashed (model, record), is RunConfig.Replay and callOrReplay (framework.go,
// runner_internal.go).
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

// replaySource is an earlier run's stashed task runs, opened for -replay. A run
// is one name across the two places it staged: results/<name> holds what its
// standard pass kept and results_ttft/<name> what its TTFT pass did (see
// NewNamedRunDir) -- two stores, because the passes' outputs are different
// measurements and each pass replays only its own.
type replaySource struct {
	name     string
	standard *TaskRunStore
	ttft     *TaskRunStore
}

// standardRuns and ttftRuns are the stores each pass hands its Runners as
// RunConfig.Replay. Both are safe on a nil *replaySource -- no -replay -- and
// then return nil, which replays nothing.
func (r *replaySource) standardRuns() *TaskRunStore {
	if r == nil {
		return nil
	}

	return r.standard
}

func (r *replaySource) ttftRuns() *TaskRunStore {
	if r == nil {
		return nil
	}

	return r.ttft
}

// replayRunName turns -replay's value into a run name. The flag takes the name
// itself, or the path its directory was logged at (../results/<name>, with or
// without a trailing slash -- what a tab-completed argument looks like); only
// the last element is kept, which is also what stops a value like ../../elsewhere
// from pointing anywhere but results/ and results_ttft/.
func replayRunName(flagValue string) string {
	return filepath.Base(filepath.Clean(strings.TrimSpace(flagValue)))
}

// openReplaySource finds the earlier run flagValue names and opens its stashes
// for replaying, or says why it can't be replayed -- before any model has been
// called, since what a replay is for is not spending money on calls it already
// has the answers to:
//
//   - there is no such run, or it stashed nothing (a mistyped name would
//     otherwise just run the whole benchmark live, at full price, under a flag
//     that promises not to);
//   - a requested task's prompt has changed since the run stashed its outputs
//     (see checkReplayPrompts).
//
// A run that has stashed only one pass is fine -- a run that was interrupted
// before its TTFT pass has no TTFT stash -- and is warned about: whatever isn't
// stashed runs live.
func openReplaySource(flagValue string, requestedTasks []string) (*replaySource, error) {
	name := replayRunName(flagValue)

	if name == "." || name == ".." || name == string(filepath.Separator) {
		return nil, fmt.Errorf("-replay %q is not a run name: pass the name of a directory under %s, or its path", flagValue, frameworkOutputPath)
	}

	if info, err := os.Stat(filepath.Join(frameworkOutputPath, name)); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("-replay %q: there is no run named %s under %s", flagValue, name, frameworkOutputPath)
	}

	source := &replaySource{
		name:     name,
		standard: NewTaskRunStore(filepath.Join(frameworkOutputPath, name)),
		ttft:     NewTaskRunStore(filepath.Join(frameworkTTFTOutputPath, name)),
	}

	standard, ttft := source.standard.Stashed(), source.ttft.Stashed()

	if standard+ttft == 0 {
		return nil, fmt.Errorf("-replay %q: run %s stashed no task runs (looked under %s and %s), so there is nothing to replay; only a run made since task outputs were stashed can be",
			flagValue, name, source.standard.Dir(), source.ttft.Dir())
	}

	log.Printf("framework: replay: run %s stashed %d standard and %d ttft task run(s): each (model, record) among them is reused instead of calling its model again; any other runs live", name, standard, ttft)

	if standard == 0 {
		log.Printf("framework: replay: warning: run %s stashed no standard task runs (%s): every standard call will run live", name, source.standard.Dir())
	}

	if ttft == 0 {
		log.Printf("framework: replay: warning: run %s stashed no ttft task runs (%s): every ttft call will run live", name, source.ttft.Dir())
	}

	if err := checkReplayPrompts(source, requestedTasks); err != nil {
		return nil, err
	}

	return source, nil
}

// checkReplayPrompts refuses a replay whose stashed outputs were made with a
// different task prompt than the one this build would send now. Replaying them
// would score the old prompt's outputs as if the new prompt had produced them --
// a wrong number in a benchmark, with nothing to show it -- so this is an
// error, not a warning, and it comes before any model is called.
//
// Every run of a task in one stash was sent the same system prompt, so reading
// one stashed request per task and pass is enough. A task that stashed nothing
// has nothing to go stale. What it does NOT check is the record text: a dataset
// regenerated under the same record IDs would be replayed as if unchanged.
//
// A stashed request whose system prompt can't be read is warned about and let
// through, not refused: it says nothing about the prompt having changed, and
// failing every replay over a request format this check doesn't understand would
// be worse.
func checkReplayPrompts(source *replaySource, requestedTasks []string) error {
	passes := []struct {
		label string
		store *TaskRunStore
	}{
		{"standard", source.standard},
		{"ttft", source.ttft},
	}

	for _, task := range requestedTasks {
		spec, ok := prompts.For(task)
		if !ok {
			continue // parseTasks has already rejected a name that isn't a task
		}

		for _, pass := range passes {
			file, ok, err := pass.store.anyRun(task)
			if err != nil {
				return fmt.Errorf("replay: reading run %s's stashed %s %s runs: %w", source.name, pass.label, task, err)
			}

			if !ok {
				continue
			}

			stashed, err := file.systemPrompt()
			if err != nil {
				log.Printf("framework: replay: warning: can't tell which %s prompt run %s's stashed %s runs were made with, replaying them as they are: %v", task, source.name, pass.label, err)

				continue
			}

			if stashed != spec.Task {
				return fmt.Errorf("replay: the %s prompt has changed since run %s stashed its %s runs, so they are not what this build would ask the models: replaying them would score the old prompt's outputs as the new one's. Run again without -replay, or leave %s out of -tasks",
					task, source.name, pass.label, task)
			}
		}
	}

	return nil
}

// checkSelection refuses a replay that would reuse nothing: none of the
// records this run will actually select — after the -buckets filter and the
// same deterministic per-bucket sampling both passes apply (see
// Runner.LoadAndSample) — has a stashed standard run for any model this run
// calls, at the reasoning effort it will call them at, for any task it runs;
// and nothing in the TTFT stash's half of the streaming buckets either (see
// TTFTBuckets: the TTFT pass never measures outside them). It answers over
// the SELECTED records, not over the stash as a whole, because those are the
// only lookups the run will ever make: a stash full of 30s outputs replays
// nothing for a -buckets 120s run, and a stash of record IDs a regenerated
// dataset no longer carries is all misses — either way the result would be
// every candidate call made live, at full price, under a flag that exists to
// avoid that, so it is refused, with what the run stashed next to what this
// one asks for. Some reuse is enough, in either pass: a replay that adds a
// model, a bucket, or a record the run didn't have is fine, and runs the new
// ones live.
//
// It samples the datasets itself, through the same task constructors and
// LoadAndSample the passes use, because it runs before them — before any
// candidate completion is requested. The gateway has already been called
// once, for the model discovery that resolves the -models list (see
// runFrameworkBenchmarkPass); that is all. A nil *Gateway is enough for the
// task constructors here: they only resolve prompts and dataset paths. The
// stores it looks things up in are fresh views of the run's stashes rather
// than the replaySource's own, so these preflight lookups don't count in the
// Reused tallies the passes report afterwards (see logReplay).
func (r *replaySource) checkSelection(models []ModelInfo, requestedTasks, buckets []string, maxSamples int) error {
	tasks, err := frameworkTasks(nil, RunConfig{
		Models: models, Buckets: buckets, MaxSamplesPerBucket: maxSamples,
	}, requestedTasks)
	if err != nil {
		return err
	}
	standard := NewTaskRunStore(filepath.Dir(r.standard.Dir()))
	ttft := NewTaskRunStore(filepath.Dir(r.ttft.Dir()))
	for _, task := range tasks {
		records, err := task.replayRecords()
		if err != nil {
			return err
		}
		for _, model := range models {
			for _, record := range records {
				if _, _, ok, err := standard.Load(model.ID, model.ReasoningEffortLabel(), task.Name(), record.RecordBucket(), record.RecordID()); err == nil && ok {
					return nil
				}
			}
			for _, record := range filterBucketed(records, TTFTBuckets) {
				if _, _, ok, err := ttft.Load(model.ID, model.ReasoningEffortLabel(), task.Name(), record.RecordBucket(), record.RecordID()); err == nil && ok {
					return nil
				}
			}
		}
	}

	asked := make([]string, 0, len(models))
	for _, model := range models {
		asked = append(asked, model.ID+" ("+model.ReasoningEffortLabel()+")")
	}

	return fmt.Errorf("replay: nothing to reuse: run %s stashed no usable standard or ttft run for the selected records, so replaying it would call everything live. This run calls %s, for %s; the run stashed standard %s, ttft %s. Match -models, -tasks, -reasoning-effort, -buckets and sampled record IDs to the run you are replaying",
		r.name, listUpTo(asked, 8), strings.Join(requestedTasks, ", "), listUpTo(r.standard.stashedModels(), 8), listUpTo(r.ttft.stashedModels(), 8))
}

// listUpTo joins items with ", ", cutting a long list off after max of them.
func listUpTo(items []string, max int) string {
	if len(items) == 0 {
		return "nothing"
	}

	if len(items) <= max {
		return strings.Join(items, ", ")
	}

	return fmt.Sprintf("%s and %d more", strings.Join(items[:max], ", "), len(items)-max)
}

// logReplay says, per task, how much of a pass came out of the replayed run's
// stash and how much was called live -- the one place a replay that meant to
// spend nothing on candidate calls, and did, shows. label prefixes the line
// ("" for the standard pass, "ttft: " for the other, like the pass's other
// log lines). samples is every sample the pass produced; a task's live count is
// its samples less the ones that were reused, so it includes live calls that
// then failed.
func logReplay(label string, store *TaskRunStore, samples []Sample) {
	if store == nil {
		return
	}

	totals := map[string]int{}
	for _, s := range samples {
		totals[s.Task]++
	}

	tasks := make([]string, 0, len(totals))
	for task := range totals {
		tasks = append(tasks, task)
	}

	sort.Strings(tasks)

	for _, task := range tasks {
		reused := store.Reused(task)

		log.Printf("framework: %sreplay: %s: reused %d of %d task run(s) from %s, called %d live", label, task, reused, totals[task], store.Dir(), totals[task]-reused)
	}
}
