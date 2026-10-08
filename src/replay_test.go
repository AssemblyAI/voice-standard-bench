package main

import (
	"bytes"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

// captureLog redirects the standard logger into a buffer for the rest of the test.
func captureLog(t *testing.T) *bytes.Buffer {
	t.Helper()

	var buf bytes.Buffer

	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	return &buf
}

// stashEarlierRun stashes one task run of task into the earlier run called name,
// under root (frameworkOutputPath for the standard pass, frameworkTTFTOutputPath
// for the TTFT pass), the way a real pass does: through TaskRunStore.Save. The
// request it was made with has systemPrompt as its system message, which is what
// openReplaySource compares with the prompt this build would send.
func stashEarlierRun(t *testing.T, root, name, task, systemPrompt string) {
	t.Helper()

	request, err := json.Marshal(map[string]any{
		"model": "candidate-model",
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": "a call"},
		},
	})
	require.NoError(t, err)

	sample := Sample{Model: "candidate-model", ReasoningEffort: "default", Task: task, Bucket: "30s", RecordID: "rec-1", TotalLatency: time.Second}
	require.NoError(t, NewTaskRunStore(filepath.Join(root, name)).Save(sample, CompletionResult{Text: "an output", RawRequest: string(request)}))
}

func TestReplayRunName(t *testing.T) {
	tests := []struct {
		flag string
		want string
	}{
		{"2026-10-08-abc", "2026-10-08-abc"},
		{"../results/2026-10-08-abc", "2026-10-08-abc"},
		{"../results/2026-10-08-abc/", "2026-10-08-abc"},
		{"/home/me/repo/results/2026-10-08-abc", "2026-10-08-abc"},
		{"  2026-10-08-abc\n", "2026-10-08-abc"},
		{"../../elsewhere", "elsewhere"}, // only the last element survives: it can name nothing outside results/
	}

	for _, tt := range tests {
		assert.Equal(t, tt.want, replayRunName(tt.flag), "%q", tt.flag)
	}
}

// TestOpenReplaySource_opensARunThatStashedBothPasses: the name finds the run's
// standard stash under results/ and its TTFT stash under results_ttft/.
func TestOpenReplaySource_opensARunThatStashedBothPasses(t *testing.T) {
	t.Chdir(t.TempDir())
	stashEarlierRun(t, frameworkOutputPath, "run-1", prompts.Summarization, mustPrompt(t, prompts.Summarization))
	stashEarlierRun(t, frameworkTTFTOutputPath, "run-1", prompts.Summarization, mustPrompt(t, prompts.Summarization))

	source, err := openReplaySource("../results/run-1", []string{prompts.Summarization})

	require.NoError(t, err)
	assert.Equal(t, "run-1", source.name)
	assert.Equal(t, 1, source.standardRuns().Stashed())
	assert.Equal(t, 1, source.ttftRuns().Stashed())
	assert.Equal(t, filepath.Join(frameworkOutputPath, "run-1", taskRunsDirName), source.standardRuns().Dir())
	assert.Equal(t, filepath.Join(frameworkTTFTOutputPath, "run-1", taskRunsDirName), source.ttftRuns().Dir())
}

// TestOpenReplaySource_aRunThatStashedOnlyOnePassIsWarnedAboutNotRefused: a run
// that died before its TTFT pass has only a standard stash; whatever isn't
// stashed runs live, and the log says so.
func TestOpenReplaySource_aRunThatStashedOnlyOnePassIsWarnedAboutNotRefused(t *testing.T) {
	t.Chdir(t.TempDir())
	stashEarlierRun(t, frameworkOutputPath, "run-1", prompts.Summarization, mustPrompt(t, prompts.Summarization))

	logged := captureLog(t)

	source, err := openReplaySource("run-1", []string{prompts.Summarization})

	require.NoError(t, err)
	assert.Zero(t, source.ttftRuns().Stashed())
	assert.Contains(t, logged.String(), "stashed no ttft task runs")
	assert.NotContains(t, logged.String(), "stashed no standard task runs")
}

// TestOpenReplaySource_refusesARunItCannotReplay: every one of these is caught
// before a model is called, because a replay is meant to spend nothing on calls
// whose answers it already has.
func TestOpenReplaySource_refusesARunItCannotReplay(t *testing.T) {
	summarization := mustPrompt(t, prompts.Summarization)

	tests := []struct {
		name  string
		setup func(t *testing.T)
		flag  string
		tasks []string
		want  string // a fragment the error must carry
	}{
		{
			name:  "there is no such run",
			flag:  "run-typo",
			tasks: []string{prompts.Summarization},
			want:  "no run named run-typo",
		},
		{
			name:  "the flag names no run",
			flag:  "..",
			tasks: []string{prompts.Summarization},
			want:  "not a run name",
		},
		{
			name: "the run stashed nothing: no stashed task runs, only a report",
			setup: func(t *testing.T) {
				require.NoError(t, os.MkdirAll(filepath.Join(frameworkOutputPath, "run-1"), 0o755))
				writeFile(t, filepath.Join(frameworkOutputPath, "run-1", "report-judge.json"), "{}")
			},
			flag:  "run-1",
			tasks: []string{prompts.Summarization},
			want:  "stashed no task runs",
		},
		{
			name: "canonical is a directory under results/, not a run",
			setup: func(t *testing.T) {
				require.NoError(t, os.MkdirAll(frameworkCanonicalRoot, 0o755))
			},
			flag:  "canonical",
			tasks: []string{prompts.Summarization},
			want:  "stashed no task runs",
		},
		{
			name: "the task prompt changed since the standard runs were stashed",
			setup: func(t *testing.T) {
				stashEarlierRun(t, frameworkOutputPath, "run-1", prompts.Summarization, "an older prompt")
			},
			flag:  "run-1",
			tasks: []string{prompts.Summarization},
			want:  "the summarization prompt has changed since run run-1 stashed its standard runs",
		},
		{
			name: "...or the TTFT runs",
			setup: func(t *testing.T) {
				stashEarlierRun(t, frameworkOutputPath, "run-1", prompts.Summarization, summarization)
				stashEarlierRun(t, frameworkTTFTOutputPath, "run-1", prompts.Summarization, "an older prompt")
			},
			flag:  "run-1",
			tasks: []string{prompts.Summarization},
			want:  "stashed its ttft runs",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Chdir(t.TempDir())

			if tt.setup != nil {
				tt.setup(t)
			}

			source, err := openReplaySource(tt.flag, tt.tasks)

			require.Error(t, err)
			assert.Nil(t, source)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

// TestOpenReplaySource_onlyChecksThePromptsOfTheTasksBeingRun: a stale prompt in a
// task this run isn't running -- or a task the earlier run never stashed -- is no
// reason to refuse.
func TestOpenReplaySource_onlyChecksThePromptsOfTheTasksBeingRun(t *testing.T) {
	t.Chdir(t.TempDir())
	stashEarlierRun(t, frameworkOutputPath, "run-1", prompts.Summarization, mustPrompt(t, prompts.Summarization))
	stashEarlierRun(t, frameworkOutputPath, "run-1", prompts.TextCorrection, "an older text_correction prompt")

	source, err := openReplaySource("run-1", []string{prompts.Summarization, prompts.Dictation})

	require.NoError(t, err)
	assert.NotNil(t, source)
}

// TestOpenReplaySource_letsThroughARequestWhosePromptItCannotRead: not being able
// to tell which prompt a stash was made with is no evidence it changed, and
// refusing every replay over a request shape this check doesn't understand would
// be worse than the risk it guards -- so it warns and replays.
func TestOpenReplaySource_letsThroughARequestWhosePromptItCannotRead(t *testing.T) {
	t.Chdir(t.TempDir())

	sample := Sample{Model: "candidate-model", ReasoningEffort: "default", Task: prompts.Summarization, Bucket: "30s", RecordID: "rec-1"}
	require.NoError(t, NewTaskRunStore(filepath.Join(frameworkOutputPath, "run-1")).Save(sample, CompletionResult{
		Text:       "an output",
		RawRequest: `{"model":"candidate-model","messages":[{"role":"user","content":"a call"}]}`, // no system message
	}))

	logged := captureLog(t)

	source, err := openReplaySource("run-1", []string{prompts.Summarization})

	require.NoError(t, err)
	assert.NotNil(t, source)
	assert.Contains(t, logged.String(), "can't tell which summarization prompt")
}

// TestReplaySource_nilReplaysNothing: no -replay is a nil *replaySource, and the
// passes ask it for their stores unconditionally.
func TestReplaySource_nilReplaysNothing(t *testing.T) {
	var source *replaySource

	assert.Nil(t, source.standardRuns())
	assert.Nil(t, source.ttftRuns())
}

// TestLogReplay: per task, how much of the pass came out of the stash -- the one
// place a replay that was meant to spend nothing on candidate calls, and did,
// shows.
func TestLogReplay(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()

	// summarization: 3 stashed, all of which the pass reuses; text_correction: 1 of 2.
	for _, run := range []struct{ task, record string }{
		{"summarization", "rec-0"}, {"summarization", "rec-1"}, {"summarization", "rec-2"}, {"text_correction", "rec-0"},
	} {
		sample.Task, sample.RecordID = run.task, run.record
		require.NoError(t, store.Save(sample, result))

		_, _, ok, err := loadRun(store, sample)
		require.NoError(t, err)
		require.True(t, ok)
	}

	samples := []Sample{
		{Task: "text_correction"}, {Task: "summarization"}, {Task: "summarization"}, {Task: "summarization"}, {Task: "text_correction"},
	}

	logged := captureLog(t)

	logReplay("ttft: ", store, samples)

	out := logged.String()
	assert.Contains(t, out, "framework: ttft: replay: summarization: reused 3 of 3 task run(s)")
	assert.Contains(t, out, "called 0 live")
	assert.Contains(t, out, "framework: ttft: replay: text_correction: reused 1 of 2 task run(s)")
	assert.Contains(t, out, "called 1 live")
	assert.Less(t, bytes.Index([]byte(out), []byte("summarization")), bytes.Index([]byte(out), []byte("text_correction")), "tasks are listed in order")

	// No replay, nothing to say.
	logged.Reset()
	logReplay("", nil, samples)
	assert.Empty(t, logged.String())
}

// mustPrompt is the task prompt this build sends for task.
func mustPrompt(t *testing.T, task string) string {
	t.Helper()

	spec, ok := prompts.For(task)
	require.True(t, ok)

	return spec.Task
}

// TestReplaySource_checkSelection: a replay with nothing to reuse is the whole
// benchmark called live at full price under a flag that exists to avoid that, so
// it is refused -- and the likeliest way into it is -models, -tasks or
// -reasoning-effort not matching the run being replayed: every run is stashed under
// its model AND its reasoning effort, so forgetting -reasoning-effort turns every
// lookup into a miss.
func TestReplaySource_checkSelection(t *testing.T) {
	t.Chdir(t.TempDir())

	// The earlier run: gpt-6-luna at reasoning effort "none", and a model whose ID
	// has a "/" in it (a file name says "openai_gpt-4o"), both on summarization.
	standard := NewTaskRunStore(filepath.Join(frameworkOutputPath, "run-1"))

	for _, run := range []struct{ model, effort string }{{"gpt-6-luna", "none"}, {"openai/gpt-4o", "default"}} {
		sample := Sample{Model: run.model, ReasoningEffort: run.effort, Task: prompts.Summarization, Bucket: "30s", RecordID: "rec-1"}
		require.NoError(t, standard.Save(sample, CompletionResult{Text: "an output"}))
	}

	source := &replaySource{name: "run-1", standard: standard, ttft: NewTaskRunStore(filepath.Join(frameworkTTFTOutputPath, "run-1"))}

	none := "none"
	lunaNone := ModelInfo{ID: "gpt-6-luna", ReasoningEffort: &none}
	lunaDefault := ModelInfo{ID: "gpt-6-luna"} // -reasoning-effort not passed: the label is "default"
	gpt4o := ModelInfo{ID: "openai/gpt-4o"}
	newModel := ModelInfo{ID: "a-new-model"}

	summarization := []string{prompts.Summarization}

	t.Run("a model and effort the run stashed", func(t *testing.T) {
		assert.NoError(t, source.checkSelection([]ModelInfo{lunaNone}, summarization))
	})

	t.Run("a model whose ID has a slash in it", func(t *testing.T) {
		assert.NoError(t, source.checkSelection([]ModelInfo{gpt4o}, summarization))
	})

	t.Run("some reuse is enough: a model added to the ones the run had runs live", func(t *testing.T) {
		assert.NoError(t, source.checkSelection([]ModelInfo{newModel, lunaNone}, summarization))
	})

	t.Run("one task it stashed among several", func(t *testing.T) {
		assert.NoError(t, source.checkSelection([]ModelInfo{lunaNone}, []string{prompts.Dictation, prompts.Summarization}))
	})

	t.Run("-reasoning-effort forgotten", func(t *testing.T) {
		err := source.checkSelection([]ModelInfo{lunaDefault}, summarization)

		require.Error(t, err)
		assert.Contains(t, err.Error(), "nothing to reuse")
		assert.Contains(t, err.Error(), "gpt-6-luna (default)", "what this run asks for")
		assert.Contains(t, err.Error(), "gpt-6-luna (none)", "what the run stashed")
		assert.Contains(t, err.Error(), "-reasoning-effort")
	})

	t.Run("a model the run never had", func(t *testing.T) {
		assert.Error(t, source.checkSelection([]ModelInfo{newModel}, summarization))
	})

	t.Run("a task the run never ran", func(t *testing.T) {
		assert.Error(t, source.checkSelection([]ModelInfo{lunaNone}, []string{prompts.TextCorrection}))
	})
}

func TestListUpTo(t *testing.T) {
	assert.Equal(t, "nothing", listUpTo(nil, 3))
	assert.Equal(t, "a, b", listUpTo([]string{"a", "b"}, 3))
	assert.Equal(t, "a, b, c", listUpTo([]string{"a", "b", "c"}, 3))
	assert.Equal(t, "a, b and 2 more", listUpTo([]string{"a", "b", "c", "d"}, 2))
}
