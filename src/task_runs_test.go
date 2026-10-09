package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sampleRun returns the Sample and CompletionResult a successful RunRecord for
// one summarization record would hand back -- the pair TaskRunStore.Save takes.
// The durations deliberately have sub-millisecond digits: a report would round
// them away, and the stash must not.
func sampleRun() (Sample, CompletionResult) {
	sample := Sample{
		Model:            "candidate-model",
		ReasoningEffort:  "default",
		Task:             "summarization",
		Bucket:           "30s",
		RecordID:         "rec-1",
		TotalLatency:     1234567891 * time.Nanosecond,
		TimeToFirstToken: 1234567891 * time.Nanosecond,
		PromptTokens:     11,
		CompletionTokens: 7,
	}

	result := CompletionResult{
		Text:        "a summary",
		RawRequest:  `{"model":"candidate-model","messages":[{"role":"user","content":"a call"}]}`,
		RawResponse: `{"id":"chatcmpl-1","choices":[{"message":{"content":"a summary"},"finish_reason":"stop"}]}`,
	}

	return sample, result
}

// readTaskRunFile decodes the task run file at path, failing the test if it
// can't. Shared across this package's test files.
func readTaskRunFile(t *testing.T, path string) taskRunFile {
	t.Helper()

	data, err := os.ReadFile(path)
	require.NoError(t, err)

	var file taskRunFile
	require.NoError(t, json.Unmarshal(data, &file))

	return file
}

// taskRunPaths lists every file store has written, in any task's directory; a
// store that hasn't written anything has no directory at all, which is an empty
// list rather than an error.
func taskRunPaths(t *testing.T, store *TaskRunStore) []string {
	t.Helper()

	var paths []string

	err := filepath.WalkDir(store.Dir(), func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() {
			paths = append(paths, path)
		}

		return nil
	})
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}

	require.NoError(t, err)

	return paths
}

// TestTaskRunStore_Save_writesOneFileUnderItsTask pins the layout the doc
// comments promise -- <run dir>/task_runs/<task>/<model>__<effort>__<bucket>__<record>.json,
// with the characters a file name can't carry made safe -- and that the file
// holds the run in full.
func TestTaskRunStore_Save_writesOneFileUnderItsTask(t *testing.T) {
	runDir := t.TempDir()
	store := NewTaskRunStore(runDir)

	sample, result := sampleRun()
	sample.Model = "openai/gpt-4o"                  // "/" must not become a subdirectory
	sample.ReasoningEffort = "low"                  //
	sample.RecordID = "aai_transcripts:30s:abc-123" // real record IDs carry ":"

	require.NoError(t, store.Save(sample, result))

	path := filepath.Join(runDir, "task_runs", "summarization", "openai_gpt-4o__low__30s__aai_transcripts_30s_abc-123.json")
	require.FileExists(t, path)
	assert.Equal(t, filepath.Join(runDir, "task_runs"), store.Dir())
	assert.Equal(t, 1, store.Count())

	got := readTaskRunFile(t, path)

	// Who it was: the original values, not their file-name-safe forms.
	assert.Equal(t, "openai/gpt-4o", got.Model)
	assert.Equal(t, "low", got.ReasoningEffort)
	assert.Equal(t, "summarization", got.Task)
	assert.Equal(t, "30s", got.Bucket)
	assert.Equal(t, "aai_transcripts:30s:abc-123", got.RecordID)

	// What it said, and the raw exchange behind it.
	assert.Equal(t, "a summary", got.Text)
	assert.JSONEq(t, result.RawRequest, string(got.Request))
	assert.JSONEq(t, result.RawResponse, string(got.Response))

	// What it measured: exact, not rounded to milliseconds the way a report
	// rounds -- and exact enough to give the very same Duration back.
	assert.Equal(t, sample.TotalLatency.Seconds(), got.TotalLatencyS)
	assert.Equal(t, sample.TimeToFirstToken.Seconds(), got.TimeToFirstTokenS)
	assert.Equal(t, sample.TotalLatency, time.Duration(math.Round(got.TotalLatencyS*1e9)))
	assert.Equal(t, 11, got.PromptTokens)
	assert.Equal(t, 7, got.CompletionTokens)
}

// TestTaskRunStore_Save_keepsTranscriptPunctuationReadable: transcripts are full
// of "&" (Q&A, AT&T), and encoding/json's default would write each one as a
// six-character unicode escape -- valid JSON, but a grep for the phrase would
// miss it.
func TestTaskRunStore_Save_keepsTranscriptPunctuationReadable(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()
	result.Text = "the Q&A with AT&T, <b>bold</b>"

	require.NoError(t, store.Save(sample, result))

	paths := taskRunPaths(t, store)
	require.Len(t, paths, 1)

	raw, err := os.ReadFile(paths[0])
	require.NoError(t, err)

	// What encoding/json's default would have written for the "&" and "<": a
	// backslash (code point 0x5c), "u", then the code point in four hex digits.
	backslash := string(rune(0x5c))

	assert.Contains(t, string(raw), "the Q&A with AT&T, <b>bold</b>")
	assert.NotContains(t, string(raw), backslash+"u0026")
	assert.NotContains(t, string(raw), backslash+"u003c")
	assert.Equal(t, result.Text, readTaskRunFile(t, paths[0]).Text, "and it still decodes to exactly what the model said")
}

// TestTaskRunStore_Save_omitsRawFieldsThatAreEmpty: a result with no raw
// request or response (a streamed one has no single response document -- see
// completionResultFromStream) still saves, as valid JSON without those keys.
func TestTaskRunStore_Save_omitsRawFieldsThatAreEmpty(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()
	result.RawRequest, result.RawResponse = "", ""

	require.NoError(t, store.Save(sample, result))

	paths := taskRunPaths(t, store)
	require.Len(t, paths, 1)

	raw, err := os.ReadFile(paths[0])
	require.NoError(t, err)

	var keys map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &keys))

	assert.Contains(t, keys, "text")
	assert.NotContains(t, keys, "request")
	assert.NotContains(t, keys, "response")
}

// TestTaskRunStore_Save_everyPartOfTheIdentityGetsItsOwnFile: two runs that
// differ in any one of model, reasoning effort, task, bucket, or record are two
// different runs and must not share, and so overwrite, a file.
func TestTaskRunStore_Save_everyPartOfTheIdentityGetsItsOwnFile(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	base, result := sampleRun()

	variants := []Sample{base}

	for _, change := range []func(*Sample){
		func(s *Sample) { s.Model = "other-model" },
		func(s *Sample) { s.ReasoningEffort = "high" },
		func(s *Sample) { s.Task = "text_correction" },
		func(s *Sample) { s.Bucket = "120s" },
		func(s *Sample) { s.RecordID = "rec-2" },
	} {
		variant := base
		change(&variant)
		variants = append(variants, variant)
	}

	// Each run says something different, so an overwrite would be visible below.
	outputFor := func(s Sample) string {
		return strings.Join([]string{s.Model, s.ReasoningEffort, s.Task, s.Bucket, s.RecordID}, "|")
	}

	for _, variant := range variants {
		result.Text = outputFor(variant)
		require.NoError(t, store.Save(variant, result))
	}

	paths := taskRunPaths(t, store)
	assert.Len(t, paths, len(variants), "no two runs may land on the same file")

	for _, path := range paths {
		file := readTaskRunFile(t, path)
		identity := Sample{Model: file.Model, ReasoningEffort: file.ReasoningEffort, Task: file.Task, Bucket: file.Bucket, RecordID: file.RecordID}
		assert.Equal(t, outputFor(identity), file.Text, "%s holds some other run's output", path)
	}
}

// TestTaskRunStore_Save_aRunSavedAgainReplacesItsFile: a run has one file, so
// saving it again -- say, after being run a second time -- replaces what was
// there rather than adding a second.
func TestTaskRunStore_Save_aRunSavedAgainReplacesItsFile(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()

	result.Text = "first attempt"
	require.NoError(t, store.Save(sample, result))

	result.Text = "second attempt"
	require.NoError(t, store.Save(sample, result))

	paths := taskRunPaths(t, store)
	require.Len(t, paths, 1)
	assert.Equal(t, "second attempt", readTaskRunFile(t, paths[0]).Text)
}

// TestTaskRunStore_createsItsDirectoryOnFirstSave: a pass that never gets an
// output to keep -- every call failed, say -- must not leave an empty task_runs
// directory in its run dir.
func TestTaskRunStore_createsItsDirectoryOnFirstSave(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())

	assert.NoDirExists(t, store.Dir())
	assert.Zero(t, store.Count())

	require.NoError(t, store.Save(sampleRun()))

	assert.DirExists(t, store.Dir())
}

// TestTaskRunStore_Save_leavesOnlyWholeReadableFiles: a save goes through a
// temporary file (see writeFileAtomic), so none may be left behind, and the
// result is as readable as every other file a run writes (os.CreateTemp alone
// would make it 0600).
func TestTaskRunStore_Save_leavesOnlyWholeReadableFiles(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()

	for i := range 3 {
		sample.RecordID = fmt.Sprintf("rec-%d", i)
		require.NoError(t, store.Save(sample, result))
	}

	entries, err := os.ReadDir(filepath.Join(store.Dir(), "summarization"))
	require.NoError(t, err)
	require.Len(t, entries, 3)

	for _, entry := range entries {
		assert.True(t, strings.HasSuffix(entry.Name(), ".json"), "%s is not a finished task run file", entry.Name())

		info, err := entry.Info()
		require.NoError(t, err)
		assert.Equal(t, os.FileMode(0o644), info.Mode().Perm())
	}
}

// TestTaskRunStore_Save_isSafeForConcurrentUse mirrors how a real run uses the
// store: every model's goroutines finishing runs at once. Meaningful under
// -race.
func TestTaskRunStore_Save_isSafeForConcurrentUse(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())

	const runs = 64

	var wg sync.WaitGroup

	for i := range runs {
		wg.Add(1)

		go func() {
			defer wg.Done()

			sample, result := sampleRun()
			sample.RecordID = fmt.Sprintf("rec-%d", i)
			result.Text = fmt.Sprintf("output %d", i)

			assert.NoError(t, store.Save(sample, result))
		}()
	}

	wg.Wait()

	assert.Equal(t, runs, store.Count())

	paths := taskRunPaths(t, store)
	require.Len(t, paths, runs)

	for _, path := range paths {
		file := readTaskRunFile(t, path)
		assert.Equal(t, strings.Replace(file.RecordID, "rec-", "output ", 1), file.Text, "%s holds some other run's output", path)
	}
}

// TestTaskRunStore_Save_reportsAnErrorWhenItsDirectoryCannotBeCreated: a store
// that can't write says so -- Runner decides what to do about that (see
// TestRunnerRunStillJudgesWhenStashingFails) -- and doesn't count the attempt.
func TestTaskRunStore_Save_reportsAnErrorWhenItsDirectoryCannotBeCreated(t *testing.T) {
	// A regular file where the run directory should be: nothing can be created
	// beneath it, whoever is running the test.
	notADir := filepath.Join(t.TempDir(), "file")
	require.NoError(t, os.WriteFile(notADir, nil, 0o644))

	store := NewTaskRunStore(notADir)

	assert.Error(t, store.Save(sampleRun()))
	assert.Zero(t, store.Count(), "a save that failed must not be counted")
}

// TestWriteFileAtomic_removesItsTemporaryFileWhenItCannotFinish: the rename is
// the step that publishes a file, and when it fails nothing may be left behind
// in the directory a reader will look in.
func TestWriteFileAtomic_removesItsTemporaryFileWhenItCannotFinish(t *testing.T) {
	dir := t.TempDir()

	// The destination is an existing directory, so renaming a file onto it fails.
	dest := filepath.Join(dir, "dest")
	require.NoError(t, os.Mkdir(dest, 0o755))

	assert.Error(t, writeFileAtomic(dest, []byte("data")))

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	require.Len(t, entries, 1, "the temporary file must have been cleaned up")
	assert.Equal(t, "dest", entries[0].Name())
}

// ---------------------------------------------------------------------------
// Load -- reading a stashed run back, which is what a replay does
// ---------------------------------------------------------------------------

// loadRun is store.Load for the run sample identifies.
func loadRun(store *TaskRunStore, s Sample) (Sample, CompletionResult, bool, error) {
	return store.Load(s.Model, s.ReasoningEffort, s.Task, s.Bucket, s.RecordID)
}

// pathOf is where store keeps the run sample identifies.
func pathOf(store *TaskRunStore, s Sample) string {
	return store.path(s.Model, s.ReasoningEffort, s.Task, s.Bucket, s.RecordID)
}

// TestTaskRunStore_Load_givesBackWhatSaveWasGiven: a replay hands the judge, and
// the report, exactly what the original call produced -- the Sample with its
// identity and measurements, the CompletionResult with its text and raw
// exchange -- not an approximation of it.
func TestTaskRunStore_Load_givesBackWhatSaveWasGiven(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()
	sample.Model = "openai/gpt-4o"                  // names are lossy; what is stored inside isn't
	sample.RecordID = "aai_transcripts:30s:abc-123" //

	require.NoError(t, store.Save(sample, result))

	gotSample, gotResult, ok, err := loadRun(store, sample)
	require.NoError(t, err)
	require.True(t, ok)

	// The Sample: identity, and measurements exact to the nanosecond (sampleRun's
	// durations have digits a report would round away).
	assert.Equal(t, sample, gotSample)

	// The CompletionResult: the output, the raw exchange, and the same
	// measurements a real call's result carries.
	assert.Equal(t, result.Text, gotResult.Text)
	assert.JSONEq(t, result.RawRequest, gotResult.RawRequest)
	assert.JSONEq(t, result.RawResponse, gotResult.RawResponse)
	assert.Equal(t, sample.TotalLatency, gotResult.TotalLatency)
	assert.Equal(t, sample.TimeToFirstToken, gotResult.TimeToFirstToken)
	assert.Equal(t, sample.PromptTokens, gotResult.PromptTokens)
	assert.Equal(t, sample.CompletionTokens, gotResult.CompletionTokens)

	assert.Equal(t, 1, store.Reused(sample.Task))
}

// TestTaskRunStore_Load_aReplayedRunIsSavedBackAsTheSameBytes: a replay stashes
// what it reused into its own run's directory (so every run's directory is
// complete, and a replay can be replayed). That copy must not drift -- the same
// bytes, however many times it is replayed.
func TestTaskRunStore_Load_aReplayedRunIsSavedBackAsTheSameBytes(t *testing.T) {
	first, second := NewTaskRunStore(t.TempDir()), NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()
	result.Text = "the Q&A with AT&T, <b>bold</b>"

	require.NoError(t, first.Save(sample, result))

	gotSample, gotResult, ok, err := loadRun(first, sample)
	require.NoError(t, err)
	require.True(t, ok)
	require.NoError(t, second.Save(gotSample, gotResult))

	original, err := os.ReadFile(pathOf(first, sample))
	require.NoError(t, err)

	copied, err := os.ReadFile(pathOf(second, sample))
	require.NoError(t, err)

	assert.Equal(t, string(original), string(copied))
}

// TestTaskRunStore_Load_nothingStashedIsNotAnError: a run that isn't there --
// its call failed, or never happened -- is the ordinary case for a replay, and
// means "make the call", not "something is wrong".
func TestTaskRunStore_Load_nothingStashedIsNotAnError(t *testing.T) {
	sample, result := sampleRun()

	t.Run("the store's directory doesn't exist", func(t *testing.T) {
		store := NewTaskRunStore(filepath.Join(t.TempDir(), "never-made"))

		_, _, ok, err := loadRun(store, sample)

		assert.NoError(t, err)
		assert.False(t, ok)
	})

	t.Run("it holds other runs but not this one", func(t *testing.T) {
		store := NewTaskRunStore(t.TempDir())
		other := sample
		other.RecordID = "rec-2"
		require.NoError(t, store.Save(other, result))

		_, _, ok, err := loadRun(store, sample)

		assert.NoError(t, err)
		assert.False(t, ok)
		assert.Zero(t, store.Reused(sample.Task), "a miss is not a reuse")
	})
}

// TestTaskRunStore_Load_refusesAFileThatHoldsADifferentRun: a file's name is a
// lossy rendering of its run (see path), so two runs can land on one name. The
// run the file really holds is the one it says it does -- using it for the other
// would score the wrong output.
func TestTaskRunStore_Load_refusesAFileThatHoldsADifferentRun(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	saved, result := sampleRun()
	asked := saved
	asked.RecordID = "rec-2"

	require.NoError(t, store.Save(saved, result))

	// What a collision would leave: the first run's file at the second run's path.
	data, err := os.ReadFile(pathOf(store, saved))
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(pathOf(store, asked), data, 0o644))

	_, _, ok, err := loadRun(store, asked)

	assert.Error(t, err)
	assert.False(t, ok)
	assert.Zero(t, store.Reused(asked.Task))
}

func TestTaskRunStore_Load_reportsAFileItCannotRead(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()
	require.NoError(t, store.Save(sample, result))

	// A run killed halfway through a write by something other than Save, say.
	require.NoError(t, os.WriteFile(pathOf(store, sample), []byte(`{"model": "candidate-mo`), 0o644))

	_, _, ok, err := loadRun(store, sample)

	assert.Error(t, err)
	assert.False(t, ok)
}

// TestTaskRunStore_Reused_countsHitsPerTask: what a replay logs per task -- how
// much of the pass came out of the stash.
func TestTaskRunStore_Reused_countsHitsPerTask(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()

	var runs []Sample

	for _, task := range []string{"summarization", "summarization", "text_correction"} {
		run := sample
		run.Task = task
		run.RecordID = fmt.Sprintf("rec-%d", len(runs))
		runs = append(runs, run)

		require.NoError(t, store.Save(run, result))
	}

	for _, run := range runs {
		_, _, ok, err := loadRun(store, run)
		require.NoError(t, err)
		require.True(t, ok)
	}

	assert.Equal(t, 2, store.Reused("summarization"))
	assert.Equal(t, 1, store.Reused("text_correction"))
	assert.Zero(t, store.Reused("dictation"))
}

// TestTaskRunStore_Stashed_countsFinishedFilesOnly: what is on disk, whoever put
// it there -- and not the temporary file a Save killed partway leaves behind.
func TestTaskRunStore_Stashed_countsFinishedFilesOnly(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	assert.Zero(t, store.Stashed(), "a directory that isn't there holds nothing")

	sample, result := sampleRun()

	for i := range 2 {
		sample.RecordID = fmt.Sprintf("rec-%d", i)
		require.NoError(t, store.Save(sample, result))
	}

	dir := filepath.Join(store.Dir(), "summarization")
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".tmp-123456"), []byte("half a file"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("not a run"), 0o644))

	assert.Equal(t, 2, store.Stashed())

	// ...and a store opened on the same directory later sees the same two.
	assert.Equal(t, 2, NewTaskRunStore(filepath.Dir(store.Dir())).Stashed())
}

// TestTaskRunFile_systemPrompt reads the task prompt a stashed run was sent --
// what -replay compares with the prompt in this build, to refuse outputs made
// with another.
func TestTaskRunFile_systemPrompt(t *testing.T) {
	tests := []struct {
		name    string
		request string
		want    string
		wantErr bool
	}{
		{
			name:    "the system message's text",
			request: `{"model":"m","messages":[{"role":"system","content":"You summarize calls."},{"role":"user","content":"a call"}]}`,
			want:    "You summarize calls.",
		},
		{
			name:    "a request with no system message",
			request: `{"model":"m","messages":[{"role":"user","content":"a call"}]}`,
			wantErr: true,
		},
		{
			name:    "a system message whose content is not plain text",
			request: `{"model":"m","messages":[{"role":"system","content":[{"type":"text","text":"x"}]}]}`,
			wantErr: true,
		},
		{
			name:    "no request at all (an empty one is left out of a stashed file)",
			request: ``,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := taskRunFile{Request: json.RawMessage(tt.request)}.systemPrompt()

			if tt.wantErr {
				assert.Error(t, err)

				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestTaskRunStore_anyRun: one of a task's stashed runs, for what is true of all
// of them; nothing, without error, when the task has none.
func TestTaskRunStore_anyRun(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())

	_, ok, err := store.anyRun("summarization")
	require.NoError(t, err)
	assert.False(t, ok, "nothing stashed yet")

	sample, result := sampleRun()
	require.NoError(t, store.Save(sample, result))

	// A leftover temporary file sorts first (".tmp" before "c") and must be skipped.
	require.NoError(t, os.WriteFile(filepath.Join(store.Dir(), "summarization", ".tmp-1"), []byte("half"), 0o644))

	file, ok, err := store.anyRun("summarization")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, "rec-1", file.RecordID)

	_, ok, err = store.anyRun("text_correction")
	require.NoError(t, err)
	assert.False(t, ok, "a different task has none")

	require.NoError(t, os.WriteFile(pathOf(store, sample), []byte("not json"), 0o644))

	_, _, err = store.anyRun("summarization")
	assert.Error(t, err, "a stashed run that can't be read is an error, not a silent miss")
}

// TestTaskRunStore_hasRunsOf: whether a stash holds any run of a task by a model
// at a reasoning effort -- going by the file names, for a whole stash at once.
func TestTaskRunStore_hasRunsOf(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()
	sample.Model = "openai/gpt-4o"
	sample.ReasoningEffort = "low"

	require.NoError(t, store.Save(sample, result))
	require.NoError(t, os.WriteFile(filepath.Join(store.Dir(), "summarization", ".tmp-1"), []byte("half"), 0o644))

	assert.True(t, store.hasRunsOf("summarization", "openai/gpt-4o", "low"))
	assert.False(t, store.hasRunsOf("summarization", "openai/gpt-4o", "high"), "another reasoning effort")
	assert.False(t, store.hasRunsOf("summarization", "openai/gpt-4", "low"), "a model whose ID merely starts the same")
	assert.False(t, store.hasRunsOf("text_correction", "openai/gpt-4o", "low"), "another task")
	assert.False(t, NewTaskRunStore(t.TempDir()).hasRunsOf("summarization", "openai/gpt-4o", "low"), "an empty stash")
}

// TestTaskRunStore_stashedModels: what the error for a replay with nothing to
// reuse shows of a stash -- each model and effort once, sorted, across tasks.
func TestTaskRunStore_stashedModels(t *testing.T) {
	store := NewTaskRunStore(t.TempDir())
	sample, result := sampleRun()

	for _, run := range []struct{ task, model, effort, record string }{
		{"summarization", "model-b", "default", "rec-1"},
		{"summarization", "model-b", "default", "rec-2"},
		{"text_correction", "model-b", "default", "rec-1"},
		{"text_correction", "model-a", "high", "rec-1"},
	} {
		sample.Task, sample.Model, sample.ReasoningEffort, sample.RecordID = run.task, run.model, run.effort, run.record
		require.NoError(t, store.Save(sample, result))
	}

	assert.Equal(t, []string{"model-a (high)", "model-b (default)"}, store.stashedModels())
	assert.Empty(t, NewTaskRunStore(t.TempDir()).stashedModels())
}
