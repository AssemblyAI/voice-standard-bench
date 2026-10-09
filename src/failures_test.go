package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewFailureRecorder_createsAUniqueRunDirUnderErrorsRoot(t *testing.T) {
	root := t.TempDir()
	errorsRoot := filepath.Join(root, "errors")

	first, err := NewFailureRecorder(errorsRoot)
	require.NoError(t, err)

	second, err := NewFailureRecorder(errorsRoot)
	require.NoError(t, err)

	assert.NotEqual(t, first.RunDir(), second.RunDir(), "two runs must never share a run directory")

	for _, dir := range []string{first.RunDir(), second.RunDir()} {
		info, err := os.Stat(dir)
		require.NoError(t, err)
		assert.True(t, info.IsDir())
		assert.Equal(t, errorsRoot, filepath.Dir(dir), "each run directory must live directly under the errors root")
	}
}

func TestFailureRecorder_Record_writesOutputFileAndMeta(t *testing.T) {
	root := t.TempDir()
	rec, err := NewFailureRecorder(filepath.Join(root, "errors"))
	require.NoError(t, err)

	err = rec.Record(FailureRecord{
		Model:    "candidate-model",
		Task:     "summarization",
		Bucket:   "30s",
		RecordID: "rec-1",
		Reason:   FailureAPIError,
		Err:      assertError("gateway returned 429"),
	})
	require.NoError(t, err)

	assert.Equal(t, 1, rec.Count())

	metaBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), "meta.json"))
	require.NoError(t, err)

	var meta []failureMetaEntry
	require.NoError(t, json.Unmarshal(metaBytes, &meta))
	require.Len(t, meta, 1)

	entry := meta[0]
	assert.Equal(t, "candidate-model", entry.Model)
	assert.Equal(t, "summarization", entry.Task)
	assert.Equal(t, "30s", entry.Bucket)
	assert.Equal(t, "rec-1", entry.RecordID)
	assert.Equal(t, FailureAPIError, entry.Reason)
	assert.Equal(t, "gateway returned 429", entry.Error)
	assert.NotEmpty(t, entry.OutputFile)

	outputBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), entry.OutputFile))
	require.NoError(t, err, "meta.json's output_file must actually exist under the run directory")

	var output failureOutput
	require.NoError(t, json.Unmarshal(outputBytes, &output))
	assert.Equal(t, "candidate-model", output.Model)
	assert.Equal(t, "gateway returned 429", output.Error)
}

func TestFailureRecorder_Record_includesResponseInOutputFileButNotMeta(t *testing.T) {
	root := t.TempDir()
	rec, err := NewFailureRecorder(filepath.Join(root, "errors"))
	require.NoError(t, err)

	const rawResponse = `{"participants": ["not the right shape"]}`

	require.NoError(t, rec.Record(FailureRecord{
		Model:    "candidate-model",
		Task:     "information_extraction",
		RecordID: "rec-1",
		Reason:   FailureInvalidSchema,
		Err:      assertError("missing required key(s): action_items, decisions, entities"),
		Response: rawResponse,
	}))

	metaBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), "meta.json"))
	require.NoError(t, err)
	assert.NotContains(t, string(metaBytes), rawResponse,
		"meta.json must stay a small index -- the raw response belongs only in the per-failure output file")

	var meta []failureMetaEntry
	require.NoError(t, json.Unmarshal(metaBytes, &meta))
	require.Len(t, meta, 1)

	outputBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), meta[0].OutputFile))
	require.NoError(t, err)

	var output failureOutput
	require.NoError(t, json.Unmarshal(outputBytes, &output))
	assert.JSONEq(t, rawResponse, string(output.Response),
		"Response is embedded as a real nested JSON value, so it round-trips to the same JSON, not necessarily the same bytes")
}

// TestFailureRecorder_Record_includesRequestInOutputFileButNotMeta mirrors
// the Response coverage above for Request: same "small index vs full detail
// in its own file" contract, same embed-as-real-JSON handling.
func TestFailureRecorder_Record_includesRequestInOutputFileButNotMeta(t *testing.T) {
	root := t.TempDir()
	rec, err := NewFailureRecorder(filepath.Join(root, "errors"))
	require.NoError(t, err)

	const rawRequest = `{"model":"candidate-model","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"}]}`

	require.NoError(t, rec.Record(FailureRecord{
		Model:    "candidate-model",
		Task:     "summarization",
		RecordID: "rec-1",
		Reason:   FailureAPIError,
		Err:      assertError("boom"),
		Request:  rawRequest,
	}))

	metaBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), "meta.json"))
	require.NoError(t, err)
	assert.NotContains(t, string(metaBytes), rawRequest,
		"meta.json must stay a small index -- the raw request belongs only in the per-failure output file")

	var meta []failureMetaEntry
	require.NoError(t, json.Unmarshal(metaBytes, &meta))
	require.Len(t, meta, 1)

	outputBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), meta[0].OutputFile))
	require.NoError(t, err)

	var output failureOutput
	require.NoError(t, json.Unmarshal(outputBytes, &output))
	assert.JSONEq(t, rawRequest, string(output.Request),
		"Request is embedded as a real nested JSON value, so it round-trips to the same JSON, not necessarily the same bytes")
}

func TestFailureRecorder_Record_accumulatesEveryFailureInOrder(t *testing.T) {
	root := t.TempDir()
	rec, err := NewFailureRecorder(filepath.Join(root, "errors"))
	require.NoError(t, err)

	for i, recordID := range []string{"rec-1", "rec-2", "rec-3"} {
		require.NoError(t, rec.Record(FailureRecord{
			Model:    "candidate-model",
			Task:     "summarization",
			RecordID: recordID,
			Reason:   FailureAPIError,
			Err:      assertError("boom"),
		}), "failure %d", i)
	}

	assert.Equal(t, 3, rec.Count())

	metaBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), "meta.json"))
	require.NoError(t, err)

	var meta []failureMetaEntry
	require.NoError(t, json.Unmarshal(metaBytes, &meta))
	require.Len(t, meta, 3)

	var ids []string
	for _, e := range meta {
		ids = append(ids, e.RecordID)
	}
	assert.Equal(t, []string{"rec-1", "rec-2", "rec-3"}, ids)
}

// TestFailureRecorder_Record_sanitizesModelIDsWithSlashes locks in that a
// model ID shaped like a path ("openai/gpt-4o", the real-world convention
// for gateway-routed models) can't smuggle a "/" into the output filename --
// which os.WriteFile would otherwise read as a subdirectory that doesn't
// exist under the run directory, and the write would fail outright.
func TestFailureRecorder_Record_sanitizesModelIDsWithSlashes(t *testing.T) {
	root := t.TempDir()
	rec, err := NewFailureRecorder(filepath.Join(root, "errors"))
	require.NoError(t, err)

	err = rec.Record(FailureRecord{
		Model:    "openai/gpt-4o",
		Task:     "summarization",
		RecordID: "rec-1",
		Reason:   FailureAPIError,
		Err:      assertError("boom"),
	})
	require.NoError(t, err)

	metaBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), "meta.json"))
	require.NoError(t, err)

	var meta []failureMetaEntry
	require.NoError(t, json.Unmarshal(metaBytes, &meta))
	require.Len(t, meta, 1)

	assert.NotContains(t, meta[0].OutputFile, "/", "the output filename itself must not contain a path separator")

	_, err = os.ReadFile(filepath.Join(rec.RunDir(), meta[0].OutputFile))
	require.NoError(t, err, "the sanitized filename must resolve to a real file directly under the run directory")
}

// TestFailureRecorder_Record_concurrentUseIsSafe exercises Record from many
// goroutines at once -- the shape every real call site uses it in (see
// runOne, invoked from many per-model goroutines sharing one recorder).
// Run with `go test -race` to make the guarantee meaningful; without it, this only
// proves the entry count, not the absence of a data race.
func TestFailureRecorder_Record_concurrentUseIsSafe(t *testing.T) {
	root := t.TempDir()
	rec, err := NewFailureRecorder(filepath.Join(root, "errors"))
	require.NoError(t, err)

	const n = 50

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			assert.NoError(t, rec.Record(FailureRecord{
				Model:    "candidate-model",
				Task:     "summarization",
				RecordID: "rec",
				Reason:   FailureAPIError,
				Err:      assertError("boom"),
			}))
		}(i)
	}

	wg.Wait()

	assert.Equal(t, n, rec.Count())

	metaBytes, err := os.ReadFile(filepath.Join(rec.RunDir(), "meta.json"))
	require.NoError(t, err)

	var meta []failureMetaEntry
	require.NoError(t, json.Unmarshal(metaBytes, &meta))
	assert.Len(t, meta, n)
}

type assertError string

func (e assertError) Error() string { return string(e) }
