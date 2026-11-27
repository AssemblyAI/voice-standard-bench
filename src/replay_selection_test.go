package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReplaySelectedRecords(t *testing.T) {
	tests := []struct {
		name, bucket, id string
		ttftOnly         bool
		wantError        bool
	}{
		{"matching record", "30s", "r1", false, false},
		{"disjoint bucket", "120s", "r120", false, true},
		{"changed record IDs", "30s", "changed", false, true},
		{"TTFT-only matching record", "30s", "r1", true, false},
		{"TTFT stash outside streaming buckets", "900s", "r900", true, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			writeFrameworkFixtures(t)
			writeFile(t, filepath.Join(normalizedRoot, "dictation", "synthetic.jsonl"),
				dictationRecordLine(tc.id, tc.bucket, "unused", "hey comma world period", "Hey, world.", allDictationMetrics))
			standard := NewTaskRunStore(t.TempDir())
			ttft := NewTaskRunStore(t.TempDir())
			store := standard
			if tc.ttftOnly {
				store = ttft
			}
			bucket, id := "30s", "r1"
			if tc.bucket == "900s" {
				bucket, id = tc.bucket, tc.id
			}
			require.NoError(t, store.Save(Sample{Model: "M", ReasoningEffort: "default", Task: "dictation", Bucket: bucket, RecordID: id}, CompletionResult{Text: "Hey, world."}))
			source := &replaySource{name: "earlier", standard: standard, ttft: ttft}
			// The CLI selection is already present in the dataset here, making
			// this assertion runnable against the original preflight too.
			err := source.checkSelection([]ModelInfo{{ID: "M"}}, []string{"dictation"}, nil, 0)
			assert.Equal(t, tc.wantError, err != nil)
			assert.Zero(t, standard.Reused("dictation"))
			assert.Zero(t, ttft.Reused("dictation"))
		})
	}
}

func TestReplaySampledIDs(t *testing.T) {
	writeFrameworkFixtures(t)
	task, err := NewDictationTask(nil)
	require.NoError(t, err)
	lines := dictationRecordLine("a", "30s", "unused", "hey comma world period", "Hey, world.", allDictationMetrics) +
		dictationRecordLine("b", "30s", "unused", "hey comma world period", "Hey, world.", allDictationMetrics)
	writeFile(t, filepath.Join(normalizedRoot, "dictation", "synthetic.jsonl"), lines)
	runner := NewRunner[DictationRecord](task, RunConfig{MaxSamplesPerBucket: 1})
	selected, err := runner.LoadAndSample()
	require.NoError(t, err)
	require.Len(t, selected, 1)
	unselected := "a"
	if selected[0].ID == unselected {
		unselected = "b"
	}
	store := NewTaskRunStore(t.TempDir())
	require.NoError(t, store.Save(Sample{Model: "M", ReasoningEffort: "default", Task: "dictation", Bucket: "30s", RecordID: unselected}, CompletionResult{Text: "Hey, world."}))
	source := &replaySource{name: "earlier", standard: store, ttft: NewTaskRunStore(t.TempDir())}
	assert.Error(t, source.checkSelection([]ModelInfo{{ID: "M"}}, []string{"dictation"}, nil, 1))
}

func TestReplaySelectionKeepsMixedAndAllReuse(t *testing.T) {
	for _, mixed := range []bool{false, true} {
		t.Run(map[bool]string{false: "all reuse", true: "mixed"}[mixed], func(t *testing.T) {
			writeFrameworkFixtures(t)
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				writeChatCompletion(w, "Hey, world.", 10, 5)
			}))
			defer server.Close()
			store := NewTaskRunStore(t.TempDir())
			require.NoError(t, store.Save(Sample{Model: "M", ReasoningEffort: "default", Task: "dictation", Bucket: "30s", RecordID: "r1"}, CompletionResult{Text: "Hey, world."}))
			if mixed {
				f, err := os.OpenFile(filepath.Join(normalizedRoot, "dictation", "synthetic.jsonl"), os.O_APPEND|os.O_WRONLY, 0600)
				require.NoError(t, err)
				_, err = f.WriteString(dictationRecordLine("new", "120s", "unused", "hey comma world period", "Hey, world.", allDictationMetrics))
				require.NoError(t, err)
				require.NoError(t, f.Close())
			}
			source := &replaySource{name: "earlier", standard: store, ttft: NewTaskRunStore(t.TempDir())}
			require.NoError(t, source.checkSelection([]ModelInfo{{ID: "M"}}, []string{"dictation"}, nil, 0))
			assert.Error(t, source.checkSelection([]ModelInfo{{ID: "M2"}}, []string{"dictation"}, nil, 0))
			task, err := NewDictationTask(NewGateway(server.URL, "offline-fake-token", time.Second))
			require.NoError(t, err)
			runner := NewRunner[DictationRecord](task, RunConfig{Models: []ModelInfo{{ID: "M"}}, JudgeModels: []string{"J1"}, Replay: store})
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			_, _, err = runner.RunBenchmark(ctx)
			require.NoError(t, err)
			want := int64(0)
			if mixed {
				want = 1
			}
			assert.Equal(t, want, calls.Load())
			assert.Equal(t, 1, store.Reused("dictation"))
		})
	}
}
