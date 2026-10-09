package main

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestContextLengthExceeded(t *testing.T) {
	tests := []struct {
		name       string
		model      ModelInfo
		system     string
		transcript string
		want       bool
	}{
		{
			name:       "well within context length",
			model:      ModelInfo{ContextLength: 128000},
			transcript: strings.Repeat("word ", 100),
			want:       false,
		},
		{
			name:       "unreported context length (0) never blocks a request",
			model:      ModelInfo{ContextLength: 0},
			transcript: strings.Repeat("word ", 1000000),
			want:       false,
		},
		{
			name:       "a transcript whose estimated tokens exceed a small context length is blocked",
			model:      ModelInfo{ContextLength: 100},
			transcript: strings.Repeat("word ", 1000), // ~5000 chars -> ~1250 estimated tokens
			want:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, contextLengthExceeded(tt.model, tt.system, tt.transcript))
		})
	}
}

func TestTaskMaxTokens(t *testing.T) {
	tests := []struct {
		name       string
		task       string
		transcript string
		model      ModelInfo
		want       int
	}{
		{
			name:       "summarization is a small fixed budget regardless of transcript length",
			task:       prompts.Summarization,
			transcript: strings.Repeat("x", 100000),
			model:      ModelInfo{},
			want:       15000,
		},
		{
			name:       "extraction is a small fixed budget too",
			task:       prompts.InformationExtraction,
			transcript: "short",
			model:      ModelInfo{},
			want:       15000,
		},
		{
			name:       "correction scales with transcript length plus 20% headroom",
			task:       prompts.TextCorrection,
			transcript: strings.Repeat("x", 4000), // estimatedTokens = 4000/4+1 = 1001
			model:      ModelInfo{},
			want:       16265,
		},
		{
			name:       "clamped to the model's own advertised max completion tokens",
			task:       prompts.Summarization,
			transcript: "short",
			model:      ModelInfo{MaxCompletionTokens: 100},
			want:       100,
		},
		{
			name:       "a model that doesn't report a max completion token limit isn't clamped",
			task:       prompts.Summarization,
			transcript: "short",
			model:      ModelInfo{MaxCompletionTokens: 0},
			want:       15000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, taskMaxTokens(tt.task, tt.transcript, tt.model))
		})
	}
}

func TestFilterModels(t *testing.T) {
	all := []ModelInfo{{ID: "a"}, {ID: "b"}, {ID: "c"}}

	t.Run("empty allowlist returns every model, and reports it as not explicit", func(t *testing.T) {
		filtered, explicit := filterModels(all, "")
		assert.Equal(t, all, filtered)
		assert.False(t, explicit)
	})

	t.Run("a comma-separated allowlist filters to exactly those ids, in the source order", func(t *testing.T) {
		filtered, explicit := filterModels(all, "c, a")
		assert.True(t, explicit)
		assert.Equal(t, []ModelInfo{{ID: "a"}, {ID: "c"}}, filtered)
	})

	t.Run("an id not present in the source list is silently absent, not an error", func(t *testing.T) {
		filtered, explicit := filterModels(all, "a,nonexistent")
		assert.True(t, explicit)
		assert.Equal(t, []ModelInfo{{ID: "a"}}, filtered)
	})
}

func TestRemoveModel(t *testing.T) {
	models := []ModelInfo{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	assert.Equal(t, []ModelInfo{{ID: "a"}, {ID: "c"}}, removeModel(models, "b"))
	assert.Equal(t, models, removeModel(models, "not-present"))
}

// TestApplyReasoningEffort locks in the "only if the model advertises
// support" contract: RunCompletion/RunCompletionStream only ever see
// ModelInfo.ReasoningEffort set when GET /v1/models' supported_parameters
// actually listed reasoning_effort for that model -- never guessed, never
// applied just because the flag was passed.
func TestApplyReasoningEffort(t *testing.T) {
	models := []ModelInfo{
		{ID: "supports-it", SupportedParameters: []string{"max_tokens", "reasoning_effort"}},
		{ID: "no-support", SupportedParameters: []string{"max_tokens"}},
		{ID: "reports-no-params-at-all"},
	}

	got := applyReasoningEffort(models, "low")
	require.Len(t, got, 3)

	want := "low"
	assert.Equal(t, &want, got[0].ReasoningEffort, "a model whose supported_parameters lists reasoning_effort gets it set")
	assert.Nil(t, got[1].ReasoningEffort, "a model that doesn't advertise support is left unset, not sent the parameter anyway")
	assert.Nil(t, got[2].ReasoningEffort, "a model reporting no supported_parameters at all is treated as unsupported, not as unknown-so-allow")

	assert.Equal(t, "no-support", got[1].ID, "every other field must be preserved unchanged")
	assert.Nil(t, models[0].ReasoningEffort, "the input slice itself must never be mutated in place")
}

func TestParseTasks(t *testing.T) {
	tests := []struct {
		name    string
		csv     string
		want    []string
		wantErr bool
	}{
		{
			name: "all three tasks by name",
			csv:  "summarization,information_extraction,text_correction",
			want: []string{"summarization", "information_extraction", "text_correction"},
		},
		{
			name: "whitespace around names is trimmed",
			csv:  " summarization , text_correction ",
			want: []string{"summarization", "text_correction"},
		},
		{
			name:    "an unknown task name is an error naming the valid set, not a silent skip",
			csv:     "summarization,not_a_real_task",
			wantErr: true,
		},
		{
			name:    "empty string is an error, not zero tasks",
			csv:     "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseTasks(tt.csv)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestTaskInput(t *testing.T) {
	tests := []struct {
		name    string
		task    string
		record  TranscriptRecord
		want    string
		wantErr bool
	}{
		{
			name:   "summarization always uses Text, ignoring MissedText even when present",
			task:   prompts.Summarization,
			record: TranscriptRecord{Text: "the real transcript", MissedText: "a corrupted one"},
			want:   "the real transcript",
		},
		{
			name:   "information_extraction always uses Text too",
			task:   prompts.InformationExtraction,
			record: TranscriptRecord{Text: "the real transcript", MissedText: "a corrupted one"},
			want:   "the real transcript",
		},
		{
			name:   "text_correction uses MissedText, never the true Text",
			task:   prompts.TextCorrection,
			record: TranscriptRecord{Text: "the real transcript", MissedText: "a corrupted one"},
			want:   "a corrupted one",
		},
		{
			name:    "text_correction with no MissedText is an error, not a silent fallback to Text",
			task:    prompts.TextCorrection,
			record:  TranscriptRecord{Text: "the real transcript", MissedText: ""},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := taskInput(tt.task, tt.record)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// readFailureMeta reads and decodes runDir/meta.json, failing the test if it
// can't. Shared across this package's test files.
func readFailureMeta(t *testing.T, runDir string) []failureMetaEntry {
	t.Helper()

	data, err := os.ReadFile(filepath.Join(runDir, "meta.json"))
	require.NoError(t, err)

	var meta []failureMetaEntry
	require.NoError(t, json.Unmarshal(data, &meta))

	return meta
}

// readFailureOutput reads and decodes one failure's own output file (named by
// its meta.json entry's OutputFile) under runDir, failing the test if it can't.
func readFailureOutput(t *testing.T, runDir, outputFile string) failureOutput {
	t.Helper()

	data, err := os.ReadFile(filepath.Join(runDir, outputFile))
	require.NoError(t, err)

	var output failureOutput
	require.NoError(t, json.Unmarshal(data, &output))

	return output
}

// failureRequestBody is the shape this package's tests decode output.Request
// into: just enough of ChatCompletionNewParams's wire format to prove the
// exact request that was sent -- which model, and what the system/user
// messages actually said -- was saved, without needing to decode the whole
// SDK type.
type failureRequestBody struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
}

func readBody(t *testing.T, r *http.Request) string {
	t.Helper()

	body, err := io.ReadAll(r.Body)
	require.NoError(t, err)

	return string(body)
}
