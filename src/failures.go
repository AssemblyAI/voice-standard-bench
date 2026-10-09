package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Failure reasons runOne records: a completion that came back and parsed,
// but didn't match the task's required output shape; a completion call that
// never came back at all (non-2xx, timeout, network error, ...); or the
// judge call itself failing — a non-2xx response, a reply with no JSON object
// in it, or a reply whose JSON doesn't match the rubric (see JudgeError).
const (
	FailureInvalidSchema = "invalid_schema"
	FailureAPIError      = "api_error"
	FailureJudgeError    = "judge_error"
)

// FailureRecord is everything about one failed (model, task, record)
// attempt that's worth keeping: enough to find it again in the report
// (Model/Task/Bucket/RecordID), why it failed, the VERBATIM raw JSON request
// that was sent (CompletionResult.RawRequest — same for both failure modes,
// and set even though the call failed, so the exact request can be replayed
// to reproduce it), and — when there is one — the VERBATIM raw JSON response
// behind the failure: the whole ChatCompletion body (messages, usage,
// everything) for an invalid-schema failure, or the gateway's raw error body
// for an API failure that returned one (see RawAPIError in gateway.go for
// both). Neither Request nor Response is a field this tool extracted from
// the call — the point is to be able to inspect and reproduce exactly what
// was sent and what came back, not this tool's summary of either. Response
// is empty when there is nothing to save at all: an API failure that never
// got a response (timeout, network error, ...).
type FailureRecord struct {
	Model string
	// ReasoningEffort is the same label ModelInfo.ReasoningEffortLabel
	// records on the Sample this failure corresponds to ("default" when
	// -reasoning-effort wasn't set or unsupported) -- so Model+
	// ReasoningEffort+Task+Bucket+RecordID still uniquely finds this
	// failure's row in the report even when the same model was benchmarked
	// at more than one reasoning effort in the same run.
	ReasoningEffort string
	Task            string
	Bucket          string
	RecordID        string
	Reason          string
	Err             error
	Request         string
	Response        string
}

// failureMetaEntry is one line of meta.json. It deliberately excludes the
// raw request/response bodies — those live in OutputFile instead — so
// meta.json stays a small, single-glance index even across a run with
// thousands of failures, with the full detail only a file open away.
type failureMetaEntry struct {
	Model           string `json:"model"`
	ReasoningEffort string `json:"reasoning_effort"`
	Task            string `json:"task"`
	Bucket          string `json:"bucket"`
	RecordID        string `json:"record_id"`
	Reason          string `json:"reason"`
	Error           string `json:"error"`
	OutputFile      string `json:"output_file"` // relative to the run directory (FailureRecorder.RunDir)
}

// failureOutput is the full per-failure JSON file: the same identifying
// fields as its meta.json entry, plus the raw request and response
// meta.json omits. Both are json.RawMessage — embedded as real nested JSON
// values, not strings — so opening the file shows the gateway request and
// response exactly as they were sent/arrived (messages, usage, ...), not
// re-encoded as an escaped string blob one level down. Request lets a
// failure be replayed directly against the gateway to debug or reproduce it.
type failureOutput struct {
	Model           string          `json:"model"`
	ReasoningEffort string          `json:"reasoning_effort"`
	Task            string          `json:"task"`
	Bucket          string          `json:"bucket"`
	RecordID        string          `json:"record_id"`
	Reason          string          `json:"reason"`
	Error           string          `json:"error"`
	Request         json.RawMessage `json:"request,omitempty"`
	Response        json.RawMessage `json:"response,omitempty"`
}

// rawJSONField turns a raw request/response string into a json.RawMessage
// suitable for embedding verbatim as a nested JSON value. Empty input
// produces nil, which json's omitempty then drops the field entirely for. A
// non-empty value that somehow isn't valid JSON — defensive only; every real
// caller's raw string comes from an already-marshaled request or an
// already-decoded API response (see CompletionResult.RawRequest/RawResponse
// and RawAPIError) — is preserved anyway, re-encoded as a plain JSON string,
// so a failure is never silently dropped just because this safety check
// doesn't hold.
func rawJSONField(raw string) json.RawMessage {
	if raw == "" {
		return nil
	}

	if json.Valid([]byte(raw)) {
		return json.RawMessage(raw)
	}

	fallback, err := json.Marshal(raw)
	if err != nil {
		return nil
	}

	return fallback
}

// FailureRecorder collects every failed sample from one benchmark run into a
// single run directory: one JSON file per failure, plus a meta.json that
// indexes every failure with its reason and a path to that file. It is safe
// for concurrent use — every candidate model's goroutines report into the
// same recorder (see runOne, runModel).
type FailureRecorder struct {
	mu      sync.Mutex
	runDir  string
	entries []failureMetaEntry
	seq     int
}

// NewFailureRecorder creates errorsRoot (if it doesn't already exist yet)
// and, under it, a fresh directory unique to this run — via os.MkdirTemp, so
// two benchmark runs started in the same process or the same second never
// collide on, or race over, the same meta.json.
func NewFailureRecorder(errorsRoot string) (*FailureRecorder, error) {
	if err := os.MkdirAll(errorsRoot, 0o755); err != nil {
		return nil, fmt.Errorf("creating errors directory %s: %w", errorsRoot, err)
	}

	runDir, err := os.MkdirTemp(errorsRoot, time.Now().UTC().Format("20060102-150405")+"-")
	if err != nil {
		return nil, fmt.Errorf("creating run directory under %s: %w", errorsRoot, err)
	}

	return &FailureRecorder{runDir: runDir}, nil
}

// RunDir is the unique directory this recorder's meta.json and per-failure
// output files live under.
func (r *FailureRecorder) RunDir() string {
	return r.runDir
}

// Count returns how many failures have been recorded so far.
func (r *FailureRecorder) Count() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return len(r.entries)
}

// Record writes fr's raw response to its own JSON file under the run
// directory, then rewrites meta.json with an index entry naming it appended
// to every entry recorded so far. Rewriting meta.json on every call (rather
// than only once at the end of the run) means it reflects everything
// captured so far even if the benchmark run is killed or crashes partway
// through — failures are the exception, not the common case, across a
// benchmark run's samples, so the extra write is cheap.
func (r *FailureRecorder) Record(fr FailureRecord) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.seq++
	filename := sanitizeFilename(fmt.Sprintf("%04d-%s-%s-%s.json", r.seq, fr.Model, fr.Task, fr.RecordID))

	out := failureOutput{
		Model:           fr.Model,
		ReasoningEffort: fr.ReasoningEffort,
		Task:            fr.Task,
		Bucket:          fr.Bucket,
		RecordID:        fr.RecordID,
		Reason:          fr.Reason,
		Request:         rawJSONField(fr.Request),
		Response:        rawJSONField(fr.Response),
	}

	if fr.Err != nil {
		out.Error = fr.Err.Error()
	}

	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling failure output: %w", err)
	}

	if err := os.WriteFile(filepath.Join(r.runDir, filename), data, 0o644); err != nil {
		return fmt.Errorf("writing failure output: %w", err)
	}

	r.entries = append(r.entries, failureMetaEntry{
		Model:           out.Model,
		ReasoningEffort: out.ReasoningEffort,
		Task:            out.Task,
		Bucket:          out.Bucket,
		RecordID:        out.RecordID,
		Reason:          out.Reason,
		Error:           out.Error,
		OutputFile:      filename,
	})

	metaData, err := json.MarshalIndent(r.entries, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling meta.json: %w", err)
	}

	if err := os.WriteFile(filepath.Join(r.runDir, "meta.json"), metaData, 0o644); err != nil {
		return fmt.Errorf("writing meta.json: %w", err)
	}

	return nil
}

// sanitizeFilename replaces every character that isn't safe to use verbatim
// in a path segment with "_". It exists because model IDs routinely contain
// "/" (e.g. "openai/gpt-4o") — used verbatim, that would be read as a
// subdirectory and either fail outright or scatter failure output outside
// the run directory.
func sanitizeFilename(s string) string {
	var b strings.Builder

	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}

	return b.String()
}
