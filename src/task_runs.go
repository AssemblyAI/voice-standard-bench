// The task-run stash: the output of every successful candidate call, written to
// disk as soon as it comes back -- before it is judged -- so the judge step can
// be repeated against the very same outputs without paying for the candidate
// prompts again: a later run reads them back (Load) and replays them in place of
// the calls. See TaskRunStore.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// taskRunsDirName is the directory, directly under a staged run directory (see
// NewRunDir), that a TaskRunStore writes into. Nothing that globs a run
// directory for its reports (report-*.json, ttft-report*.json -- see
// canonical.go and summary.go) looks inside it.
const taskRunsDirName = "task_runs"

// taskRunFile is the on-disk form of one task run -- one candidate call, for
// one (model, reasoning effort, task, bucket, record) -- as TaskRunStore.Save
// writes it. It carries everything Task.RunRecord handed back, so a judge step
// reading it later has the same CompletionResult, and the same measurements for
// the Sample, as one running right after the call did:
//
//   - which run it is: the five fields that identify a Sample everywhere else
//     (a report-*.json sample has the same names). They are in the body, not
//     only in the file name, because the name is derived from them lossily (see
//     TaskRunStore.path);
//   - what the model said: Text, the field every task's RunJudge scores;
//   - what the call measured, from the Sample (that is what a report is built
//     from), unrounded -- a report rounds these to milliseconds, a stash must
//     not;
//   - Request and Response: the verbatim gateway request and response (see
//     CompletionResult.RawRequest/RawResponse), embedded as nested JSON rather
//     than escaped strings, the way failureOutput does. Response also carries
//     what Text alone doesn't, e.g. the finish reason that says whether an
//     output was cut off.
type taskRunFile struct {
	Model             string          `json:"model"`
	ReasoningEffort   string          `json:"reasoning_effort"`
	Task              string          `json:"task"`
	Bucket            string          `json:"bucket"`
	RecordID          string          `json:"record_id"`
	Text              string          `json:"text"`
	TotalLatencyS     float64         `json:"total_latency_s"`
	TimeToFirstTokenS float64         `json:"time_to_first_token_s"`
	PromptTokens      int             `json:"prompt_tokens"`
	CompletionTokens  int             `json:"completion_tokens"`
	Request           json.RawMessage `json:"request,omitempty"`
	Response          json.RawMessage `json:"response,omitempty"`
}

// sampleAndResult rebuilds what Task.RunRecord (or RunTTFT) returned -- the
// inverse of what Save keeps of it. The latency fields come back exactly (see
// secondsToDuration), and every one that Save took from the Sample is set on the
// CompletionResult too, so the two agree, as they do coming out of a real call.
func (f taskRunFile) sampleAndResult() (Sample, CompletionResult) {
	total, firstToken := secondsToDuration(f.TotalLatencyS), secondsToDuration(f.TimeToFirstTokenS)

	sample := Sample{
		Model:            f.Model,
		ReasoningEffort:  f.ReasoningEffort,
		Task:             f.Task,
		Bucket:           f.Bucket,
		RecordID:         f.RecordID,
		TotalLatency:     total,
		TimeToFirstToken: firstToken,
		PromptTokens:     f.PromptTokens,
		CompletionTokens: f.CompletionTokens,
	}

	result := CompletionResult{
		Text:             f.Text,
		RawRequest:       string(f.Request),
		RawResponse:      string(f.Response),
		TotalLatency:     total,
		TimeToFirstToken: firstToken,
		PromptTokens:     f.PromptTokens,
		CompletionTokens: f.CompletionTokens,
	}

	return sample, result
}

// systemPrompt returns the text of the system message in the request this run
// was made with -- the task prompt as it was when the call was made.
func (f taskRunFile) systemPrompt() (string, error) {
	var request struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}

	if err := json.Unmarshal(f.Request, &request); err != nil {
		return "", fmt.Errorf("parsing the stashed request: %w", err)
	}

	for _, message := range request.Messages {
		if message.Role != "system" {
			continue
		}

		var text string
		if err := json.Unmarshal(message.Content, &text); err != nil {
			return "", fmt.Errorf("the stashed system message's content is not plain text: %w", err)
		}

		return text, nil
	}

	return "", errors.New("the stashed request has no system message")
}

// secondsToDuration turns the seconds a taskRunFile holds back into the Duration
// they came from. Duration.Seconds loses nothing a latency has to lose -- a
// float64 holds nanoseconds exactly up to about 100 days -- so rounding to the
// nearest nanosecond gives back the very same value.
func secondsToDuration(seconds float64) time.Duration {
	return time.Duration(math.Round(seconds * float64(time.Second)))
}

// TaskRunStore keeps each successful task run in its own JSON file under one
// directory: <dir>/<task>/<model>__<reasoning effort>__<bucket>__<record>.json.
// It is what lets a benchmark's two halves be paid for separately: every model
// call costs money, and the judge step is the half that gets reworked (a new
// rubric, a new judge model, a judge that was down) -- so redoing the judging
// shouldn't mean paying for the model calls again. Save writes a run as it
// finishes; Load reads one back, which is what lets a later run replay an earlier
// one (see RunConfig.Replay). A run is found by the key every report row is
// already merged by -- (model, reasoning effort, task, bucket) -- plus the record.
//
// One file per run, rather than one growing file, because runs finish
// concurrently and constantly all through a benchmark: each is written once,
// whole, by whoever finished it (see Save), so there is no shared file to
// rewrite or lock, and a run killed partway through keeps every file it had
// already finished. A failed task run has no output to keep, so it never
// reaches the stash; those are recorded by FailureRecorder as before.
//
// Each pass has a store of its own: the standard pass's outputs go under its
// run directory in results/, the TTFT pass's under the run directory of the
// same name in results_ttft/ (see runFrameworkBenchmarkPass).
//
// It is safe for concurrent use: every Save writes a different file, and Load
// only reads.
type TaskRunStore struct {
	dir   string
	saved atomic.Int64

	mu     sync.Mutex
	reused map[string]int // by task: how many Loads found a stashed run
}

// NewTaskRunStore returns a store writing under runDir/task_runs -- a staged
// run directory, so the outputs sit beside the report scored from them. The
// directory itself is created on the first Save, so a run that produces no
// output leaves none behind.
func NewTaskRunStore(runDir string) *TaskRunStore {
	return &TaskRunStore{dir: filepath.Join(runDir, taskRunsDirName), reused: map[string]int{}}
}

// Dir is the directory this store reads and writes under.
func (s *TaskRunStore) Dir() string {
	return s.dir
}

// Count returns how many task runs this store has saved so far.
func (s *TaskRunStore) Count() int {
	return int(s.saved.Load())
}

// Reused returns how many times Load has found a stashed run of task -- how much
// of a replaying pass came out of the stash rather than from a live call.
func (s *TaskRunStore) Reused(task string) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.reused[task]
}

// Stashed returns how many task runs are on disk under the store's directory,
// whoever saved them -- this process or an earlier one. A directory that doesn't
// exist has none.
func (s *TaskRunStore) Stashed() int {
	var n int

	_ = filepath.WalkDir(s.dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // e.g. no such directory: nothing stashed
		}

		if !d.IsDir() && isTaskRunFileName(d.Name()) {
			n++
		}

		return nil
	})

	return n
}

// hasRunsOf reports whether the store holds any run of task by model at
// reasoningEffort, for any bucket and record. It goes by the start of the file
// names (see path) rather than opening them, so it can answer for a whole stash
// cheaply -- and, the names being lossy, can in principle say yes for a model whose
// ID sanitizes the same as another's, which is harmless for what it is used for:
// telling a replay with nothing to reuse from one with something.
func (s *TaskRunStore) hasRunsOf(task, model, reasoningEffort string) bool {
	entries, err := os.ReadDir(filepath.Join(s.dir, sanitizeFilename(task)))
	if err != nil {
		return false
	}

	prefix := sanitizeFilename(model + "__" + reasoningEffort + "__")

	for _, entry := range entries {
		if !entry.IsDir() && isTaskRunFileName(entry.Name()) && strings.HasPrefix(entry.Name(), prefix) {
			return true
		}
	}

	return false
}

// stashedModels lists the "model (reasoning effort)" pairs the store holds runs
// of, for telling a person what a stash contains. They are read off the file
// names, so a model ID appears as its file-name-safe form.
func (s *TaskRunStore) stashedModels() []string {
	seen := map[string]bool{}

	taskDirs, _ := os.ReadDir(s.dir)
	for _, taskDir := range taskDirs {
		if !taskDir.IsDir() {
			continue
		}

		entries, _ := os.ReadDir(filepath.Join(s.dir, taskDir.Name()))
		for _, entry := range entries {
			if entry.IsDir() || !isTaskRunFileName(entry.Name()) {
				continue
			}

			if model, rest, ok := strings.Cut(entry.Name(), "__"); ok {
				effort, _, _ := strings.Cut(rest, "__")
				seen[model+" ("+effort+")"] = true
			}
		}
	}

	pairs := make([]string, 0, len(seen))
	for pair := range seen {
		pairs = append(pairs, pair)
	}

	sort.Strings(pairs)

	return pairs
}

// isTaskRunFileName reports whether name is a finished task run file, as
// opposed to the temporary file a Save in progress -- or one that was killed
// partway -- leaves beside it (see writeFileAtomic).
func isTaskRunFileName(name string) bool {
	return strings.HasSuffix(name, ".json") && !strings.HasPrefix(name, ".")
}

// Save writes one successful task run: its identity and measurements from
// sample (RunRecord always populates Model/ReasoningEffort/Task/Bucket/RecordID,
// and the latency and token counts on success -- see the Task interface) and
// the output, with the raw request and response behind it, from the
// CompletionResult RunRecord returned alongside it. Saving the same run again
// replaces its file.
//
// The file is written to a temporary name in its own directory and renamed
// into place, so what is at the final path is always a whole file: a reader,
// or a run killed mid-write, never meets a half-written one.
func (s *TaskRunStore) Save(sample Sample, result CompletionResult) error {
	file := taskRunFile{
		Model:             sample.Model,
		ReasoningEffort:   sample.ReasoningEffort,
		Task:              sample.Task,
		Bucket:            sample.Bucket,
		RecordID:          sample.RecordID,
		Text:              result.Text,
		TotalLatencyS:     sample.TotalLatency.Seconds(),
		TimeToFirstTokenS: sample.TimeToFirstToken.Seconds(),
		PromptTokens:      sample.PromptTokens,
		CompletionTokens:  sample.CompletionTokens,
		Request:           rawJSONField(result.RawRequest),
		Response:          rawJSONField(result.RawResponse),
	}

	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false) // transcripts are full of "&" (Q&A, AT&T): keep them greppable, not a six-character unicode escape

	if err := enc.Encode(file); err != nil {
		return fmt.Errorf("marshaling task run: %w", err)
	}

	path := s.path(sample.Model, sample.ReasoningEffort, sample.Task, sample.Bucket, sample.RecordID)
	if err := writeFileAtomic(path, buf.Bytes()); err != nil {
		return fmt.Errorf("writing task run: %w", err)
	}

	s.saved.Add(1)

	return nil
}

// Load returns the task run stashed for one (model, reasoning effort, task,
// bucket, record): the Sample and CompletionResult as Save was given them, which
// is everything a judge call or a report row needs from the call that was made
// (CostUSD, which Save never took, is stamped afresh by Runner from the model's
// pricing). ok is false if nothing is stashed for it. That is not an error: a
// call that failed was never saved, and a run that never happened -- another
// model, a bigger sample -- was never there to be.
//
// An error means there IS a file, and it can't be trusted: unreadable, not valid
// JSON, or holding some other run than its name says (the name is lossy -- see
// path). The caller should treat that like ok=false and make the call.
func (s *TaskRunStore) Load(model, reasoningEffort, task, bucket, recordID string) (sample Sample, result CompletionResult, ok bool, err error) {
	path := s.path(model, reasoningEffort, task, bucket, recordID)

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return Sample{}, CompletionResult{}, false, nil
	}

	if err != nil {
		return Sample{}, CompletionResult{}, false, fmt.Errorf("reading task run: %w", err)
	}

	var file taskRunFile
	if err := json.Unmarshal(data, &file); err != nil {
		return Sample{}, CompletionResult{}, false, fmt.Errorf("parsing %s: %w", path, err)
	}

	if file.Model != model || file.ReasoningEffort != reasoningEffort || file.Task != task || file.Bucket != bucket || file.RecordID != recordID {
		return Sample{}, CompletionResult{}, false, fmt.Errorf("%s holds the run of %s/%s/%s/%s/%s, not of %s/%s/%s/%s/%s",
			path, file.Model, file.ReasoningEffort, file.Task, file.Bucket, file.RecordID, model, reasoningEffort, task, bucket, recordID)
	}

	s.mu.Lock()
	if s.reused == nil {
		s.reused = map[string]int{}
	}

	s.reused[task]++
	s.mu.Unlock()

	sample, result = file.sampleAndResult()

	return sample, result, true, nil
}

// anyRun returns one of the runs stashed for task -- whichever the directory
// lists first -- or ok=false if there are none. It is for what is true of every
// run of a task in one stash, such as the system prompt they were all sent.
func (s *TaskRunStore) anyRun(task string) (file taskRunFile, ok bool, err error) {
	dir := filepath.Join(s.dir, sanitizeFilename(task))

	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return taskRunFile{}, false, nil
	}

	if err != nil {
		return taskRunFile{}, false, err
	}

	for _, entry := range entries {
		if entry.IsDir() || !isTaskRunFileName(entry.Name()) {
			continue
		}

		path := filepath.Join(dir, entry.Name())

		data, err := os.ReadFile(path)
		if err != nil {
			return taskRunFile{}, false, err
		}

		if err := json.Unmarshal(data, &file); err != nil {
			return taskRunFile{}, false, fmt.Errorf("parsing %s: %w", path, err)
		}

		return file, true, nil
	}

	return taskRunFile{}, false, nil
}

// path is where one task run's file lives, each part passed through
// sanitizeFilename -- a model ID like "openai/gpt-4o" or a record ID like
// "aai_transcripts:30s:<uuid>" must neither become a subdirectory nor survive
// as a name some filesystem rejects. Taking the five parts rather than a
// Sample is deliberate: whoever wants a run's file back wants it before they
// have a Sample for it.
//
// It is built from the run's identity alone -- not a sequence number, the way
// FailureRecorder names its files -- so a run's file can be opened directly,
// with no listing or index, and saving a run twice replaces one file instead of
// adding a second. The mapping is lossy (every character outside
// sanitizeFilename's safe set becomes "_"), so a reader should trust the
// identity inside a file over its name.
func (s *TaskRunStore) path(model, reasoningEffort, task, bucket, recordID string) string {
	name := sanitizeFilename(strings.Join([]string{model, reasoningEffort, bucket, recordID}, "__"))

	return filepath.Join(s.dir, sanitizeFilename(task), name+".json")
}

// writeFileAtomic writes data to path, creating its directory if need be, by
// way of a temporary file beside it and a rename -- see TaskRunStore.Save. The
// temporary file is removed again if anything fails.
func writeFileAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}

	err = writeAndClose(tmp, data)
	if err == nil {
		err = os.Rename(tmp.Name(), path)
	}

	if err != nil {
		_ = os.Remove(tmp.Name())
	}

	return err
}

// writeAndClose writes data to f, makes it world-readable like every other file
// this tool writes (os.CreateTemp creates it 0600), and closes it.
func writeAndClose(f *os.File, data []byte) error {
	_, err := f.Write(data)

	if err == nil {
		err = f.Chmod(0o644)
	}

	if closeErr := f.Close(); err == nil {
		err = closeErr
	}

	return err
}
