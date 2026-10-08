package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeRecord is the minimal record shape fakeTask needs.
type fakeRecord struct {
	ID     string
	Bucket string
}

// RecordBucket implements framework.go's Bucketed, the constraint NewRunner
// requires of its type argument.
func (r fakeRecord) RecordBucket() string {
	return r.Bucket
}

// peakTracker tracks the maximum number of concurrent enter/leave pairs
// observed -- used to assert a Runner actually bounds concurrency rather
// than just accepting a limit value and ignoring it.
type peakTracker struct {
	mu      sync.Mutex
	current int
	max     int
}

func (p *peakTracker) enter() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.current++
	if p.current > p.max {
		p.max = p.current
	}
}

func (p *peakTracker) leave() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.current--
}

func (p *peakTracker) Max() int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return p.max
}

// fakeTask is a minimal, fully in-memory Task[fakeRecord] used to exercise
// Runner without any real dataset, gateway, or judge model -- concurrency
// and failure injection are driven directly by the fields below so tests
// can assert on Runner's orchestration behavior precisely.
type fakeTask struct {
	records []fakeRecord
	loadErr error

	// delay is held for the duration of every RunRecord/RunJudge/RunTTFT
	// call -- long enough, relative to JobLimiter's 10ms acquire-lock poll,
	// for genuine concurrent overlap to be observable by the peak trackers
	// below.
	delay time.Duration

	// failRecordID, if set, makes RunRecord fail for exactly that record ID.
	failRecordID string

	recordCalls atomic.Int32
	recordPeak  peakTracker
	judgeCalls  atomic.Int32
	judgePeak   peakTracker
	ttftCalls   atomic.Int32
}

func (t *fakeTask) Name() string { return "fake" }

func (t *fakeTask) LoadDataset() ([]fakeRecord, error) {
	if t.loadErr != nil {
		return nil, t.loadErr
	}

	return t.records, nil
}

func (t *fakeTask) SampleDataset(records []fakeRecord, _ int) []fakeRecord {
	return records
}

func (t *fakeTask) RunRecord(_ context.Context, model ModelInfo, record fakeRecord, _ *FailureRecorder) (Sample, CompletionResult, error) {
	t.recordCalls.Add(1)
	t.recordPeak.enter()
	defer t.recordPeak.leave()

	time.Sleep(t.delay)

	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID}

	if record.ID == t.failRecordID {
		err := fmt.Errorf("simulated failure for %s", record.ID)
		sample.Err = err

		return sample, CompletionResult{}, err
	}

	// Deliberately non-zero and distinguishable from RunJudge's own
	// (deliberately zero-valued) Sample -- TestRunnerRunPreservesRunRecordMeasurementsAfterJudging
	// checks these specific values survive into the final judged Sample.
	sample.TotalLatency = 42 * time.Millisecond
	sample.TimeToFirstToken = 42 * time.Millisecond
	sample.CompletionTokens = 7
	sample.PromptTokens = 11

	return sample, CompletionResult{Text: "output for " + record.ID}, nil
}

func (t *fakeTask) RunTTFT(_ context.Context, model ModelInfo, record fakeRecord, _ *FailureRecorder) (Sample, CompletionResult, error) {
	t.ttftCalls.Add(1)

	time.Sleep(t.delay)

	sample := Sample{Model: model.ID, Task: t.Name(), Bucket: record.Bucket, RecordID: record.ID}

	return sample, CompletionResult{Text: "streamed output for " + record.ID}, nil
}

func (t *fakeTask) RunJudge(_ context.Context, model ModelInfo, record fakeRecord, _ string, _ CompletionResult, _ *FailureRecorder) (Sample, error) {
	t.judgeCalls.Add(1)
	t.judgePeak.enter()
	defer t.judgePeak.leave()

	time.Sleep(t.delay)

	return Sample{
		Model:    model.ID,
		Task:     t.Name(),
		Bucket:   record.Bucket,
		RecordID: record.ID,
		Judge:    map[string]float64{"quality": 5},
	}, nil
}

func (t *fakeTask) DimensionRanges() map[string]scale {
	return map[string]scale{"quality": {lo: 1, hi: 5}}
}

func testModels(n int) []ModelInfo {
	models := make([]ModelInfo, n)
	for i := range models {
		models[i] = ModelInfo{ID: fmt.Sprintf("model-%d", i)}
	}

	return models
}

func testRecords(n int) []fakeRecord {
	records := make([]fakeRecord, n)
	for i := range records {
		records[i] = fakeRecord{ID: fmt.Sprintf("rec-%d", i), Bucket: "30s"}
	}

	return records
}

func TestRunnerRunProducesOneJudgedSamplePerModelPerRecord(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond}
	records := testRecords(4)
	models := testModels(2)

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:              models,
		JudgeModel:          "judge",
		MaxSamplesPerBucket: -1,
		Concurrency:         3,
	})

	samples := runner.Run(context.Background(), records)

	require.Len(t, samples, len(records)*len(models))

	for _, s := range samples {
		assert.NoError(t, s.Err)
		assert.Equal(t, float64(5), s.Judge["quality"])
		assert.NotEmpty(t, s.Model)
		assert.NotEmpty(t, s.RecordID)
	}

	assert.EqualValues(t, len(records)*len(models), task.recordCalls.Load())
	assert.EqualValues(t, len(records)*len(models), task.judgeCalls.Load())
}

// TestRunnerRunPreservesRunRecordMeasurementsAfterJudging is a regression
// test for a bug found while building DictationTask: runRecord used to
// return RunJudge's Sample wholesale on success, silently discarding
// RunRecord's latency/token measurements (fakeTask.RunJudge deliberately
// leaves them zero-valued, unlike fakeTask.RunRecord -- see its comment).
// runRecord must merge only Judge/SchemaInvalid from RunJudge's Sample onto
// RunRecord's, per RunJudge's doc comment on the Task interface.
func TestRunnerRunPreservesRunRecordMeasurementsAfterJudging(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond}
	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:      testModels(1),
		JudgeModel:  "judge",
		Concurrency: 1,
	})

	samples := runner.Run(context.Background(), testRecords(1))

	require.Len(t, samples, 1)
	s := samples[0]
	assert.Equal(t, 42*time.Millisecond, s.TotalLatency)
	assert.Equal(t, 42*time.Millisecond, s.TimeToFirstToken)
	assert.Equal(t, 7, s.CompletionTokens)
	assert.Equal(t, 11, s.PromptTokens)
	// The judge-scoring fields still come from RunJudge, not RunRecord.
	assert.Equal(t, float64(5), s.Judge["quality"])
}

// TestRunnerRunComputesCostFromModelPricing proves runRecord (not any Task
// implementation) stamps CostUSD, using the model's pricing against
// RunRecord's real token usage -- fakeTask.RunRecord always reports 11
// prompt / 7 completion tokens (see its comment).
func TestRunnerRunComputesCostFromModelPricing(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond}
	models := []ModelInfo{{ID: "model-0", PromptPricePerMillion: 3, CompletionPricePerMillion: 15}}

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:      models,
		JudgeModel:  "judge",
		Concurrency: 1,
	})

	samples := runner.Run(context.Background(), testRecords(1))

	require.Len(t, samples, 1)
	want := 11*3.0/1e6 + 7*15.0/1e6
	assert.InDelta(t, want, samples[0].CostUSD, 1e-12)
}

// TestRunnerRunTTFTNeverComputesCost proves the streaming, unjudged pass
// never stamps CostUSD -- cost is only meaningful alongside a quality score
// (see summary.go), and RunTTFT bypasses runRecord entirely (see runAll).
func TestRunnerRunTTFTNeverComputesCost(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond}
	models := []ModelInfo{{ID: "model-0", PromptPricePerMillion: 3, CompletionPricePerMillion: 15}}

	runner := NewRunner[fakeRecord](task, RunConfig{Models: models, Concurrency: 1})

	samples := runner.RunTTFT(context.Background(), testRecords(1))

	require.Len(t, samples, 1)
	assert.Zero(t, samples[0].CostUSD)
}

func TestRunnerRunSkipsJudgeOnRunRecordFailure(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond, failRecordID: "rec-1"}
	records := testRecords(3)

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:      testModels(1),
		JudgeModel:  "judge",
		Concurrency: 2,
	})

	samples := runner.Run(context.Background(), records)

	require.Len(t, samples, 3)

	var failed int

	for _, s := range samples {
		if s.RecordID == "rec-1" {
			assert.Error(t, s.Err)
			assert.Nil(t, s.Judge)

			failed++
		}
	}

	assert.Equal(t, 1, failed)
	// Only the 2 non-failing records should ever reach RunJudge.
	assert.EqualValues(t, 2, task.judgeCalls.Load())
}

func TestRunnerRunBoundsConcurrencyPerModel(t *testing.T) {
	task := &fakeTask{delay: 30 * time.Millisecond}
	records := testRecords(6)

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:      testModels(1),
		JudgeModel:  "judge",
		Concurrency: 2,
	})

	runner.Run(context.Background(), records)

	assert.LessOrEqual(t, task.recordPeak.Max(), 2)
	assert.GreaterOrEqual(t, task.recordPeak.Max(), 2, "expected genuine overlap up to the configured limit")
}

func TestRunnerRunBoundsJudgeConcurrencyAcrossModels(t *testing.T) {
	task := &fakeTask{delay: 30 * time.Millisecond}
	records := testRecords(4)

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:      testModels(2), // 2 models running fully in parallel...
		JudgeModel:  "judge",
		Concurrency: 2, // ...but at most 2 judge calls in flight TOTAL, shared between them.
	})

	runner.Run(context.Background(), records)

	assert.LessOrEqual(t, task.judgePeak.Max(), 2)
	assert.GreaterOrEqual(t, task.judgePeak.Max(), 2, "expected judge concurrency to be shared, not per-model")
}

func TestRunnerRunTTFTNeverCallsJudge(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond}
	records := testRecords(3)

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:      testModels(2),
		Concurrency: 2,
	})

	samples := runner.RunTTFT(context.Background(), records)

	require.Len(t, samples, 6)

	for _, s := range samples {
		assert.Nil(t, s.Judge)
	}

	assert.EqualValues(t, 0, task.judgeCalls.Load())
	assert.EqualValues(t, 6, task.ttftCalls.Load())
}

// TestRunnerRunTTFTFiltersToTTFTBuckets proves RunTTFT restricts records to
// TTFTBuckets (30s/120s) before a single streaming call is made -- a record
// from any other bucket must never reach Task.RunTTFT at all, not just be
// excluded from the returned samples.
func TestRunnerRunTTFTFiltersToTTFTBuckets(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond}
	records := []fakeRecord{
		{ID: "rec-30s", Bucket: "30s"},
		{ID: "rec-120s", Bucket: "120s"},
		{ID: "rec-900s", Bucket: "900s"},
		{ID: "rec-3600s", Bucket: "3600s"},
	}

	runner := NewRunner[fakeRecord](task, RunConfig{Models: testModels(1), Concurrency: 2})

	samples := runner.RunTTFT(context.Background(), records)

	require.Len(t, samples, 2, "only the 30s and 120s records should produce a sample")
	assert.EqualValues(t, 2, task.ttftCalls.Load(), "the 900s/3600s records must never reach Task.RunTTFT at all")

	var buckets []string
	for _, s := range samples {
		buckets = append(buckets, s.Bucket)
	}

	assert.ElementsMatch(t, []string{"30s", "120s"}, buckets)
}

func TestRunnerName(t *testing.T) {
	runner := NewRunner[fakeRecord](&fakeTask{}, RunConfig{})
	assert.Equal(t, "fake", runner.Name())
}

func TestRunnerLoadAndSample(t *testing.T) {
	task := &fakeTask{records: testRecords(5)}
	runner := NewRunner[fakeRecord](task, RunConfig{MaxSamplesPerBucket: -1})

	records, err := runner.LoadAndSample()

	require.NoError(t, err)
	assert.Len(t, records, 5)
}

// TestRunnerLoadAndSampleFiltersToRequestedBuckets proves RunConfig.Buckets,
// when non-empty, restricts LoadAndSample's output to just those buckets --
// mirroring RunTTFT's own fixed restriction to TTFTBuckets
// (TestRunnerRunTTFTFiltersToTTFTBuckets), but driven by the caller (e.g.
// -buckets) instead.
func TestRunnerLoadAndSampleFiltersToRequestedBuckets(t *testing.T) {
	task := &fakeTask{records: []fakeRecord{
		{ID: "rec-30s", Bucket: "30s"},
		{ID: "rec-120s", Bucket: "120s"},
		{ID: "rec-900s", Bucket: "900s"},
	}}

	runner := NewRunner[fakeRecord](task, RunConfig{MaxSamplesPerBucket: -1, Buckets: []string{"900s"}})

	records, err := runner.LoadAndSample()

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "900s", records[0].Bucket)
}

// TestRunnerLoadAndSampleEmptyBucketsMeansEveryBucket confirms the default,
// zero-value RunConfig.Buckets applies no filter at all -- the same "empty
// means everything" convention -models/filterModels uses.
func TestRunnerLoadAndSampleEmptyBucketsMeansEveryBucket(t *testing.T) {
	task := &fakeTask{records: []fakeRecord{
		{ID: "rec-30s", Bucket: "30s"},
		{ID: "rec-900s", Bucket: "900s"},
	}}

	runner := NewRunner[fakeRecord](task, RunConfig{MaxSamplesPerBucket: -1})

	records, err := runner.LoadAndSample()

	require.NoError(t, err)
	assert.Len(t, records, 2)
}

// TestRunnerRunBenchmarkFiltersToRequestedBuckets proves a bucket outside
// RunConfig.Buckets never reaches Task.RunRecord at all -- not just excluded
// from RunBenchmark's returned samples/cells -- the standard-pass
// counterpart to TestRunnerRunTTFTFiltersToTTFTBuckets.
func TestRunnerRunBenchmarkFiltersToRequestedBuckets(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond, records: []fakeRecord{
		{ID: "rec-30s", Bucket: "30s"},
		{ID: "rec-3600s", Bucket: "3600s"},
	}}

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:              testModels(1),
		JudgeModel:          "judge",
		MaxSamplesPerBucket: -1,
		Concurrency:         1,
		Buckets:             []string{"3600s"},
	})

	samples, cells, err := runner.RunBenchmark(context.Background())

	require.NoError(t, err)
	require.Len(t, samples, 1)
	assert.Equal(t, "3600s", samples[0].Bucket)
	require.Len(t, cells, 1)
	assert.Equal(t, "3600s", cells[0].Bucket)
	assert.EqualValues(t, 1, task.recordCalls.Load(), "the 30s record must never reach Task.RunRecord at all")
}

func TestRunnerRunBenchmarkLoadsSamplesAndAggregates(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond, records: testRecords(2)}

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:              testModels(1),
		JudgeModel:          "judge",
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	})

	samples, cells, err := runner.RunBenchmark(context.Background())

	require.NoError(t, err)
	assert.Len(t, samples, 2)
	// Both records share bucket "30s" for one model -- a single Cell.
	require.Len(t, cells, 1)
	assert.Equal(t, 2, cells[0].N)
	assert.Equal(t, 0, cells[0].Errors)
}

func TestRunnerRunTTFTBenchmarkLoadsSamplesAndAggregates(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond, records: testRecords(2)}

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:              testModels(1),
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
	})

	samples, cells, err := runner.RunTTFTBenchmark(context.Background())

	require.NoError(t, err)
	assert.Len(t, samples, 2)
	require.Len(t, cells, 1)
	assert.Equal(t, 0, cells[0].Judged)
}

func TestRunnerRunBenchmarkPropagatesLoadDatasetError(t *testing.T) {
	task := &fakeTask{loadErr: errors.New("boom")}

	runner := NewRunner[fakeRecord](task, RunConfig{Models: testModels(1), Concurrency: 1})

	_, _, err := runner.RunBenchmark(context.Background())

	assert.Error(t, err)
}

func TestRunnerZeroConcurrencyDoesNotDeadlock(t *testing.T) {
	task := &fakeTask{delay: time.Millisecond}
	records := testRecords(2)

	runner := NewRunner[fakeRecord](task, RunConfig{
		Models:      testModels(1),
		JudgeModel:  "judge",
		Concurrency: 0, // must be normalized to 1, not passed through to JobLimiter as-is.
	})

	done := make(chan []Sample, 1)
	go func() { done <- runner.Run(context.Background(), records) }()

	select {
	case samples := <-done:
		assert.Len(t, samples, 2)
	case <-time.After(5 * time.Second):
		t.Fatal("Run deadlocked with Concurrency: 0")
	}
}
