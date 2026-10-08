package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestTextCorrectionTask constructs a TextCorrectionTask for tests that
// don't need a real gateway or dataset.
func newTestTextCorrectionTask(t *testing.T, gw *Gateway) *TextCorrectionTask {
	t.Helper()

	task, err := NewTextCorrectionTask(gw, false)
	require.NoError(t, err)

	return task
}

// textCorrectionJudgeReply is what a judge sends back for a text_correction
// judge call. There are two: the reference-free one (faithfulness, improvement)
// and the reference-based meaning comparison, recognizable by the REFERENCE:
// section of its user message (meaning_similarity). A fake gateway answers
// whichever it was asked, so a test never has to care which order RunJudge
// makes them in.
func textCorrectionJudgeReply(requestBody string, faithfulness, improvement, meaning int) string {
	if strings.Contains(requestBody, "REFERENCE:") {
		return fmt.Sprintf(`{"meaning_similarity": %d, "notes": "the meaning note"}`, meaning)
	}

	return fmt.Sprintf(`{"faithfulness": %d, "improvement": %d, "notes": "ok"}`, faithfulness, improvement)
}

func TestNewTextCorrectionTaskResolvesSpecAndName(t *testing.T) {
	task, err := NewTextCorrectionTask(nil, false)
	require.NoError(t, err)
	assert.Equal(t, prompts.TextCorrection, task.Name())
}

// ---------------------------------------------------------------------------
// LoadDataset / SampleDataset -- thin adapters around dataset.go, mirroring
// summarization_task_test.go's coverage.
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskLoadDataset(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "text_correction", "aai_transcripts.jsonl"),
		recordLineWithMissedText("r1", "30s", "the true text", "the corrupted text"))

	task, err := NewTextCorrectionTask(nil, false)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "the true text", records[0].Text)
	assert.Equal(t, "the corrupted text", records[0].MissedText)
	assert.Equal(t, "30s", records[0].Bucket)
}

func TestTextCorrectionTaskLoadDatasetOnlyReturnsTextCorrectionRecords(t *testing.T) {
	t.Chdir(t.TempDir())
	writeFile(t, filepath.Join(normalizedRoot, "text_correction", "aai_transcripts.jsonl"),
		recordLineWithMissedText("c1", "30s", "true", "corrupted"))
	// A sibling task folder must never leak in, same as SummarizationTask.
	writeFile(t, filepath.Join(normalizedRoot, "summarization", "aai_transcripts.jsonl"), recordLine("s1", "30s", "summarization transcript"))

	task, err := NewTextCorrectionTask(nil, false)
	require.NoError(t, err)

	records, err := task.LoadDataset()

	require.NoError(t, err)
	require.Len(t, records, 1, "summarization's sibling folder must never leak into text_correction's own, task-scoped glob")
	assert.Equal(t, "c1", records[0].ID)
}

func TestTextCorrectionTaskLoadDatasetPropagatesGlobError(t *testing.T) {
	t.Chdir(t.TempDir()) // empty -- normalized/text_correction/aai_transcripts.jsonl matches nothing

	task, err := NewTextCorrectionTask(nil, false)
	require.NoError(t, err)

	_, err = task.LoadDataset()
	assert.Error(t, err)
}

func TestTextCorrectionTaskSampleDatasetMatchesSampleByBucket(t *testing.T) {
	task := newTestTextCorrectionTask(t, nil)

	records := []TranscriptRecord{
		{ID: "c", Bucket: "30s"}, {ID: "a", Bucket: "30s"}, {ID: "b", Bucket: "30s"},
		{ID: "z", Bucket: "120s"},
	}

	got := task.SampleDataset(records, 2)

	byBucket := map[string][]TranscriptRecord{}
	for _, r := range records {
		byBucket[r.Bucket] = append(byBucket[r.Bucket], r)
	}

	sampled := SampleByBucket(byBucket, 2)

	var want []TranscriptRecord
	for _, bucket := range Buckets(sampled) {
		want = append(want, sampled[bucket]...)
	}

	assert.Equal(t, want, got)
}

// ---------------------------------------------------------------------------
// maxCompletionTokens -- must match the text_correction branch of the
// original benchmark's taskMaxTokens exactly (shared there with dictation's
// identical formula).
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskMaxCompletionTokensMatchesOriginalTaskMaxTokens(t *testing.T) {
	task := newTestTextCorrectionTask(t, nil)

	cases := []struct {
		name  string
		input string
		model ModelInfo
	}{
		{"short input, no model cap", "the corrupted text", ModelInfo{ID: "m"}},
		{"empty input", "", ModelInfo{ID: "m"}},
		{"long input, no model cap", strings.Repeat("word ", 2000), ModelInfo{ID: "m"}},
		{"model cap clamps a small budget", "the corrupted text", ModelInfo{ID: "m", MaxCompletionTokens: 100}},
		{"model cap does not clamp a budget already under it", strings.Repeat("word ", 2000), ModelInfo{ID: "m", MaxCompletionTokens: 1_000_000}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := taskMaxTokens(prompts.TextCorrection, tc.input, tc.model)
			got := task.maxCompletionTokens(tc.input, tc.model)
			assert.Equal(t, want, got)
		})
	}
}

// ---------------------------------------------------------------------------
// RunRecord / RunJudge
// ---------------------------------------------------------------------------

// TestTextCorrectionTaskKeepsTheAnswerKeyFromTheModelAndTheReferenceFreeJudge:
// record.Text is the hidden answer key and must never reach the candidate
// model, nor the reference-free judge whose faithfulness and improvement scores
// have to stay independent of it. It goes to exactly one place: the
// reference-based meaning comparison, which exists to read it.
func TestTextCorrectionTaskKeepsTheAnswerKeyFromTheModelAndTheReferenceFreeJudge(t *testing.T) {
	const trueText = "TRUE-TEXT-MUST-NOT-LEAK the meeting is at three"
	const corrupted = "the meating is at three"
	const modelOutput = "the meeting is at four" // a plausible but imperfect correction

	var taskRequestBody, referenceFreeBody, meaningBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)

		if strings.Contains(body, `"model":"the-judge"`) {
			if strings.Contains(body, "REFERENCE:") {
				meaningBody = body
			} else {
				referenceFreeBody = body
			}

			writeChatCompletion(w, textCorrectionJudgeReply(body, 4, 4, 3), 20, 10)

			return
		}

		taskRequestBody = body
		writeChatCompletion(w, modelOutput, 30, 8)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)
	model := ModelInfo{ID: "candidate-model"}
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: trueText, MissedText: corrupted}

	sample, result, err := task.RunRecord(context.Background(), model, record, nil)
	require.NoError(t, err)
	assert.NoError(t, sample.Err)

	judged, err := task.RunJudge(context.Background(), model, record, "the-judge", result, nil)
	require.NoError(t, err)

	require.NotEmpty(t, taskRequestBody, "the task call must have happened")
	require.NotEmpty(t, referenceFreeBody, "the reference-free judge call must have happened")
	require.NotEmpty(t, meaningBody, "the meaning comparison call must have happened")

	assert.Contains(t, taskRequestBody, corrupted, "the model must be given the corrupted transcript")
	assert.NotContains(t, taskRequestBody, "TRUE-TEXT-MUST-NOT-LEAK", "the model must never see the true text")

	assert.Contains(t, referenceFreeBody, corrupted, "the reference-free judge grades against what the model was shown")
	assert.Contains(t, referenceFreeBody, modelOutput)
	assert.NotContains(t, referenceFreeBody, "TRUE-TEXT-MUST-NOT-LEAK", "the reference-free judge must never see the true text")

	assert.Contains(t, meaningBody, "TRUE-TEXT-MUST-NOT-LEAK", "the meaning comparison is the one call that is shown the true text")
	assert.Contains(t, meaningBody, modelOutput, "next to the candidate's output")
	assert.NotContains(t, meaningBody, corrupted, "and not the corrupted input, which it has no use for")

	assert.Equal(t, 4.0, judged.Judge["faithfulness"])
	assert.Equal(t, 4.0, judged.Judge["improvement"])
	assert.Equal(t, 3.0, judged.Judge["meaning_similarity"])
}

// TestTextCorrectionTaskRunJudgeScoresMeaningAndRecordsWordLevelContext is the
// end-to-end check of what a judged sample carries: the reference-free judge's
// faithfulness and improvement, the meaning comparison's meaning_similarity
// (the truth-based quality score), and -- as context only -- the word error
// rates of the output and of the input against the truth.
func TestTextCorrectionTaskRunJudgeScoresMeaningAndRecordsWordLevelContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(readBody(t, r), 4, 3, 5), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{
		ID: "rec-1", Bucket: "30s",
		Text:       "we agreed to ship the budget review on friday",
		MissedText: "we agreed to shop the budjet review on friday", // 2 wrong words of 9
	}
	result := CompletionResult{Text: "We agreed to ship the budjet review on Friday."} // fixes one, re-punctuates

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", result, nil)

	require.NoError(t, err)
	assert.Equal(t, 4.0, judged.Judge["faithfulness"])
	assert.Equal(t, 3.0, judged.Judge["improvement"])
	assert.Equal(t, 5.0, judged.Judge["meaning_similarity"], "the judge's verdict, taken as given")
	assert.InDelta(t, 100.0/9, judged.Judge[scoreTruthWER], 1e-9, "one wrong word of nine left; casing and punctuation are not words")
	assert.InDelta(t, 200.0/9, judged.Judge[scoreInputWER], 1e-9, "the do-nothing baseline: two wrong words of nine")

	for _, retired := range []string{"accuracy", "edit_ratio", "error_reduction"} {
		assert.NotContains(t, judged.Judge, retired, "%s is no longer recorded", retired)
	}
}

// TestTextCorrectionTaskRunJudgeSendsTheMeaningJudgeTheTruthAndTheOutput pins
// the shape of the reference-based call: a REFERENCE section holding the
// truth, a CANDIDATE section holding the output, and the Compare system prompt
// -- not the reference-free one.
func TestTextCorrectionTaskRunJudgeSendsTheMeaningJudgeTheTruthAndTheOutput(t *testing.T) {
	var meaning []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if strings.Contains(body, "REFERENCE:") {
			meaning = []byte(body)
		}

		writeChatCompletion(w, textCorrectionJudgeReply(body, 5, 5, 5), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the real truth", MissedText: "the corrupted input"}

	_, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: "the model output"}, nil)
	require.NoError(t, err)
	require.NotEmpty(t, meaning)

	var request struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}

	require.NoError(t, json.Unmarshal(meaning, &request))
	require.Len(t, request.Messages, 2)

	assert.Equal(t, "system", request.Messages[0].Role)
	assert.Equal(t, task.spec.Compare, request.Messages[0].Content, "the Compare prompt, not the reference-free one")
	assert.Equal(t, "user", request.Messages[1].Role)
	assert.Equal(t, "REFERENCE:\nthe real truth\n\nCANDIDATE:\nthe model output", request.Messages[1].Content)
}

// A sample is judged by both calls or not at all: if the meaning comparison
// fails the sample comes back unjudged, with the failure recorded, rather than
// carrying a half-filled rubric.
func TestTextCorrectionTaskRunJudgeFailsTheSampleWhenTheMeaningComparisonFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if strings.Contains(body, "REFERENCE:") {
			writeChatCompletion(w, "I would rather not compare these.", 20, 10)

			return
		}

		writeChatCompletion(w, textCorrectionJudgeReply(body, 4, 4, 4), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the meeting is at three", MissedText: "the meating is at three"}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: "the meeting is at three"}, failures)

	require.Error(t, err)
	assert.Nil(t, judged.Judge, "no partial rubric")

	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, FailureJudgeError, meta[0].Reason)

	output := readFailureOutput(t, failures.RunDir(), meta[0].OutputFile)
	assert.Contains(t, string(output.Request), "REFERENCE:", "the saved request is the meaning comparison's, so the failure can be replayed")
	assert.Contains(t, string(output.Response), "I would rather not compare these.")
}

// The word-level numbers are context: a do-nothing candidate sits exactly as
// far from the truth as its input, however the judge scored it.
func TestTextCorrectionTaskRunJudgeRecordsADoNothingCandidateAtTheInputsDistance(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(readBody(t, r), 5, 1, 4), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	truth := strings.Repeat("the quarterly numbers look strong across every region ", 20)
	corrupted := strings.Replace(truth, "strong", "strung", 3)
	record := TranscriptRecord{ID: "rec-1", Bucket: "120s", Text: truth, MissedText: corrupted}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: corrupted}, nil)

	require.NoError(t, err)
	assert.InDelta(t, judged.Judge[scoreInputWER], judged.Judge[scoreTruthWER], 1e-9)
	assert.Greater(t, 100-judged.Judge[scoreTruthWER], 95.0, "a do-nothing candidate is 'nearly perfect' word for word, which is why the word-level numbers are not the score")
}

// TestTextCorrectionTaskRunRecordNoMissedTextIsAFailedSample: a record the
// corruption simulation didn't cover fails closed instead of silently
// correcting the true text against itself.
func TestTextCorrectionTaskRunRecordNoMissedTextIsAFailedSample(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the true text", MissedText: ""}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen at all when there's nothing to correct")
}

func TestTextCorrectionTaskRunRecordRecordsAPIErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: "corrupted"}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model"}, record, failures)

	require.Error(t, err)
	require.Error(t, sample.Err)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.TextCorrection, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureAPIError, meta[0].Reason)
}

func TestTextCorrectionTaskRunRecordContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunRecord(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen once the context-length guard trips")
}

func TestTextCorrectionTaskRunJudgeRecordsJudgeErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "I refuse to grade this.", 10, 8)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: "corrupted"}

	_, err = task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", CompletionResult{Text: "a correction"}, failures)

	require.Error(t, err, "a judge reply with no JSON object must surface as an error")
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.TextCorrection, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureJudgeError, meta[0].Reason)
}

// TestTextCorrectionTaskRunJudgeRecordsLowAccuracyFailure confirms a word
// accuracy against the truth below lowAccuracyThreshold is recorded as
// FailureLowAccuracy (via recordLowAccuracy).
func TestTextCorrectionTaskRunJudgeRecordsLowAccuracyFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(readBody(t, r), 3, 3, 1), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "completely unrelated true text here", MissedText: "corrupted input"}
	// Shares no words with record.Text -> every word wrong, WER 100%, word
	// accuracy 0, well below the floor.
	result := CompletionResult{Text: "a totally different sentence entirely"}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", result, failures)

	require.NoError(t, err)
	assert.Equal(t, 100.0, judged.Judge[scoreTruthWER])
	assert.Less(t, 1-judged.Judge[scoreTruthWER]/100, lowAccuracyThreshold)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, FailureLowAccuracy, meta[0].Reason)
	require.NotNil(t, meta[0].TruthWER, "meta.json must surface the truth WER for a low_accuracy entry")
	assert.InDelta(t, judged.Judge[scoreTruthWER], *meta[0].TruthWER, 1e-9)
}

// TestTextCorrectionTaskRunJudgeRecordsRunawayGeneration covers the failure
// the old word-overlap accuracy was added to catch and the kimi-k3 runs
// actually hit: output many times longer than the truth. WER passes 100% (more
// errors than the truth has words), so word accuracy goes negative and trips
// the floor.
func TestTextCorrectionTaskRunJudgeRecordsRunawayGeneration(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(readBody(t, r), 1, 1, 1), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	truth := "the meeting is at three"
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: truth, MissedText: "the meating is at three"}
	result := CompletionResult{Text: truth + strings.Repeat(" and then the meeting is at three", 40)}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", result, failures)

	require.NoError(t, err)
	assert.Greater(t, judged.Judge[scoreTruthWER], 100.0, "more wrong words than the truth has words")
	assert.Equal(t, 1.0, judged.Judge["meaning_similarity"], "and the meaning judge scored it the worst there is")
	assert.Equal(t, 1, failures.Count())
}

// TestTextCorrectionTaskRunJudgeDoesNotRecordAMereRegression: a candidate
// that is worse than its input but still mostly right is a lower score, not a
// grossly-wrong output worth a transcript-sized file in errors/ -- the failure
// capture stays reserved for the latter.
func TestTextCorrectionTaskRunJudgeDoesNotRecordAMereRegression(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(readBody(t, r), 3, 2, 3), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	// Ten words; the input has one wrong, the candidate three.
	record := TranscriptRecord{
		ID: "rec-1", Bucket: "30s",
		Text:       "one two three four five six seven eight nine ten",
		MissedText: "one two three four five six seven eight nine tin",
	}
	result := CompletionResult{Text: "one two three four five six seven ait nin tin"}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", result, failures)

	require.NoError(t, err)
	assert.Greater(t, judged.Judge[scoreTruthWER], judged.Judge[scoreInputWER], "it did get worse than its input")
	assert.Equal(t, 0, failures.Count(), "but 70% word accuracy is nowhere near a gross failure")
}

func TestTextCorrectionTaskRunJudgeSkipsLowAccuracyRecordingAboveThreshold(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(readBody(t, r), 5, 5, 5), 20, 10)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	task := newTestTextCorrectionTask(t, gw)
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the meeting is at three", MissedText: "the meating is at three"}
	result := CompletionResult{Text: "the meeting is at three"}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model", result, failures)

	require.NoError(t, err)
	assert.Equal(t, 0.0, judged.Judge[scoreTruthWER])
	assert.GreaterOrEqual(t, 1-judged.Judge[scoreTruthWER]/100, lowAccuracyThreshold)
	assert.Equal(t, 0, failures.Count())
}

// ---------------------------------------------------------------------------
// RunTTFT -- the original benchmark already ran a TTFT pass over
// text_correction (ttft.go's runTTFTOne, generic across the three transcript
// tasks), so this is a real comparison, like summarization_task_test.go's.
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskRunTTFTMatchesRunTTFTOne(t *testing.T) {
	var bodies []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodies = append(bodies, readBody(t, r))

		w.Header().Set("Content-Type", "text/event-stream")
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"the meeting "},"finish_reason":null}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"is at three"},"finish_reason":"stop"}]}`)
		writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	model := ModelInfo{ID: "candidate-model"}
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the meeting is at three", MissedText: "the meating is at three"}

	spec, ok := prompts.For(prompts.TextCorrection)
	require.True(t, ok)

	// Old path.
	want := runTTFTOne(context.Background(), gw, model, prompts.TextCorrection, spec, record, nil)
	require.NoError(t, want.Err)

	// New path.
	task := newTestTextCorrectionTask(t, gw)
	sample, result, err := task.RunTTFT(context.Background(), model, record, nil)
	require.NoError(t, err)

	assert.Equal(t, want.Model, sample.Model)
	assert.Equal(t, want.Task, sample.Task)
	assert.Equal(t, want.Bucket, sample.Bucket)
	assert.Equal(t, want.RecordID, sample.RecordID)
	assert.Equal(t, "the meeting is at three", result.Text)
	assert.Nil(t, sample.Judge, "a TTFT sample is never judged")

	require.Len(t, bodies, 2, "both paths must make exactly one streaming call each")

	var oldReq, newReq chatWireRequest
	require.NoError(t, json.Unmarshal([]byte(bodies[0]), &oldReq))
	require.NoError(t, json.Unmarshal([]byte(bodies[1]), &newReq))

	assert.Equal(t, oldReq.MaxTokens, newReq.MaxTokens)
	assert.Equal(t, oldReq.Messages, newReq.Messages, "both paths must send MissedText, never Text")
}

func TestTextCorrectionTaskRunTTFTContextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "true", MissedText: strings.Repeat("word ", 1000)}

	sample, _, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model", ContextLength: 10}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called)
}

func TestTextCorrectionTaskRunTTFTNoMissedTextSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	task := newTestTextCorrectionTask(t, gw)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the true text", MissedText: ""}

	sample, _, err := task.RunTTFT(context.Background(), ModelInfo{ID: "candidate-model"}, record, nil)

	assert.Error(t, err)
	assert.Error(t, sample.Err)
	assert.False(t, called)
}

// ---------------------------------------------------------------------------
// DimensionRanges
// ---------------------------------------------------------------------------

func TestTextCorrectionTaskDimensionRanges(t *testing.T) {
	task := newTestTextCorrectionTask(t, nil)

	spec, ok := prompts.For(prompts.TextCorrection)
	require.True(t, ok)

	ranges := task.DimensionRanges()
	require.Len(t, ranges, len(spec.Rubric)+len(spec.CompareRubric), "the reference-free rubric plus the meaning comparison's")

	for _, dim := range append(append([]string{}, spec.Rubric...), spec.CompareRubric...) {
		got, ok := ranges[dim]
		require.True(t, ok, "missing DimensionRanges entry for %q", dim)
		assert.Equal(t, scale{lo: 1, hi: 5}, got, "%q is an LLM judge's 1-5 score", dim)
	}

	assert.Contains(t, ranges, "meaning_similarity")

	// The informational WERs and the retired scores are not quality
	// dimensions, so they must not appear -- and must agree with summary.go's
	// dimensionRange, which is what actually decides what reaches quality_score.
	for _, notQuality := range []string{scoreTruthWER, scoreInputWER, "accuracy", "edit_ratio", "error_reduction"} {
		assert.NotContains(t, ranges, notQuality)
		assert.NotContains(t, dimensionRange, notQuality)
	}

	for dim, got := range ranges {
		assert.Equal(t, dimensionRange[dim], got, "%q must be scored on the same scale as summary.go's dimensionRange", dim)
	}
}
