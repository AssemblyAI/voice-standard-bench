package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChrF(t *testing.T) {
	assert.InDelta(t, 100, ChrF("El gato está en la casa.", "El gato está en la casa."), 1e-9)
	assert.InDelta(t, 100, ChrF("", ""), 1e-9)
	assert.InDelta(t, 0, ChrF("", "Hola"), 1e-9)
	assert.InDelta(t, 0, ChrF("xyz", "Hola"), 1e-9)

	// Whitespace is ignored, as in sacrebleu.
	assert.InDelta(t, 100, ChrF("El gato  está", "El gato está"), 1e-9)

	// A closer translation scores higher than a further one.
	ref := "The meeting starts at nine tomorrow morning."
	close := ChrF("The meeting begins at nine tomorrow morning.", ref)
	far := ChrF("Tomorrow there is something.", ref)
	assert.Greater(t, close, far)
	assert.Less(t, close, 100.0)

	// Works on Japanese without a tokenizer.
	assert.Greater(t, ChrF("明日の会議は九時に始まります。", "明日の会議は九時に始まります。"), 99.9)
	assert.Greater(t, ChrF("明日の会議は九時からです。", "明日の会議は九時に始まります。"), 20.0)
}

func writeLiveTranslationFixture(t *testing.T, lines ...string) string {
	t.Helper()

	dir := t.TempDir()
	path := filepath.Join(dir, "fleurs.jsonl")
	require.NoError(t, os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644))

	return filepath.Join(dir, "*.jsonl")
}

const liveTranslationFixtureLine = `{"id":"s1-c2-es","input":{"source_language":"en","target_language":"es","turn":"The cat is in the house.","history":["Hello.","How are you?"],"source_truth":"the cat is in the house"},"expected":{"reference":"El gato está en la casa."},"baselines":{"async_su_translation":"El gato está en casa."}}`

func TestLoadLiveTranslationRecords(t *testing.T) {
	glob := writeLiveTranslationFixture(t,
		liveTranslationFixtureLine,
		`{"id":"s1-c3-es","input":{"source_language":"en","target_language":"es","turn":""},"expected":{"reference":"Adiós."}}`,
		`{"id":"s1-c2-es-max_accuracy","input":{"source_language":"en","target_language":"es","turn":"Hi.","variant":"max_accuracy"},"expected":{"reference":"Hola."}}`,
		`not json`,
		`{"id":"missing-reference","input":{"source_language":"en","target_language":"es","turn":"Hi"},"expected":{}}`,
	)

	records, skipped, err := LoadLiveTranslationRecords(glob)
	require.NoError(t, err)
	assert.Equal(t, 2, skipped)
	require.Len(t, records, 3)
	assert.Equal(t, "en-es/max_accuracy", records[2].Bucket)

	r := records[0]
	assert.Equal(t, "en-es", r.Bucket)
	assert.Equal(t, "The cat is in the house.", r.Turn)
	assert.Equal(t, []string{"Hello.", "How are you?"}, r.History)
	assert.Equal(t, "El gato está en la casa.", r.Reference)
	assert.Equal(t, "El gato está en casa.", r.AsyncBaseline)

	// A turn the speech model returned empty is kept, so the miss is scored.
	assert.Equal(t, "", records[1].Turn)
}

func TestLoadLiveTranslationRecordsNoFiles(t *testing.T) {
	_, _, err := LoadLiveTranslationRecords(filepath.Join(t.TempDir(), "*.jsonl"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prep.py")
}

func TestSampleLiveTranslationRecordsIsPerPairAndStable(t *testing.T) {
	var records []LiveTranslationRecord
	for _, pair := range []string{"en-es", "de-en"} {
		for i := 0; i < 10; i++ {
			records = append(records, LiveTranslationRecord{ID: pair + string(rune('a'+i)), Bucket: pair})
		}
	}

	a := SampleLiveTranslationRecords(records, 3)
	b := SampleLiveTranslationRecords(records, 3)
	require.Len(t, a, 6)
	assert.Equal(t, a, b)

	assert.Len(t, SampleLiveTranslationRecords(records, 0), 20)
}

func TestLiveTranslationUserMessage(t *testing.T) {
	interpreter := prompts.LiveTranslationPrompts["interpreter_v1"].User
	msg := liveTranslationUserMessage(interpreter, LiveTranslationRecord{TargetLanguage: "ja", Turn: "See you at nine.", History: []string{"Hi.", "When do we meet?"}})
	assert.Contains(t, msg, "Target language: Japanese")
	assert.Contains(t, msg, "Hi.\nWhen do we meet?\n")
	assert.True(t, strings.HasSuffix(msg, "Current turn:\nSee you at nine."))

	first := liveTranslationUserMessage(interpreter, LiveTranslationRecord{TargetLanguage: "xx", Turn: "Hello."})
	assert.Contains(t, first, "Target language: xx")
	assert.Contains(t, first, "(none)")
}

func TestLiveTranslationTaskRunRecordSendsPromptAndTurn(t *testing.T) {
	var body map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		writeChatCompletion(w, "El gato está en la casa.", 30, 8)
	}))
	defer server.Close()

	task, err := NewLiveTranslationTask(NewGateway(server.URL, "test-key", 5*time.Second))
	require.NoError(t, err)

	records, _, err := LoadLiveTranslationRecords(writeLiveTranslationFixture(t, liveTranslationFixtureLine))
	require.NoError(t, err)

	model := ModelInfo{ID: "gemini-3.1-flash-lite"}

	sample, result, err := task.RunRecord(context.Background(), model, records[0], nil)
	require.NoError(t, err)
	assert.Equal(t, prompts.LiveTranslation, sample.Task)
	assert.Equal(t, "en-es", sample.Bucket)
	assert.Equal(t, 8, sample.CompletionTokens)

	spec, _ := prompts.For(prompts.LiveTranslation)
	sent, _ := json.Marshal(body["messages"])
	assert.Contains(t, string(sent), "simultaneous interpreter")
	assert.Contains(t, string(sent), "Current turn:\\nThe cat is in the house.")
	assert.NotContains(t, string(sent), "El gato", "the reference and baseline must never reach the candidate")
	assert.NotEmpty(t, spec.Task)

	judged, err := task.RunJudge(context.Background(), model, records[0], "", result, nil)
	require.NoError(t, err)
	assert.InDelta(t, 100, judged.Judge["chrf"], 1e-9)
}

func TestLiveTranslationAsyncBaselineNeverCallsTheGateway(t *testing.T) {
	var calls int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		writeChatCompletion(w, "x", 1, 1)
	}))
	defer server.Close()

	task, err := NewLiveTranslationTask(NewGateway(server.URL, "test-key", 5*time.Second))
	require.NoError(t, err)

	records, _, err := LoadLiveTranslationRecords(writeLiveTranslationFixture(t, liveTranslationFixtureLine))
	require.NoError(t, err)

	baseline := ModelInfo{ID: AsyncBaselineModelID}

	_, result, err := task.RunRecord(context.Background(), baseline, records[0], nil)
	require.NoError(t, err)
	assert.Equal(t, "El gato está en casa.", result.Text)

	_, _, err = task.RunTTFT(context.Background(), baseline, records[0], nil)
	require.NoError(t, err)

	judged, err := task.RunJudge(context.Background(), baseline, records[0], "", result, nil)
	require.NoError(t, err)
	assert.Greater(t, judged.Judge["chrf"], 50.0)
	assert.Less(t, judged.Judge["chrf"], 100.0)

	assert.Equal(t, int32(0), atomic.LoadInt32(&calls))
}

func TestLiveTranslationRunJudgeAddsJudgeScoresToChrF(t *testing.T) {
	var judgeInput string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		judgeInput = string(raw)
		writeChatCompletion(w, `Fine. {"translation_accuracy": 4, "fluency": 5, "notes": "ok"}`, 10, 10)
	}))
	defer server.Close()

	task, err := NewLiveTranslationTask(NewGateway(server.URL, "test-key", 5*time.Second))
	require.NoError(t, err)

	records, _, err := LoadLiveTranslationRecords(writeLiveTranslationFixture(t, liveTranslationFixtureLine))
	require.NoError(t, err)

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate"}, records[0], "judge-model", CompletionResult{Text: "El gato está en casa."}, nil)
	require.NoError(t, err)
	assert.Equal(t, 4.0, judged.Judge["translation_accuracy"])
	assert.Equal(t, 5.0, judged.Judge["fluency"])
	assert.Contains(t, judged.Judge, "chrf")
	assert.Contains(t, judgeInput, "REFERENCE:")
	assert.Contains(t, judgeInput, "TARGET LANGUAGE: Spanish")

	for dim := range judged.Judge {
		_, ok := task.DimensionRanges()[dim]
		assert.True(t, ok, "%s needs a dimension range", dim)
		_, ok = normalize(dim, judged.Judge[dim])
		assert.True(t, ok, "%s must count toward quality_score", dim)
	}
}

func TestLiveTranslationConstructorAddsAsyncBaselineRow(t *testing.T) {
	models := []ModelInfo{{ID: "gemini-3.1-flash-lite"}}

	withBaseline := withAsyncBaseline(models)
	require.Len(t, withBaseline, 2)
	assert.Equal(t, AsyncBaselineModelID, withBaseline[1].ID)
	assert.Len(t, models, 1, "the caller's slice is not modified")
	assert.Len(t, withAsyncBaseline(withBaseline), 2, "added once")

	_, ok := frameworkTaskConstructors[prompts.LiveTranslation]
	assert.True(t, ok)
}

func TestLiveTranslationPromptVariants(t *testing.T) {
	r := LiveTranslationRecord{SourceLanguage: "en", TargetLanguage: "es", Turn: "See you at nine.", History: []string{"When do we meet?"}}

	oneliner := liveTranslationUserMessage(prompts.LiveTranslationPrompts["docs_oneliner"].User, r)
	assert.Equal(t, "Translate the following English into natural Spanish and reply with only the translation: See you at nine.", oneliner)

	withHistory := liveTranslationUserMessage(prompts.LiveTranslationPrompts["docs_history"].User, r)
	assert.Contains(t, withHistory, "When do we meet?")
	assert.True(t, strings.HasSuffix(withHistory, "reply with only the translation: See you at nine."))

	for name, p := range prompts.LiveTranslationPrompts {
		assert.Contains(t, p.User, "{{turn}}", name)
		assert.Contains(t, p.User, "{{target}}", name)
	}
}

func TestLiveTranslationPromptSelection(t *testing.T) {
	t.Setenv("LIVE_TRANSLATION_PROMPT", "")
	name, _, ok := prompts.ActiveLiveTranslationPrompt()
	assert.True(t, ok)
	assert.Equal(t, prompts.DefaultLiveTranslationPrompt, name)

	t.Setenv("LIVE_TRANSLATION_PROMPT", "docs_oneliner")
	task, err := NewLiveTranslationTask(NewGateway("http://unused", "k", time.Second))
	require.NoError(t, err)
	assert.Empty(t, task.prompt.System)

	t.Setenv("LIVE_TRANSLATION_PROMPT", "nope")
	_, err = NewLiveTranslationTask(NewGateway("http://unused", "k", time.Second))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "docs_history")
}

func TestOneMessagePromptSendsNoSystemMessage(t *testing.T) {
	var body map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		writeChatCompletion(w, "Nos vemos a las nueve.", 10, 5)
	}))
	defer server.Close()

	t.Setenv("LIVE_TRANSLATION_PROMPT", "docs_oneliner")
	task, err := NewLiveTranslationTask(NewGateway(server.URL, "k", 5*time.Second))
	require.NoError(t, err)

	_, _, err = task.RunRecord(context.Background(), ModelInfo{ID: "m"}, LiveTranslationRecord{ID: "r", Bucket: "en-es", SourceLanguage: "en", TargetLanguage: "es", Turn: "See you at nine."}, nil)
	require.NoError(t, err)

	msgs := body["messages"].([]any)
	require.Len(t, msgs, 1)
	assert.Equal(t, "user", msgs[0].(map[string]any)["role"])
}

func TestLiveTranslationRecordWithoutReferenceIsJudgedAgainstTranscript(t *testing.T) {
	glob := writeLiveTranslationFixture(t,
		`{"id":"ami-ES2004a-t001-es","input":{"source_language":"en","target_language":"es","turn":"So the remote should be cheap.","source_truth":"so the uh remote should be cheap","variant":"ami"},"expected":{"reference":""},"baselines":{"async_su_translation":"El mando debe ser barato."}}`,
	)

	records, skipped, err := LoadLiveTranslationRecords(glob)
	require.NoError(t, err)
	assert.Zero(t, skipped)
	require.Len(t, records, 1)
	assert.Equal(t, "en-es/ami", records[0].Bucket)

	var judgeInput string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		judgeInput = string(raw)
		writeChatCompletion(w, `{"translation_accuracy": 5, "fluency": 5}`, 10, 10)
	}))
	defer server.Close()

	task, err := NewLiveTranslationTask(NewGateway(server.URL, "k", 5*time.Second))
	require.NoError(t, err)

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "m"}, records[0], "judge", CompletionResult{Text: "El mando debería ser barato."}, nil)
	require.NoError(t, err)
	assert.NotContains(t, judged.Judge, "chrf", "no reference, so no chrF")
	assert.Equal(t, 5.0, judged.Judge["translation_accuracy"])
	assert.Contains(t, judgeInput, "WHAT WAS SAID")
	assert.Contains(t, judgeInput, "There is no reference translation")
}
