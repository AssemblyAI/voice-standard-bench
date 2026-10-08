package main

import (
	"context"
	"encoding/json"
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

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

// dictationRecordLine builds one dictation/synthetic.jsonl-shaped line,
// mirroring dataset_test.go's recordLine for TranscriptRecord.
func dictationRecordLine(id, bucket, instruction, text, truth string, metrics []string) string {
	metricsJSON := `["` + strings.Join(metrics, `","`) + `"]`
	if len(metrics) == 0 {
		metricsJSON = "[]"
	}

	return `{"id":"` + id + `","input":{"text":"` + text + `","instruction":"` + instruction +
		`","context":{"audio_length_bucket":"` + bucket + `"}},"expected":{"truth_text":"` + truth +
		`"},"metrics":` + metricsJSON + `}` + "\n"
}

func TestLoadDictationRecords(t *testing.T) {
	t.Run("loads text, bucket, truth_text, and metrics, ignoring the instruction key entirely", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "synthetic.jsonl")
		writeFile(t, path, dictationRecordLine("r1", "30s", "follow the commands", "hey comma world period",
			"Hey, world.", []string{"exact_match", "word_error_rate"}))

		records, skipped, err := LoadDictationRecords(filepath.Join(root, "*.jsonl"))
		require.NoError(t, err)
		assert.Equal(t, 0, skipped)
		require.Len(t, records, 1)

		r := records[0]
		assert.Equal(t, "r1", r.ID)
		assert.Equal(t, "30s", r.Bucket)
		assert.Equal(t, "hey comma world period", r.Input)
		assert.Equal(t, "Hey, world.", r.TruthText)
		assert.Equal(t, []string{"exact_match", "word_error_rate"}, r.Metrics)
		// The line built above embeds a real "instruction" value
		// ("follow the commands") in its JSON -- DictationRecord simply has
		// no field to hold it (see rawDictationRecord's doc comment), so its
		// presence in the raw data neither errors nor surfaces anywhere.
	})

	t.Run("a malformed line is skipped and counted, not fatal", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "synthetic.jsonl")
		writeFile(t, path,
			dictationRecordLine("good", "30s", "instr", "text", "truth", []string{"exact_match"})+
				"{not valid json\n")

		records, skipped, err := LoadDictationRecords(filepath.Join(root, "*.jsonl"))
		require.NoError(t, err)
		assert.Equal(t, 1, skipped)
		require.Len(t, records, 1)
		assert.Equal(t, "good", records[0].ID)
	})

	t.Run("missing text, bucket, or truth_text is skipped and counted", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "synthetic.jsonl")
		writeFile(t, path,
			dictationRecordLine("good", "30s", "instr", "text", "truth", nil)+
				`{"id":"no-text","input":{"text":"","instruction":"i","context":{"audio_length_bucket":"30s"}},"expected":{"truth_text":"t"}}`+"\n"+
				`{"id":"no-bucket","input":{"text":"hi","instruction":"i","context":{}},"expected":{"truth_text":"t"}}`+"\n"+
				`{"id":"no-truth","input":{"text":"hi","instruction":"i","context":{"audio_length_bucket":"30s"}},"expected":{"truth_text":""}}`+"\n")

		records, skipped, err := LoadDictationRecords(filepath.Join(root, "*.jsonl"))
		require.NoError(t, err)
		assert.Equal(t, 3, skipped)
		require.Len(t, records, 1)
		assert.Equal(t, "good", records[0].ID)
	})

	t.Run("no file matches the glob is an error, not an empty result", func(t *testing.T) {
		_, _, err := LoadDictationRecords(filepath.Join(t.TempDir(), "nothing-here", "*.jsonl"))
		require.Error(t, err)
	})
}

func TestSampleDictationRecords(t *testing.T) {
	records := []DictationRecord{
		{ID: "c", Bucket: "30s"}, {ID: "a", Bucket: "30s"}, {ID: "b", Bucket: "30s"},
		{ID: "z", Bucket: "120s"},
	}

	t.Run("caps per bucket after sorting by id, for reproducible sampling", func(t *testing.T) {
		sampled := SampleDictationRecords(records, 2)

		var thirtySec []string
		for _, r := range sampled {
			if r.Bucket == "30s" {
				thirtySec = append(thirtySec, r.ID)
			}
		}

		assert.Equal(t, []string{"a", "b"}, thirtySec)
	})

	t.Run("a cap on one bucket doesn't touch another bucket's records", func(t *testing.T) {
		sampled := SampleDictationRecords(records, 2)

		var oneTwenty []string
		for _, r := range sampled {
			if r.Bucket == "120s" {
				oneTwenty = append(oneTwenty, r.ID)
			}
		}

		assert.Equal(t, []string{"z"}, oneTwenty)
	})

	t.Run("max <= 0 means no cap and returns the input unchanged", func(t *testing.T) {
		assert.Equal(t, records, SampleDictationRecords(records, 0))
	})
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

func TestSplitPunct(t *testing.T) {
	tests := []struct {
		name                             string
		field                            string
		wantLeading, wantCore, wantTrail string
	}{
		{"plain word", "world", "", "world", ""},
		{"trailing comma", "Hey,", "", "Hey", ","},
		{"trailing question mark", "tonight?", "", "tonight", "?"},
		{"leading open quote", "\"hello", "\"", "hello", ""},
		{"trailing close-quote then period", "game\".", "", "game", "\"."},
		{"leading open paren", "(case", "(", "case", ""},
		{"trailing close-paren then period", "four).", "", "four", ")."},
		{"internal hyphen stays in the core", "all-hands", "", "all-hands", ""},
		{"internal apostrophe (possessive) stays in the core", "Jordan's", "", "Jordan's", ""},
		{"internal apostrophe (contraction) stays in the core", "don't", "", "don't", ""},
		{"no word rune at all", "---", "---", "", ""},
		{"empty field", "", "", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			leading, core, trailing := splitPunct(tt.field)
			assert.Equal(t, tt.wantLeading, leading, "leading")
			assert.Equal(t, tt.wantCore, core, "core")
			assert.Equal(t, tt.wantTrail, trailing, "trailing")
		})
	}
}

// surfaces is a small test helper: the surface field of every token, in
// order, for asserting tokenizeText's output without repeating every other
// field in each expectation.
func surfaces(tokens []dictationToken) []string {
	out := make([]string, len(tokens))
	for i, tok := range tokens {
		out[i] = tok.surface
	}

	return out
}

func TestTokenizeText(t *testing.T) {
	t.Run("plain sentence: leading/trailing punctuation attaches to the adjacent word", func(t *testing.T) {
		tokens := tokenizeText("Hey, are you free?")
		require.Len(t, tokens, 4)
		assert.Equal(t, []string{"Hey", "are", "you", "free"}, surfaces(tokens))
		assert.Equal(t, ",", tokens[0].trailing)
		assert.Equal(t, "?", tokens[3].trailing)
		assert.Equal(t, breakNone, tokens[0].break_)
	})

	t.Run("case is preserved in surface but lowercased in normalized", func(t *testing.T) {
		tokens := tokenizeText("THAI")
		require.Len(t, tokens, 1)
		assert.Equal(t, "THAI", tokens[0].surface)
		assert.Equal(t, "thai", tokens[0].normalized)
	})

	t.Run("a blank line is a paragraph break on the first token after it", func(t *testing.T) {
		tokens := tokenizeText("First.\n\nSecond.")
		require.Len(t, tokens, 2)
		assert.Equal(t, breakNone, tokens[0].break_)
		assert.Equal(t, breakParagraph, tokens[1].break_)
	})

	t.Run("a single newline is a line break, distinct from a paragraph break", func(t *testing.T) {
		tokens := tokenizeText("First.\nSecond.")
		require.Len(t, tokens, 2)
		assert.Equal(t, breakLine, tokens[1].break_)
	})

	t.Run("the very first token never carries a break, even with leading blank lines", func(t *testing.T) {
		tokens := tokenizeText("\n\nFirst.")
		require.Len(t, tokens, 1)
		assert.Equal(t, breakNone, tokens[0].break_)
	})

	t.Run("quotes wrap a span: leading on the first word, trailing on the last", func(t *testing.T) {
		tokens := tokenizeText(`He said "hello there" to me.`)
		require.Len(t, tokens, 6)
		assert.Equal(t, []string{"He", "said", "hello", "there", "to", "me"}, surfaces(tokens))
		assert.Equal(t, "\"", tokens[2].leading)
		assert.Equal(t, "", tokens[2].trailing)
		assert.Equal(t, "", tokens[3].leading)
		assert.Equal(t, "\"", tokens[3].trailing)
	})

	t.Run("parentheses wrap a span the same way quotes do", func(t *testing.T) {
		tokens := tokenizeText("case number (four four seven two).")
		require.Len(t, tokens, 6)
		assert.Equal(t, "(", tokens[2].leading)
		assert.Equal(t, ").", tokens[5].trailing)
	})

	t.Run("a hyphenated compound is one token whose surface keeps the hyphen", func(t *testing.T) {
		tokens := tokenizeText("cross-functional team")
		require.Len(t, tokens, 2)
		assert.Equal(t, "cross-functional", tokens[0].surface)
	})

	t.Run("a possessive 's is one token whose surface keeps the apostrophe", func(t *testing.T) {
		tokens := tokenizeText("Jordan's report")
		require.Len(t, tokens, 2)
		assert.Equal(t, "Jordan's", tokens[0].surface)
	})

	t.Run("empty text produces no tokens, not a panic", func(t *testing.T) {
		assert.Empty(t, tokenizeText(""))
	})
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

// normTokens is a small test helper: build dictationTokens with only
// normalized set, for exercising alignTokens (the only field it reads)
// without needing a full tokenizeText round trip.
func normTokens(words ...string) []dictationToken {
	tokens := make([]dictationToken, len(words))
	for i, w := range words {
		tokens[i] = dictationToken{normalized: w}
	}

	return tokens
}

func TestAlignTokens(t *testing.T) {
	t.Run("identical sequences: every position matches, zero edits", func(t *testing.T) {
		ref := normTokens("a", "b", "c")
		edits, ops := alignTokens(ref, ref)
		assert.Equal(t, 0, edits)
		require.Len(t, ops, 3)

		for _, op := range ops {
			assert.Equal(t, opMatch, op.op)
		}
	})

	t.Run("one substitution", func(t *testing.T) {
		ref := normTokens("a", "b", "c")
		hyp := normTokens("a", "x", "c")
		edits, ops := alignTokens(ref, hyp)
		assert.Equal(t, 1, edits)
		require.Len(t, ops, 3)
		assert.Equal(t, opSub, ops[1].op)
		assert.Equal(t, 1, ops[1].refIdx)
		assert.Equal(t, 1, ops[1].hypIdx)
	})

	t.Run("one deletion: a ref word missing from hyp", func(t *testing.T) {
		ref := normTokens("a", "b", "c")
		hyp := normTokens("a", "c")
		edits, ops := alignTokens(ref, hyp)
		assert.Equal(t, 1, edits)
		require.Len(t, ops, 3)
		assert.Equal(t, opDel, ops[1].op)
		assert.Equal(t, 1, ops[1].refIdx)
		assert.Equal(t, -1, ops[1].hypIdx)
	})

	t.Run("one insertion: a hyp word not present in ref", func(t *testing.T) {
		ref := normTokens("a", "c")
		hyp := normTokens("a", "b", "c")
		edits, ops := alignTokens(ref, hyp)
		assert.Equal(t, 1, edits)
		require.Len(t, ops, 3)
		assert.Equal(t, opIns, ops[1].op)
		assert.Equal(t, -1, ops[1].refIdx)
		assert.Equal(t, 1, ops[1].hypIdx)
	})

	t.Run("empty ref: every hyp word is an insertion", func(t *testing.T) {
		edits, ops := alignTokens(nil, normTokens("a", "b"))
		assert.Equal(t, 2, edits)
		require.Len(t, ops, 2)

		for _, op := range ops {
			assert.Equal(t, opIns, op.op)
		}
	})

	t.Run("empty hyp: every ref word is a deletion", func(t *testing.T) {
		edits, ops := alignTokens(normTokens("a", "b"), nil)
		assert.Equal(t, 2, edits)
		require.Len(t, ops, 2)

		for _, op := range ops {
			assert.Equal(t, opDel, op.op)
		}
	})
}

// ---------------------------------------------------------------------------
// Individual metric functions
// ---------------------------------------------------------------------------

func TestExactMatch(t *testing.T) {
	assert.True(t, exactMatch("Hey, world.", "Hey, world."))
	assert.True(t, exactMatch("Hey, world.", "  Hey, world.\n"), "surrounding whitespace is trimmed")
	assert.False(t, exactMatch("Hey, world.", "hey, world."), "case must match exactly")
	assert.False(t, exactMatch("Hey, world.", "Hey world."), "missing punctuation is not an exact match")
}

func TestWordErrorRate(t *testing.T) {
	assert.Equal(t, 0.0, wordErrorRate(0, 0), "no reference words: defined as zero, not a divide-by-zero")
	assert.Equal(t, 0.0, wordErrorRate(5, 0))
	assert.InDelta(t, 20.0, wordErrorRate(5, 1), 1e-9)
	assert.InDelta(t, 125.0, wordErrorRate(4, 5), 1e-9, "more edits than reference words is a real, expected outcome (heavy insertion), not clamped here")
}

func TestPossessiveMarker(t *testing.T) {
	assert.Equal(t, "'s", possessiveMarker("Jordan's"))
	assert.Equal(t, "'s", possessiveMarker("cdc'S"), "case-insensitive")
	assert.Equal(t, "", possessiveMarker("Jordans"))
	assert.Equal(t, "", possessiveMarker("s"), "too short to contain 's -- must not panic")
	assert.Equal(t, "", possessiveMarker(""))
}

func TestCheckAccumulator(t *testing.T) {
	var acc checkAccumulator

	acc.add("", "")   // both empty: not applicable, skipped
	acc.add(",", ",") // applicable, correct
	acc.add(".", "")  // applicable, wrong
	acc.add("", "!")  // applicable (spurious insertion), wrong

	score, ok := acc.percent()
	require.True(t, ok)
	assert.InDelta(t, 100*1.0/3.0, score, 1e-9)

	var empty checkAccumulator
	_, ok = empty.percent()
	assert.False(t, ok, "no applicable position at all must report not-applicable, not a fabricated 100 or 0")
}

// ---------------------------------------------------------------------------
// ScoreDictation -- full metric integration, one dimension isolated per case
// ---------------------------------------------------------------------------

// dictationTruth is a real truth_text (record 0 of dictation/synthetic.jsonl,
// domain "text_message") reused across several cases below: 20 words, four
// terminal-punctuation marks, one all-caps word, and -- deliberately -- no
// quote/paren/hyphen/possessive/paragraph feature at all, which is itself
// something worth asserting (see the "no formatting features" case).
const dictationTruth = "Hey, are you free tonight? I was thinking we could grab dinner at that new THAI place. Let me know!"

var allDictationMetrics = []string{
	"exact_match", "word_error_rate", "punctuation_accuracy", "capitalization_accuracy", "formatting_command_accuracy",
}

func TestScoreDictation_perfectCandidate(t *testing.T) {
	scores := ScoreDictation(dictationTruth, dictationTruth, allDictationMetrics)

	assert.Equal(t, 1.0, scores["exact_match"])
	assert.Equal(t, 0.0, scores["word_error_rate"])
	assert.Equal(t, 100.0, scores["punctuation_accuracy"])
	assert.Equal(t, 100.0, scores["capitalization_accuracy"])
	_, ok := scores["formatting_command_accuracy"]
	assert.False(t, ok, "this truth text has no quote/paren/hyphen/possessive/paragraph feature at all, so the metric is not applicable")
}

func TestScoreDictation_onlyRequestedMetricsAreComputed(t *testing.T) {
	scores := ScoreDictation(dictationTruth, dictationTruth, []string{"exact_match"})
	assert.Equal(t, map[string]float64{"exact_match": 1}, scores)
}

func TestScoreDictation_unrecognizedMetricNameIsIgnoredNotAnError(t *testing.T) {
	scores := ScoreDictation(dictationTruth, dictationTruth, []string{"exact_match", "some_future_metric"})
	assert.Equal(t, map[string]float64{"exact_match": 1}, scores)
}

func TestScoreDictation_wrongWord(t *testing.T) {
	candidate := "Hey, are you free tonight? I was thinking we could lunch dinner at that new THAI place. Let me know!"
	scores := ScoreDictation(dictationTruth, candidate, allDictationMetrics)

	assert.Equal(t, 0.0, scores["exact_match"])
	assert.InDelta(t, 100.0/20.0, scores["word_error_rate"], 1e-9, "one of 20 truth words substituted")
}

func TestScoreDictation_missingWord(t *testing.T) {
	candidate := "Hey, are you free tonight? I was thinking we could dinner at that new THAI place. Let me know!" // "grab" dropped
	scores := ScoreDictation(dictationTruth, candidate, allDictationMetrics)
	assert.InDelta(t, 100.0/20.0, scores["word_error_rate"], 1e-9, "one deletion out of 20 truth words")
}

func TestScoreDictation_extraWord(t *testing.T) {
	candidate := "Hey, are you free tonight? I was really thinking we could grab dinner at that new THAI place. Let me know!" // "really" inserted
	scores := ScoreDictation(dictationTruth, candidate, allDictationMetrics)
	assert.InDelta(t, 100.0/20.0, scores["word_error_rate"], 1e-9, "one insertion, scored against the 20 truth words")
}

func TestScoreDictation_missingPunctuation(t *testing.T) {
	candidate := "Hey are you free tonight? I was thinking we could grab dinner at that new THAI place. Let me know!" // comma after "Hey" dropped
	scores := ScoreDictation(dictationTruth, candidate, allDictationMetrics)

	assert.Equal(t, 0.0, scores["word_error_rate"], "dropping a comma is not a word error")
	assert.InDelta(t, 75.0, scores["punctuation_accuracy"], 1e-9, "3 of the 4 expected punctuation marks still correct")
	assert.Equal(t, 100.0, scores["capitalization_accuracy"], "unaffected by punctuation")
}

func TestScoreDictation_wrongCapitalization(t *testing.T) {
	candidate := "Hey, are you free tonight? I was thinking we could grab dinner at that new Thai place. Let me know!" // THAI -> Thai
	scores := ScoreDictation(dictationTruth, candidate, allDictationMetrics)

	assert.Equal(t, 0.0, scores["word_error_rate"], "case does not count as a word error")
	assert.InDelta(t, 100.0*19.0/20.0, scores["capitalization_accuracy"], 1e-9, "19 of 20 letter-bearing words keep their exact case")
	assert.Equal(t, 100.0, scores["punctuation_accuracy"], "unaffected by capitalization")
}

// formattingTruth exercises every formatting_command_accuracy sub-feature at
// once: a quoted span, a hyphenated compound, a possessive 's, and a
// paragraph break -- five applicable positions in total (the quote's leading
// AND trailing edge count separately).
const formattingTruth = "He said \"hello there\" to the cross-functional team.\n\nJordan's update was clear."

func TestScoreDictation_formattingPerfectCandidate(t *testing.T) {
	scores := ScoreDictation(formattingTruth, formattingTruth, []string{"formatting_command_accuracy"})
	assert.Equal(t, 100.0, scores["formatting_command_accuracy"])
}

func TestScoreDictation_missingQuotes(t *testing.T) {
	candidate := "He said hello there to the cross-functional team.\n\nJordan's update was clear."
	scores := ScoreDictation(formattingTruth, candidate, allDictationMetrics)

	assert.InDelta(t, 60.0, scores["formatting_command_accuracy"], 1e-9, "2 of 5 applicable positions (the quote's leading and trailing edge) now wrong")
	assert.Equal(t, 0.0, scores["word_error_rate"], "removing quote marks doesn't change any word's core")
	assert.Equal(t, 100.0, scores["capitalization_accuracy"])
}

func TestScoreDictation_missingHyphen(t *testing.T) {
	candidate := "He said \"hello there\" to the cross functional team.\n\nJordan's update was clear."
	scores := ScoreDictation(formattingTruth, candidate, []string{"formatting_command_accuracy"})
	assert.InDelta(t, 80.0, scores["formatting_command_accuracy"], 1e-9, "only the hyphen check now fails; quotes, possessive, and paragraph break are untouched")
}

func TestScoreDictation_missingPossessive(t *testing.T) {
	candidate := "He said \"hello there\" to the cross-functional team.\n\nJordans update was clear."
	scores := ScoreDictation(formattingTruth, candidate, allDictationMetrics)

	assert.InDelta(t, 80.0, scores["formatting_command_accuracy"], 1e-9, "only the possessive check now fails")
	assert.InDelta(t, 100.0*11.0/12.0, scores["capitalization_accuracy"], 1e-9, "the substituted word (Jordan's -> Jordans) can't have correct capitalization either")
}

func TestScoreDictation_lineBreakInsteadOfParagraphBreak(t *testing.T) {
	candidate := "He said \"hello there\" to the cross-functional team.\nJordan's update was clear."
	scores := ScoreDictation(formattingTruth, candidate, []string{"formatting_command_accuracy"})
	assert.InDelta(t, 80.0, scores["formatting_command_accuracy"], 1e-9, "a single newline is a LINE break, distinct from the truth's paragraph break")
}

// ---------------------------------------------------------------------------
// runDictationOne / runDictationForModel
// ---------------------------------------------------------------------------

func TestRunDictationOne_sendsSharedPromptAsSystemPromptAndScoresAgainstTruth(t *testing.T) {
	const rawInput = "hey comma world period"
	const candidateOutput = "Hey, world."
	const truth = "Hey, world."

	spec, ok := prompts.For(prompts.Dictation)
	require.True(t, ok)
	require.NotEmpty(t, spec.Task, "dictation must have a real fixed system prompt to send")

	var requestBody string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestBody = readBody(t, r)
		writeChatCompletion(w, candidateOutput, 10, 5)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)

	// DictationRecord has no field for a per-record instruction at all (see
	// rawDictationRecord's doc comment) -- there is nothing here to
	// accidentally send instead of the shared prompt.
	record := DictationRecord{
		ID: "rec-1", Bucket: "30s",
		Input: rawInput, TruthText: truth,
		Metrics: []string{"exact_match", "word_error_rate"},
	}

	sample := runDictationOne(context.Background(), gw, ModelInfo{ID: "candidate-model"}, spec, record, nil)

	require.NoError(t, sample.Err)
	assert.Equal(t, prompts.Dictation, sample.Task)
	assert.Equal(t, "30s", sample.Bucket)
	assert.Equal(t, "rec-1", sample.RecordID)

	var req failureRequestBody
	require.NoError(t, json.Unmarshal([]byte(requestBody), &req))
	assert.Equal(t, "candidate-model", req.Model)
	require.Len(t, req.Messages, 2)
	assert.Equal(t, "system", req.Messages[0].Role)
	assert.Equal(t, spec.Task, req.Messages[0].Content, "the shared dictation prompt must be sent as the system prompt, the same for every record")
	assert.Equal(t, "user", req.Messages[1].Role)
	assert.Equal(t, rawInput, req.Messages[1].Content)

	assert.Equal(t, map[string]float64{"exact_match": 1, "word_error_rate": 0}, sample.Judge)
}

func TestRunDictationOne_recordsAPIErrorFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	failures, err := NewFailureRecorder(filepath.Join(t.TempDir(), "errors"))
	require.NoError(t, err)

	spec, ok := prompts.For(prompts.Dictation)
	require.True(t, ok)

	record := DictationRecord{ID: "rec-1", Bucket: "30s", Input: "hello", TruthText: "Hello."}

	sample := runDictationOne(context.Background(), gw, ModelInfo{ID: "candidate-model"}, spec, record, failures)

	require.Error(t, sample.Err)
	require.Equal(t, 1, failures.Count())

	meta := readFailureMeta(t, failures.RunDir())
	require.Len(t, meta, 1)
	assert.Equal(t, prompts.Dictation, meta[0].Task)
	assert.Equal(t, "rec-1", meta[0].RecordID)
	assert.Equal(t, FailureAPIError, meta[0].Reason)
}

func TestRunDictationOne_contextLengthExceededSkipsTheCallEntirely(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeChatCompletion(w, "should never be requested", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)

	spec, ok := prompts.For(prompts.Dictation)
	require.True(t, ok)

	record := DictationRecord{
		ID: "rec-1", Bucket: "30s",
		Input: strings.Repeat("word ", 1000), TruthText: "Hello.",
	}

	sample := runDictationOne(context.Background(), gw, ModelInfo{ID: "candidate-model", ContextLength: 10}, spec, record, nil)

	assert.Error(t, sample.Err)
	assert.False(t, called, "no HTTP call should happen once the context-length guard trips")
}

func TestRunDictationForModel_runsEveryRecordAndPoolsSamples(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "Hey.", 5, 2)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)

	records := []DictationRecord{
		{ID: "r1", Bucket: "30s", Input: "hey period", TruthText: "Hey.", Metrics: []string{"exact_match"}},
		{ID: "r2", Bucket: "30s", Input: "hey period", TruthText: "Hey.", Metrics: []string{"exact_match"}},
		{ID: "r3", Bucket: "120s", Input: "hey period", TruthText: "Hey.", Metrics: []string{"exact_match"}},
	}

	samples := runDictationForModel(context.Background(), gw, ModelInfo{ID: "candidate-model"}, records, 2, nil)

	require.Len(t, samples, 3)

	ids := make([]string, len(samples))
	for i, s := range samples {
		ids[i] = s.RecordID
		assert.Equal(t, prompts.Dictation, s.Task)
		assert.Equal(t, map[string]float64{"exact_match": 1}, s.Judge)
	}

	assert.ElementsMatch(t, []string{"r1", "r2", "r3"}, ids)
}
