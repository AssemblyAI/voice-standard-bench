package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// cells returns one row of a view as plain strings, brackets included -- what
// a reader would see in that row, column by column.
func cells(v correctionView, row func(viewColumn) viewCell) []string {
	out := make([]string, len(v.columns))
	for i, c := range v.columns {
		out[i] = row(c).display()
	}

	return out
}

func truthRow(c viewColumn) viewCell  { return c.truth }
func inputRow(c viewColumn) viewCell  { return c.input }
func outputRow(c viewColumn) viewCell { return c.output }

func TestBuildCorrectionView_bracketsWhatDiffersFromTheTruth(t *testing.T) {
	view, ok := buildCorrectionView(
		"we agreed to ship the budget review on friday",
		"we agreed to shop the budjet review on um friday",
		"we agreed to ship the budjet review on friday",
	)

	require.True(t, ok)
	assert.Equal(t, 9, view.truthWords)
	assert.Equal(t, 3, view.inputEdits, "shop, budjet, and the inserted um")
	assert.Equal(t, 1, view.outputEdits, "budjet was left unfixed; ship and the removal of um were right")

	assert.Equal(t, []string{"we", "agreed", "to", "ship", "the", "budget", "review", "on", debugGap, "friday"}, cells(view, truthRow))
	assert.Equal(t, []string{"we", "agreed", "to", "[shop]", "the", "[budjet]", "review", "on", "[um]", "friday"}, cells(view, inputRow))
	assert.Equal(t, []string{"we", "agreed", "to", "ship", "the", "[budjet]", "review", "on", debugGap, "friday"}, cells(view, outputRow))
}

// Case and punctuation are shown as written but are not errors: the score
// ignores them, so the view must not bracket them.
func TestBuildCorrectionView_showsStyleDifferencesWithoutCallingThemErrors(t *testing.T) {
	view, ok := buildCorrectionView("We shipped it on Friday.", "we shipped it on friday", "We shipped it on Friday!")

	require.True(t, ok)
	assert.Equal(t, 0, view.inputEdits)
	assert.Equal(t, 0, view.outputEdits)
	assert.Equal(t, []string{"We", "shipped", "it", "on", "Friday."}, cells(view, truthRow))
	assert.Equal(t, []string{"we", "shipped", "it", "on", "friday"}, cells(view, inputRow), "visibly different text, but no brackets")
	assert.Equal(t, []string{"We", "shipped", "it", "on", "Friday!"}, cells(view, outputRow))
}

func TestBuildCorrectionView_aTruthWordTheOutputDroppedShowsABracketedGap(t *testing.T) {
	view, ok := buildCorrectionView("so um we agreed to ship", "so um we agreed to ship", "so we agreed to ship")

	require.True(t, ok)
	assert.Equal(t, 0, view.inputEdits)
	assert.Equal(t, 1, view.outputEdits)
	assert.Equal(t, []string{"so", "um", "we", "agreed", "to", "ship"}, cells(view, truthRow))
	assert.Equal(t, []string{"so", "[" + debugGap + "]", "we", "agreed", "to", "ship"}[1], cells(view, outputRow)[1], "the model deleted a word the truth has")
}

// When both rows inserted words at the same spot, a word the model left in
// place must share a column with the input's copy of it -- not be zipped with
// whatever happens to be at the same position.
func TestBuildCorrectionView_insertionsAreAlignedToEachOther(t *testing.T) {
	view, ok := buildCorrectionView(
		"we ship friday",
		"we um uh ship friday",
		"we uh ship friday", // removed "um", left "uh"
	)

	require.True(t, ok)
	assert.Equal(t, []string{"we", debugGap, debugGap, "ship", "friday"}, cells(view, truthRow))
	assert.Equal(t, []string{"we", "[um]", "[uh]", "ship", "friday"}, cells(view, inputRow))
	assert.Equal(t, []string{"we", debugGap, "[uh]", "ship", "friday"}, cells(view, outputRow), "the leftover uh lines up under the input's uh")
	assert.Equal(t, 2, view.inputEdits)
	assert.Equal(t, 1, view.outputEdits)
}

func TestBuildCorrectionView_wordsInsertedAtTheEndsAreKept(t *testing.T) {
	view, ok := buildCorrectionView("the meeting", "the meeting", "well the meeting is at three")

	require.True(t, ok)
	assert.Equal(t, []string{debugGap, "the", "meeting", debugGap, debugGap, debugGap}, cells(view, truthRow))
	assert.Equal(t, []string{debugGap, "the", "meeting", debugGap, debugGap, debugGap}, cells(view, inputRow))
	assert.Equal(t, []string{"[well]", "the", "meeting", "[is]", "[at]", "[three]"}, cells(view, outputRow))
	assert.Equal(t, 4, view.outputEdits)
}

// The invariant the whole view rests on: its bracket counts are the score's
// edit counts, because both come from the same normalized words and the same
// optimal alignment. If these ever diverged, -debug would be showing something
// other than what was scored.
func TestBuildCorrectionView_bracketCountsAreTheScoresEditCounts(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	vocab := []string{"the", "meeting", "is", "at", "three", "we", "agreed", "um", "Friday,", "friday", "all-hands", "5", "five", "—"}

	randomText := func(n int) string {
		parts := make([]string, n)
		for i := range parts {
			parts[i] = vocab[rng.Intn(len(vocab))]
		}

		return strings.Join(parts, " ")
	}

	for trial := 0; trial < 600; trial++ {
		truth, input, output := randomText(rng.Intn(40)), randomText(rng.Intn(40)), randomText(rng.Intn(60))

		view, ok := buildCorrectionView(truth, input, output)
		require.True(t, ok)

		score := ScoreCorrection(truth, input, output)

		require.Equal(t, score.TruthWords, view.truthWords, "trial %d", trial)
		require.Equal(t, score.InputEdits, view.inputEdits, "trial %d: truth=%q input=%q", trial, truth, input)
		require.Equal(t, score.OutputEdits, view.outputEdits, "trial %d: truth=%q output=%q", trial, truth, output)

		// And the truth row, read left to right, is the truth's words in order.
		var truthWords []string

		for _, c := range view.columns {
			if c.truth.text != debugGap || c.firstWord != 0 {
				truthWords = append(truthWords, c.truth.text)
			}
		}

		var want []string
		for _, w := range correctionWords(truth) {
			want = append(want, w.Surface)
		}

		require.Equal(t, want, truthWords, "trial %d", trial)
	}
}

func TestBuildCorrectionView_tooLargeToAlignReportsNotOK(t *testing.T) {
	original := alignMaxCells
	alignMaxCells = 10

	t.Cleanup(func() { alignMaxCells = original })

	view, ok := buildCorrectionView(strings.Repeat("a ", 50), strings.Repeat("b ", 50), strings.Repeat("c ", 50))

	assert.False(t, ok)
	assert.Equal(t, 50, view.truthWords)
	assert.Empty(t, view.columns)
}

func TestFormatCorrectionView_alignsColumnsAndLabelsTheRows(t *testing.T) {
	view, ok := buildCorrectionView(
		"we agreed to ship the budget review on friday",
		"we agreed to shop the budjet review on um friday",
		"we agreed to ship the budjet review on friday",
	)
	require.True(t, ok)

	want := "(truth words 1-9)\n" +
		"TRUTH  we  agreed  to  ship    the  budget    review  on  ·     friday\n" +
		"INPUT  we  agreed  to  [shop]  the  [budjet]  review  on  [um]  friday\n" +
		"OUTPUT we  agreed  to  ship    the  [budjet]  review  on  ·     friday\n" +
		"\n"

	assert.Equal(t, want, formatCorrectionView(view, debugViewWidth))
}

func TestFormatCorrectionView_wrapsAtTheWidthKeepingEveryRowInStep(t *testing.T) {
	view, ok := buildCorrectionView(
		"we agreed to ship the budget review on friday",
		"we agreed to shop the budjet review on um friday",
		"we agreed to ship the budjet review on friday",
	)
	require.True(t, ok)

	want := "(truth words 1-5)\n" +
		"TRUTH  we  agreed  to  ship    the\n" +
		"INPUT  we  agreed  to  [shop]  the\n" +
		"OUTPUT we  agreed  to  ship    the\n" +
		"\n" +
		"(truth words 6-8)\n" +
		"TRUTH  budget    review  on  ·\n" +
		"INPUT  [budjet]  review  on  [um]\n" +
		"OUTPUT [budjet]  review  on  ·\n" +
		"\n" +
		"(truth word 9)\n" +
		"TRUTH  friday\n" +
		"INPUT  friday\n" +
		"OUTPUT friday\n" +
		"\n"

	got := formatCorrectionView(view, 40)
	assert.Equal(t, want, got)

	for _, line := range strings.Split(got, "\n") {
		assert.LessOrEqual(t, len([]rune(line)), 40, "no line wider than the requested width: %q", line)
	}
}

func TestFormatCorrectionView_aColumnWiderThanTheLineStillPrints(t *testing.T) {
	view, ok := buildCorrectionView("a supercalifragilisticexpialidocious b", "a supercalifragilisticexpialidocious b", "a supercalifragilisticexpialidocious b")
	require.True(t, ok)

	got := formatCorrectionView(view, 10)

	assert.Contains(t, got, "supercalifragilisticexpialidocious", "an over-wide word is never dropped or truncated")
	assert.Equal(t, 3, strings.Count(got, "TRUTH"), "one block per column when none fit alongside another")
}

func TestCollapseIdentical(t *testing.T) {
	same := func(word string, n int) viewColumn {
		return viewColumn{truth: viewCell{text: word}, input: viewCell{text: word}, output: viewCell{text: word}, firstWord: n, lastWord: n, same: true}
	}
	differs := func(n int) viewColumn {
		return viewColumn{truth: viewCell{text: "x"}, input: viewCell{text: "y", wrong: true}, output: viewCell{text: "x"}, firstWord: n, lastWord: n}
	}

	columns := func(n int) []viewColumn {
		var cols []viewColumn

		cols = append(cols, differs(1))
		for i := 2; i < 2+n; i++ {
			cols = append(cols, same("w", i))
		}

		return append(cols, differs(2+n))
	}

	t.Run("a long identical run keeps context on both sides and folds the middle", func(t *testing.T) {
		got := collapseIdentical(columns(100), 40, 8)

		// differs + 8 context + marker + 8 context + differs
		require.Len(t, got, 1+8+1+8+1)

		marker := got[9]
		assert.Equal(t, "… 84 identical words …", marker.truth.text, "100 identical words minus 8 kept on each side")
		assert.Equal(t, 10, marker.firstWord, "the marker stands in for truth words 10-93")
		assert.Equal(t, 93, marker.lastWord)
		assert.Equal(t, 94, got[10].firstWord, "and the context after it resumes right where it left off")
		assert.Equal(t, 102, got[len(got)-1].firstWord)
	})

	t.Run("a run shorter than the threshold is left whole", func(t *testing.T) {
		assert.Len(t, collapseIdentical(columns(39), 40, 8), 1+39+1)
	})

	t.Run("a run the context would swallow is left whole", func(t *testing.T) {
		assert.Len(t, collapseIdentical(columns(50), 40, 30), 1+50+1, "nothing would be hidden, so nothing is folded")
	})

	t.Run("columns that differ are never folded", func(t *testing.T) {
		var cols []viewColumn
		for i := 1; i <= 100; i++ {
			cols = append(cols, differs(i))
		}

		assert.Len(t, collapseIdentical(cols, 40, 8), 100)
	})
}

func TestFormatCorrectionView_foldsALongIdenticalStretchAndKeepsWordNumbersTrue(t *testing.T) {
	filler := strings.Repeat("alpha beta gamma delta ", 25) // 100 words, identical in all rows
	view, ok := buildCorrectionView("start "+filler+"end", "start "+filler+"edn", "start "+filler+"end")
	require.True(t, ok)

	got := formatCorrectionView(view, debugViewWidth)

	assert.Contains(t, got, "… 85 identical words …", "101 same columns minus 8 context on each side")
	assert.Contains(t, got, "[edn]", "the one real difference survives the folding")
	assert.Contains(t, got, "(truth words ", "blocks still say which truth words they cover")
	assert.NotContains(t, got, strings.Repeat("alpha beta gamma delta ", 5), "the folded middle is not printed")
	assert.Contains(t, got, "102", "the last block's numbering counts the folded words: the final truth word is #102")
}

func TestFormatScores(t *testing.T) {
	assert.Equal(t, "faithfulness=5  input_wer=37.5  meaning_similarity=4  truth_wer=21.88", formatScores(map[string]float64{
		"truth_wer": 21.875, "faithfulness": 5, "meaning_similarity": 4, "input_wer": 37.5,
	}), "name order, four significant digits")
}

func TestFormatDebugSample(t *testing.T) {
	const (
		truth  = "we agreed to ship the budget review on friday"
		input  = "we agreed to shop the budjet review on um friday"
		output = "we agreed to ship the budjet review on friday"
	)

	record := TranscriptRecord{ID: "rec-7", Bucket: "30s", Text: truth, MissedText: input}
	correction := ScoreCorrection(truth, input, output)
	scores := map[string]float64{"faithfulness": 5, "improvement": 4, "meaning_similarity": 4}

	for name, v := range correction.Scores() {
		scores[name] = v
	}

	got := formatDebugSample(ModelInfo{ID: "candidate-model"}, record, input, output, correction, scores, "the output still says budjet where the reference says budget")

	assert.Contains(t, got, "text_correction debug  model=candidate-model  reasoning_effort=default  bucket=30s  record=rec-7")
	assert.Contains(t, got, "scores:  faithfulness=5  improvement=4  input_wer=33.33  meaning_similarity=4  truth_wer=11.11")
	assert.Contains(t, got, "judge:   the output still says budjet where the reference says budget", "the meaning judge's note sits right under the scores")
	assert.Contains(t, got, "TRUTH    9 words -- the original text, the answer key (never shown to the model)")
	assert.Contains(t, got, "INPUT    3 edits from the truth, word for word (33.33% WER) -- the corrupted text the model was shown")
	assert.Contains(t, got, "OUTPUT   1 edit from the truth, word for word (11.11% WER) -- the model's corrected text", "singular for one")
	assert.Contains(t, got, "legend")
	assert.Contains(t, got, "the judge scores meaning, so fillers, stutters and number formatting show here without costing the model")
	assert.Contains(t, got, "INPUT  we  agreed  to  [shop]  the  [budjet]  review  on  [um]  friday")
	assert.Contains(t, got, "OUTPUT we  agreed  to  ship    the  [budjet]  review  on  ·     friday")
	assert.NotContains(t, got, "WARNING", "the view and the word-level counts agree")
}

func TestFormatDebugSample_judgeNoteIsOptionalAndOneLine(t *testing.T) {
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "one two three"}
	correction := ScoreCorrection(record.Text, "one two three", "one two three")

	t.Run("no note, no judge line", func(t *testing.T) {
		got := formatDebugSample(ModelInfo{ID: "m"}, record, "one two three", "one two three", correction, correction.Scores(), "")

		assert.NotContains(t, got, "judge:")
	})

	t.Run("a multi-line note is folded onto one line", func(t *testing.T) {
		got := formatDebugSample(ModelInfo{ID: "m"}, record, "one two three", "one two three", correction, correction.Scores(), "first line\n  second   line\n")

		assert.Contains(t, got, "judge:   first line second line\n")
	})
}

// If the aligned view and the score ever disagree about how many edits there
// are, say so loudly rather than quietly showing a misleading picture -- the
// whole point of -debug is to be trusted when the score looks wrong.
func TestFormatDebugSample_warnsWhenTheViewAndTheScoreDisagree(t *testing.T) {
	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "one two three"}
	wrong := CorrectionResult{TruthWords: 3, InputEdits: 0, OutputEdits: 0} // claims a perfect output

	got := formatDebugSample(ModelInfo{ID: "m"}, record, "one two three", "one two four", wrong, wrong.Scores(), "")

	assert.Contains(t, got, "WARNING: the view below counts 0/1 edits (input/output) but the score counted 0/0")
}

func TestFormatDebugSample_fallsBackWhenTheTextsAreTooLargeToAlign(t *testing.T) {
	original := alignMaxCells
	alignMaxCells = 10

	t.Cleanup(func() { alignMaxCells = original })

	truth, input, output := strings.Repeat("a ", 100), strings.Repeat("b ", 100), strings.Repeat("c ", 100)
	record := TranscriptRecord{ID: "rec-1", Bucket: "900s", Text: truth}

	got := formatDebugSample(ModelInfo{ID: "m"}, record, input, output, ScoreCorrection(truth, input, output), nil, "")

	assert.Contains(t, got, "alignment skipped")
	assert.Contains(t, got, "first 40 words of the output:")
	assert.Contains(t, got, "c c c c", "a glance at what the model returned is still given")
	assert.NotContains(t, got, "(truth word", "no aligned blocks")
}

// ---------------------------------------------------------------------------
// The task: -debug on and off, and concurrent samples
// ---------------------------------------------------------------------------

func debugJudgeServer(t *testing.T) *Gateway {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, textCorrectionJudgeReply(readBody(t, r), 4, 3, 4), 20, 10)
	}))
	t.Cleanup(server.Close)

	return NewGateway(server.URL, "test-key", 5*time.Second)
}

func TestNewTextCorrectionTask_debugFlagSelectsStandardOutput(t *testing.T) {
	off, err := NewTextCorrectionTask(nil, false)
	require.NoError(t, err)
	assert.Nil(t, off.debugOut, "-debug off: nothing is ever written")

	on, err := NewTextCorrectionTask(nil, true)
	require.NoError(t, err)
	assert.Same(t, os.Stdout, on.debugOut, "-debug on: the review goes to standard output")
}

func TestTextCorrectionTaskRunJudgePrintsTheAlignedReviewInDebugMode(t *testing.T) {
	task := newTestTextCorrectionTask(t, debugJudgeServer(t))

	var out bytes.Buffer

	task.debugOut = &out

	record := TranscriptRecord{
		ID: "rec-1", Bucket: "30s",
		Text:       "we agreed to ship the budget review on friday",
		MissedText: "we agreed to shop the budjet review on um friday",
	}

	judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate-model"}, record, "judge-model",
		CompletionResult{Text: "we agreed to ship the budjet review on friday"}, nil)
	require.NoError(t, err)

	got := out.String()

	assert.Contains(t, got, "model=candidate-model")
	assert.Contains(t, got, "record=rec-1")
	assert.Contains(t, got, "TRUTH  we  agreed  to  ship    the  budget    review  on  ·     friday")
	assert.Contains(t, got, "INPUT  we  agreed  to  [shop]  the  [budjet]  review  on  [um]  friday")
	assert.Contains(t, got, "OUTPUT we  agreed  to  ship    the  [budjet]  review  on  ·     friday")
	assert.Contains(t, got, "meaning_similarity=4", "the printed score is the recorded one")
	assert.Contains(t, got, "judge:   the meaning note", "and so is the meaning judge's note")
	assert.Equal(t, 4.0, judged.Judge["meaning_similarity"])
}

func TestTextCorrectionTaskRunJudgePrintsNothingWithoutDebugMode(t *testing.T) {
	task := newTestTextCorrectionTask(t, debugJudgeServer(t))
	require.Nil(t, task.debugOut)

	record := TranscriptRecord{ID: "rec-1", Bucket: "30s", Text: "the meeting is at three", MissedText: "the meating is at three"}

	// Anything written to the real standard output would be a bug; there is
	// nowhere else for output to go, so the strongest check is that debugSample
	// is a no-op that cannot panic or write on a nil writer.
	assert.NotPanics(t, func() {
		_, err := task.RunJudge(context.Background(), ModelInfo{ID: "m"}, record, "judge-model", CompletionResult{Text: "the meeting is at three"}, nil)
		assert.NoError(t, err)
	})
}

// RunJudge runs on many goroutines at once; a sample's multi-line block must
// come out whole, never interleaved with another's.
func TestTextCorrectionTaskDebugBlocksNeverInterleave(t *testing.T) {
	task := newTestTextCorrectionTask(t, nil)

	var out bytes.Buffer

	task.debugOut = &out

	const samples = 40

	var wg sync.WaitGroup

	for i := 0; i < samples; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			text := fmt.Sprintf("sample %02d is a short test transcript", i)
			record := TranscriptRecord{ID: fmt.Sprintf("rec-%02d", i), Bucket: "30s", Text: text}
			correction := ScoreCorrection(text, text, text)

			task.debugSample(ModelInfo{ID: "m"}, record, text, text, correction, correction.Scores(), "")
		}(i)
	}

	wg.Wait()

	blocks := strings.Split(strings.TrimSpace(out.String()), strings.Repeat("=", debugViewWidth))
	blocks = blocks[1:] // the text before the first separator is empty

	require.Len(t, blocks, samples)

	seen := map[string]bool{}

	for _, block := range blocks {
		require.Equal(t, 1, strings.Count(block, "text_correction debug"), "exactly one sample per block")

		var id string

		_, err := fmt.Sscanf(block[strings.Index(block, "record=")+len("record="):], "%s", &id)
		require.NoError(t, err)

		// The three rows (TRUTH, INPUT, OUTPUT) of that block must all be about
		// this one sample, and about no other: the rows render each word in its
		// own column, two spaces apart.
		number := strings.TrimPrefix(id, "rec-")

		assert.Equal(t, 3, strings.Count(block, fmt.Sprintf("sample  %s  is", number)), "block for %s is missing a row of its own", id)
		assert.Equal(t, 3, strings.Count(block, "sample  "), "block for %s mixes in another sample's rows", id)

		seen[id] = true
	}

	assert.Len(t, seen, samples, "every sample's block printed exactly once")
}

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what
// was written to it. The reader runs concurrently so output larger than the
// pipe's buffer can't block fn.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	reader, writer, err := os.Pipe()
	require.NoError(t, err)

	original := os.Stdout
	os.Stdout = writer

	captured := make(chan string)

	go func() {
		data, _ := io.ReadAll(reader)
		captured <- string(data)
	}()

	func() {
		defer func() {
			os.Stdout = original
			_ = writer.Close()
		}()

		fn()
	}()

	return <-captured
}

// runTextCorrectionPass runs one text_correction record through the full
// framework run (a candidate that fixes one of two corrupted words), with the
// given debug setting, and returns everything printed to standard output.
func runTextCorrectionPass(t *testing.T, debug bool) string {
	t.Helper()

	t.Chdir(t.TempDir())
	writeFile(t, fmt.Sprintf("%s/text_correction/aai_transcripts.jsonl", normalizedRoot),
		recordLineWithMissedText("r1", "30s", "we agreed to ship the budget review on friday", "we agreed to shop the budjet review on friday"))

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		switch {
		case strings.Contains(body, `"model":"the-judge"`):
			writeChatCompletion(w, textCorrectionJudgeReply(body, 4, 4, 4), 20, 10)
		case strings.Contains(body, `"stream":true`):
			writeChatCompletionStream(w, []string{"we agreed to ship the budjet review on friday"}, 10, 5)
		default:
			writeChatCompletion(w, "we agreed to ship the budjet review on friday", 10, 5)
		}
	}))
	defer server.Close()

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "text_correction",
		JudgeModel:          "the-judge",
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
		Debug:               debug,
	}

	return captureStdout(t, func() {
		runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))
	})
}

// The flag's whole journey: frameworkPassConfig.Debug -> RunConfig.Debug ->
// NewTextCorrectionTask -> RunJudge -> standard output.
func TestRunFrameworkBenchmarkPass_debugPrintsTheAlignedReviewForTextCorrection(t *testing.T) {
	got := runTextCorrectionPass(t, true)

	assert.Contains(t, got, "text_correction debug  model=candidate-model")
	assert.Contains(t, got, "TRUTH  we  agreed  to  ship    the  budget    review  on  friday")
	assert.Contains(t, got, "INPUT  we  agreed  to  [shop]  the  [budjet]  review  on  friday")
	assert.Contains(t, got, "OUTPUT we  agreed  to  ship    the  [budjet]  review  on  friday")
	assert.Equal(t, 1, strings.Count(got, "text_correction debug"), "once per judged sample -- the TTFT pass never judges, so never prints")
}

func TestRunFrameworkBenchmarkPass_withoutDebugNothingExtraIsPrinted(t *testing.T) {
	got := runTextCorrectionPass(t, false)

	assert.NotContains(t, got, "text_correction debug")
	assert.NotContains(t, got, "TRUTH  ")
	assert.Contains(t, got, "=== text_correction ===", "the normal report is untouched")
}

func TestRunFrameworkBenchmarkPass_debugWithoutTextCorrectionSaysSo(t *testing.T) {
	writeFrameworkFixtures(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"candidate-model","name":"Candidate","creator":"Acme"}]}`))

			return
		}

		body := readBody(t, r)

		switch {
		case strings.Contains(body, `"model":"the-judge"`):
			writeChatCompletion(w, `{"faithfulness": 5, "coverage": 5, "conciseness": 5, "notes": "ok"}`, 20, 10)
		case strings.Contains(body, `"stream":true`):
			writeChatCompletionStream(w, []string{"a summary"}, 10, 5)
		default:
			writeChatCompletion(w, "a summary", 10, 5)
		}
	}))
	defer server.Close()

	var logs bytes.Buffer

	log.SetOutput(&logs)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	cfg := frameworkPassConfig{
		GatewayURL:          server.URL,
		APIKey:              "test-key",
		Timeout:             5 * time.Second,
		TasksFlag:           "summarization",
		JudgeModel:          "the-judge",
		MaxSamplesPerBucket: -1,
		Concurrency:         2,
		Debug:               true,
	}

	_ = captureStdout(t, func() {
		runFrameworkBenchmarkPass(context.Background(), runFrameworkTestCanonicalConfig(t, cfg))
	})

	assert.Contains(t, logs.String(), "-debug only prints for text_correction, which isn't in -tasks")
}
