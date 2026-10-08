package main

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCorrectionTokens(t *testing.T) {
	tests := []struct {
		name string
		text string
		want []string
	}{
		{"lowercases", "Hello World", []string{"hello", "world"}},
		{"strips attached punctuation", `"Well," she said.`, []string{"well", "she", "said"}},
		{"straight and curly apostrophes normalize to the same word", "don't don’t", []string{"dont", "dont"}},
		{"a hyphenated compound is one word", "all-hands meeting", []string{"allhands", "meeting"}},
		{"standalone punctuation is not a word", "wait — what … really", []string{"wait", "what", "really"}},
		{"digits are word characters", "at 3:30 for $5,000", []string{"at", "330", "for", "5000"}},
		{"non-ASCII letters survive and lowercase", "Ångström été", []string{"ångström", "été"}},
		{"newlines and runs of spaces are just whitespace", "one\n\ntwo   three", []string{"one", "two", "three"}},
		{"empty", "", []string{}},
		{"only punctuation", "— … ?!", []string{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, correctionTokens(tt.text))
		})
	}
}

func TestScoreCorrection(t *testing.T) {
	const truth = "we agreed to ship the budget review on friday after the all-hands"

	tests := []struct {
		name      string
		input     string
		candidate string

		wantInputEdits  int
		wantOutputEdits int
	}{
		{
			name:            "a perfect correction leaves no errors",
			input:           "we agreed to shop the budjet review on friday after the all-hands",
			candidate:       truth,
			wantInputEdits:  2,
			wantOutputEdits: 0,
		},
		{
			name:            "doing nothing leaves every error",
			input:           "we agreed to shop the budjet review on friday after the all-hands",
			candidate:       "we agreed to shop the budjet review on friday after the all-hands",
			wantInputEdits:  2,
			wantOutputEdits: 2,
		},
		{
			name:            "fixing one of two leaves one",
			input:           "we agreed to shop the budjet review on friday after the all-hands",
			candidate:       "we agreed to ship the budjet review on friday after the all-hands",
			wantInputEdits:  2,
			wantOutputEdits: 1,
		},
		{
			name:            "removing an inserted word counts as a fix",
			input:           "we agreed to also ship the budget review on friday after the all-hands",
			candidate:       truth,
			wantInputEdits:  1,
			wantOutputEdits: 0,
		},
		{
			name:            "case and punctuation are not words: a re-styled perfect correction has no errors",
			input:           "we agreed to shop the budget review on friday after the all-hands",
			candidate:       "We agreed to ship the budget review on Friday, after the all-hands.",
			wantInputEdits:  1,
			wantOutputEdits: 0,
		},
		{
			name:            "making it worse than the input leaves more errors than it started with",
			input:           "we agreed to shop the budget review on friday after the all-hands",
			candidate:       "we agreed to shop the budjet reviw on fryday after the all-hands",
			wantInputEdits:  1,
			wantOutputEdits: 4,
		},
		{
			name:            "an input that was already clean, left alone, has no errors",
			input:           truth,
			candidate:       truth,
			wantInputEdits:  0,
			wantOutputEdits: 0,
		},
		{
			name:            "an empty candidate loses every word",
			input:           "we agreed to shop the budget review on friday after the all-hands",
			candidate:       "",
			wantInputEdits:  1,
			wantOutputEdits: 12,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ScoreCorrection(truth, tt.input, tt.candidate)

			assert.Equal(t, 12, got.TruthWords)
			assert.Equal(t, tt.wantInputEdits, got.InputEdits, "InputEdits")
			assert.Equal(t, tt.wantOutputEdits, got.OutputEdits, "OutputEdits")
		})
	}
}

func TestCorrectionResult_derivedScores(t *testing.T) {
	r := CorrectionResult{TruthWords: 200, InputEdits: 8, OutputEdits: 2}

	assert.InDelta(t, 1.0, r.TruthWER(), 1e-9)
	assert.InDelta(t, 4.0, r.InputWER(), 1e-9)
	assert.InDelta(t, 0.99, r.WordAccuracy(), 1e-9)

	assert.Equal(t, map[string]float64{
		scoreTruthWER: 1.0,
		scoreInputWER: 4.0,
	}, r.Scores(), "context only: neither is a quality dimension")

	t.Run("a runaway output has more errors than the truth has words", func(t *testing.T) {
		runaway := CorrectionResult{TruthWords: 100, InputEdits: 4, OutputEdits: 500}

		assert.InDelta(t, 500.0, runaway.TruthWER(), 1e-9)
		assert.InDelta(t, -4.0, runaway.WordAccuracy(), 1e-9, "word accuracy goes negative rather than clamping, so the failure floor still trips")
	})
}

// The word-level numbers are context, not the quality score, and these two
// tests are why. The corruption is light, so even doing nothing is within a few
// percent of the truth; and the truth is a raw transcript, so doing exactly what
// the task prompt says -- drop the fillers, leave the numbers as spoken --
// moves a model AWAY from it. Both are what the meaning judge (RunReferenceJudge)
// is there to see through.
func TestScoreCorrection_wordLevelNumbersCannotBeTheQualityScore(t *testing.T) {
	t.Run("doing nothing is within a few percent of the truth", func(t *testing.T) {
		truthWords := make([]string, 0, 200)
		for i := 0; i < 200; i++ {
			truthWords = append(truthWords, "w"+strings.Repeat("x", i%7)+string(rune('a'+i%26)))
		}

		corrupted := append([]string(nil), truthWords...)
		for _, i := range []int{10, 45, 80, 120, 160, 190} {
			corrupted[i] = "garbled"
		}

		got := ScoreCorrection(strings.Join(truthWords, " "), strings.Join(corrupted, " "), strings.Join(corrupted, " "))

		assert.Equal(t, 6, got.InputEdits)
		assert.InDelta(t, 0.97, got.WordAccuracy(), 1e-9, "a do-nothing model is 97% word-accurate")
	})

	t.Run("a model that follows the prompt is marked down against a raw-transcript truth", func(t *testing.T) {
		// The truth keeps its fillers and writes the numeral; the model is shown
		// the number spelled out and is told to drop fillers and leave it be.
		truth := "we have um 5 people on the call uh today"
		shown := "we have um five people on the call uh today"
		obedient := "we have five people on the call today"

		got := ScoreCorrection(truth, shown, obedient)

		assert.Equal(t, 1, got.InputEdits, "the input differs from the truth by one formatting choice")
		assert.Equal(t, 3, got.OutputEdits, "the obedient output also loses the two fillers the truth kept")
		assert.Greater(t, got.TruthWER(), got.InputWER(), "so doing as told looks like a regression word for word")
	})
}

// A word-for-word comparison is order-aware: the truth's own words scrambled
// are nowhere near it (the bag-of-words overlap this replaced rated them 1.0).
func TestScoreCorrection_isOrderAware(t *testing.T) {
	words := strings.Fields("one two three four five six seven eight nine ten eleven twelve")
	reversed := make([]string, len(words))

	for i, w := range words {
		reversed[len(words)-1-i] = w
	}

	got := ScoreCorrection(strings.Join(words, " "), strings.Join(words, " "), strings.Join(reversed, " "))

	assert.Greater(t, got.TruthWER(), 80.0)
}

func TestScoreCorrection_handlesDegenerateTruth(t *testing.T) {
	t.Run("a truth with no comparable words has a zero-denominator WER", func(t *testing.T) {
		got := ScoreCorrection("— …", "— …", "anything at all")

		require.Equal(t, 0, got.TruthWords)
		assert.Equal(t, 3, got.OutputEdits)
		assert.Equal(t, 0.0, got.TruthWER(), "wordErrorRate reports 0 for an empty reference rather than dividing by zero")
	})
}

func TestCorrectionWords(t *testing.T) {
	t.Run("keeps each word as written beside its normalized key", func(t *testing.T) {
		got := correctionWords(`"Well," Don’t go!`)

		assert.Equal(t, []correctionWord{
			{Key: "well", Surface: `"Well,"`},
			{Key: "dont", Surface: "Don’t"},
			{Key: "go", Surface: "go!"},
		}, got)
	})

	t.Run("standalone punctuation rides on the word before it", func(t *testing.T) {
		got := correctionWords("wait — what … really")

		assert.Equal(t, []correctionWord{
			{Key: "wait", Surface: "wait —"},
			{Key: "what", Surface: "what …"},
			{Key: "really", Surface: "really"},
		}, got, "dropped from the comparison, but still visible to a reviewer")
	})

	t.Run("leading punctuation rides on the first word", func(t *testing.T) {
		assert.Equal(t, []correctionWord{{Key: "hello", Surface: "— … hello"}}, correctionWords("— … hello"))
	})

	t.Run("a text with no words has none", func(t *testing.T) {
		assert.Empty(t, correctionWords("— … ?!"))
		assert.Empty(t, correctionWords(""))
	})

	// The view aligns Keys and the score compares correctionTokens: they must
	// be the same sequence, or what -debug shows would not be what is scored.
	t.Run("its keys are exactly correctionTokens", func(t *testing.T) {
		for _, text := range []string{
			"", "plain words only", `Mixed "Case," and — dashes … plus 5% and $3,000!`,
			"Ångström été 日本語 テスト", "  leading and trailing   ", "all-hands don't can’t", "\n\ttabs\nand\nnewlines\n",
		} {
			keys := make([]string, 0)
			for _, w := range correctionWords(text) {
				keys = append(keys, w.Key)
			}

			assert.Equal(t, correctionTokens(text), keys, "text %q", text)
		}
	})
}
