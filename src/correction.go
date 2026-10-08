package main

import (
	"strings"
	"unicode"
)

// Word-level measures of a text_correction output against the truth.
//
// A text_correction record carries three texts: the known-true original
// (TranscriptRecord.Text), a simulated corrupted copy of it (MissedText --
// the ONLY thing the model is shown), and the model's corrected output. How
// GOOD a correction is -- whether it says what the truth says -- is the
// reference-based LLM judge's call, in actual meaning (RunReferenceJudge,
// meaning_similarity), because a word-for-word comparison with the truth
// punishes a model for following its prompt: the truth keeps the fillers and
// stutters the prompt says to remove, and writes numerals where the model is
// shown the spelled-out numbers it is told to leave alone (see correctionSpec
// in ./prompts for the numbers).
//
// What this file keeps is the word-level distance, as CONTEXT that is cheap,
// objective, and the same one the -debug view aligns:
//
//   - truth_wer / input_wer: the output's and the uncorrected input's word
//     error rates against the truth, so a reviewer can see at a glance how
//     much text actually changed and set it beside the judge's score;
//   - the word accuracy (1 - WER) that flags a GROSSLY wrong output --
//     truncated, refused, or a runaway generation many times the truth's length
//     -- for failure capture, where no judge is needed to know something broke.
//
// Neither is a quality dimension (see dimensionRange in summary.go). The
// comparison is order-aware (an edit distance, not a bag of words) and ignores
// case and punctuation, but it does NOT treat fillers, stutters, or number
// formatting as equivalent: that is the judge's job, which is the point.

// The score names ScoreCorrection's results are recorded under in
// Sample.Judge. Both are informational context, deliberately NOT in
// dimensionRange, so they can never be averaged into quality_score.
const (
	// scoreTruthWER: the candidate's word error rate against the truth, in
	// percent (lower is better; above 100 when it adds more words than the
	// truth has).
	scoreTruthWER = "truth_wer"
	// scoreInputWER: the same for the uncorrected input -- the do-nothing
	// baseline, identical for every model given the same record.
	scoreInputWER = "input_wer"
)

// normalizeWord is the comparable form of one whitespace-delimited field:
// lowercased, with every rune that isn't a letter or digit removed (so
// "Don't," and "don’t" both become "dont"). It is empty for a field with no
// letter or digit at all -- standalone punctuation such as an em dash.
func normalizeWord(field string) string {
	var word strings.Builder

	for _, r := range field {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			word.WriteRune(unicode.ToLower(r))
		}
	}

	return word.String()
}

// correctionTokens splits text into the comparable word tokens: whitespace
// delimited and normalized by normalizeWord, with the empty ones --
// standalone punctuation -- dropped.
func correctionTokens(text string) []string {
	fields := strings.Fields(text)
	tokens := make([]string, 0, len(fields))

	for _, field := range fields {
		if word := normalizeWord(field); word != "" {
			tokens = append(tokens, word)
		}
	}

	return tokens
}

// correctionWord is one comparable word of a text: Key is the normalized form
// the scores compare, Surface is how it was actually written -- case,
// punctuation and all -- which is what a person reviewing an alignment wants
// to read. Standalone punctuation (an em dash, an ellipsis) is not a word to
// compare, but it is still text, so it rides along on the Surface of the word
// before it (or, at the very start of a text, the word after).
type correctionWord struct {
	Key     string
	Surface string
}

// correctionWords is correctionTokens with each token's as-written form kept
// beside it. Its Keys are exactly correctionTokens(text) -- one source of
// truth for what counts as a word -- so an alignment of Keys is an alignment
// of precisely what the scores measure.
func correctionWords(text string) []correctionWord {
	fields := strings.Fields(text)
	words := make([]correctionWord, 0, len(fields))

	var leading string // standalone punctuation seen before the first word

	for _, field := range fields {
		key := normalizeWord(field)

		switch {
		case key != "":
			words = append(words, correctionWord{Key: key, Surface: leading + field})
			leading = ""
		case len(words) > 0:
			words[len(words)-1].Surface += " " + field
		default:
			leading += field + " "
		}
	}

	return words
}

// CorrectionResult is the raw word-edit counts behind one correction's
// scores. It is kept as counts, with every score derived from them, so a
// caller (RunJudge's failure capture) and the recorded scores can never
// disagree about what was measured.
type CorrectionResult struct {
	// TruthWords is the number of comparable words in the truth -- the WER
	// denominator.
	TruthWords int
	// InputEdits is the word edit distance from the uncorrected input to the
	// truth: how many errors the model was handed, i.e. the do-nothing
	// baseline.
	InputEdits int
	// OutputEdits is the word edit distance from the candidate to the truth:
	// how many errors are left.
	OutputEdits int
}

// ScoreCorrection measures candidate against truth, relative to input -- the
// corrupted text the model was actually shown. See this file's package-level
// comment for what each number means and why.
func ScoreCorrection(truth, input, candidate string) CorrectionResult {
	truthTokens := correctionTokens(truth)

	return CorrectionResult{
		TruthWords:  len(truthTokens),
		InputEdits:  editDistance(truthTokens, correctionTokens(input)),
		OutputEdits: editDistance(truthTokens, correctionTokens(candidate)),
	}
}

// TruthWER is the candidate's word error rate against the truth, in percent.
func (r CorrectionResult) TruthWER() float64 {
	return wordErrorRate(r.TruthWords, r.OutputEdits)
}

// InputWER is the uncorrected input's word error rate against the truth, in
// percent -- what a do-nothing "correction" would score.
func (r CorrectionResult) InputWER() float64 {
	return wordErrorRate(r.TruthWords, r.InputEdits)
}

// WordAccuracy is 1 - TruthWER as a fraction: 1 for a perfect match with the
// truth, and negative once the candidate has more errors than the truth has
// words (a runaway generation). It is the quantity the low-accuracy failure
// capture thresholds on.
func (r CorrectionResult) WordAccuracy() float64 {
	return 1 - r.TruthWER()/100
}

// Scores returns the two values RunJudge records on the Sample, keyed by the
// score names above.
func (r CorrectionResult) Scores() map[string]float64 {
	return map[string]float64{
		scoreTruthWER: r.TruthWER(),
		scoreInputWER: r.InputWER(),
	}
}
