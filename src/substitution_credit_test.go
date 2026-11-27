package main

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestScoreDictationSubstitutionCredit(t *testing.T) {
	tests := []struct {
		name, truth, candidate, metric string
		want                           float64
	}{
		{"substituted comma word", dictationTruth, strings.Replace(dictationTruth, "Hey,", "Bye,", 1), "punctuation_accuracy", 75},
		{"substituted word without comma", dictationTruth, strings.Replace(dictationTruth, "Hey,", "Bye", 1), "punctuation_accuracy", 75},
		{"deleted comma word", dictationTruth, strings.Replace(dictationTruth, "Hey, ", "", 1), "punctuation_accuracy", 75},
		{"case-only comma word", dictationTruth, strings.Replace(dictationTruth, "Hey,", "hey,", 1), "punctuation_accuracy", 100},
		{"missing comma", dictationTruth, strings.Replace(dictationTruth, "Hey,", "Hey", 1), "punctuation_accuracy", 75},
		{"spurious comma", dictationTruth, strings.Replace(dictationTruth, "are ", "are, ", 1), "punctuation_accuracy", 80},
		{"inserted comma word", dictationTruth, strings.Replace(dictationTruth, "are ", "extra, are ", 1), "punctuation_accuracy", 100},
		{"quoted substitution", `Say "hello" now.`, `Say "bye" now.`, "formatting_command_accuracy", 0},
		{"quoted deletion", `Say "hello" now.`, `Say now.`, "formatting_command_accuracy", 0},
		{"quoted case-only match", `Say "hello" now.`, `Say "Hello" now.`, "formatting_command_accuracy", 100},
		{"missing quotes", `Say "hello" now.`, `Say hello now.`, "formatting_command_accuracy", 0},
		{"spurious quotes", `Say hello now.`, `Say "hello" now.`, "formatting_command_accuracy", 0},
		{"inserted quoted word", `Say "hello" now.`, `Say "extra" "hello" now.`, "formatting_command_accuracy", 100},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			scores := ScoreDictation(tc.truth, tc.candidate, allDictationMetrics)
			assert.Equal(t, tc.want, scores[tc.metric])
		})
	}
	t.Run("real fixture quality", func(t *testing.T) {
		scores := ScoreDictation(dictationTruth, strings.Replace(dictationTruth, "Hey,", "Bye,", 1), allDictationMetrics)
		assert.Equal(t, 5.0, scores["word_error_rate"])
		assert.Equal(t, 95.0, scores["capitalization_accuracy"])
		q, ok := sampleQuality(scores)
		assert.True(t, ok)
		assert.InDelta(t, (0.0+0.95+0.75+0.95)/4, q, 1e-9)
	})
}
