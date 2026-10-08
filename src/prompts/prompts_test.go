package prompts

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Every judge prompt has to name, in the JSON shape it asks for, each score the
// caller will read -- a rubric entry the prompt never mentions would make every
// judge reply fail to parse.
func TestEverySpecsRubricIsNamedByItsJudgePrompt(t *testing.T) {
	for _, task := range All {
		spec, ok := For(task)
		require.True(t, ok, task)

		if spec.Judge == "" {
			continue // dictation: graded mechanically, its Rubric names those scores, not a judge's JSON
		}

		for _, dim := range spec.Rubric {
			assert.Contains(t, spec.Judge, `"`+dim+`"`, "%s's judge prompt must show %q in its JSON shape", task, dim)
		}
	}
}

func TestEverySpecsCompareRubricIsNamedByItsComparePrompt(t *testing.T) {
	for _, task := range All {
		spec, ok := For(task)
		require.True(t, ok, task)

		if spec.Compare == "" {
			assert.Empty(t, spec.CompareRubric, "%s has no comparison prompt, so no comparison rubric", task)

			continue
		}

		require.NotEmpty(t, spec.CompareRubric, "%s has a comparison prompt but nothing for it to score", task)

		for _, dim := range spec.CompareRubric {
			assert.Contains(t, spec.Compare, `"`+dim+`"`, "%s's comparison prompt must show %q in its JSON shape", task, dim)
		}
	}
}

// Only a task with a real answer key has a comparison prompt, and today that is
// text_correction alone.
func TestOnlyTextCorrectionComparesAgainstTheTruth(t *testing.T) {
	for _, task := range All {
		spec, _ := For(task)

		assert.Equal(t, task == TextCorrection, spec.Compare != "", task)
	}
}

// The reference-free judge must stay reference-free: it is graded against the
// input the model was shown and has to be unable to lean on the answer key.
func TestTextCorrectionReferenceFreeJudgeNeverMentionsAReference(t *testing.T) {
	spec, ok := For(TextCorrection)
	require.True(t, ok)

	assert.NotContains(t, spec.Judge, "REFERENCE")
	assert.Contains(t, spec.Judge, "You do not have a reference")
}

// The comparison prompt is what makes this a meaning comparison rather than a
// word count, so the differences it must look past -- and the ones it must not
// -- are pinned here: if someone trims the prompt, the score silently goes back
// to punishing a model for doing as it was told.
func TestTextCorrectionComparePromptSeparatesWordingFromMeaning(t *testing.T) {
	spec, ok := For(TextCorrection)
	require.True(t, ok)

	assert.Equal(t, []string{"meaning_similarity"}, spec.CompareRubric)

	lookPast := []string{
		"filler words", `"um"`, `"uh"`, `"like"`, "stutters", "punctuation", "capitalization",
		`"5" and "five"`, `"10%" and "ten percent"`, "do NOT change the meaning",
	}
	for _, phrase := range lookPast {
		assert.Contains(t, spec.Compare, phrase, "the prompt must tell the judge to look past this")
	}

	counts := []string{
		"DO change the meaning", "sound-alike", `"week" for "weak"`, "wrong name", "wrong number",
		"dropped sentence", "text that stops early", "repeated or looping text", "invented material",
	}
	for _, phrase := range counts {
		assert.Contains(t, spec.Compare, phrase, "the prompt must tell the judge this still counts")
	}

	// It is asked for in a fixed shape the caller can parse.
	assert.Contains(t, spec.Compare, `{"meaning_similarity": <1-5>`)
	assert.Contains(t, spec.Compare, "REFERENCE is the correct transcript")
	assert.Contains(t, spec.Compare, "CANDIDATE")
}

// The meaning judge looks past "like", so the word-level filler list that used
// to be tied to the task prompt no longer matters -- but the task prompt must
// still tell the model what to do with fillers, since the comparison is only
// fair if the two prompts agree about it.
func TestTextCorrectionTaskAndComparePromptsAgreeAboutFillers(t *testing.T) {
	spec, ok := For(TextCorrection)
	require.True(t, ok)

	assert.Contains(t, spec.Task, "Remove filler words (um, uh, like)")
	assert.Contains(t, spec.Compare, `"um", "uh", "like"`, "what the task prompt tells the model to remove is what the comparison prompt tells the judge to ignore")
}
