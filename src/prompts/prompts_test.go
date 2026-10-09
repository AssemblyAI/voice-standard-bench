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

// The reference-free judge must stay reference-free: it is graded against the
// input the model was shown and has to be unable to lean on the answer key.
func TestTextCorrectionReferenceFreeJudgeNeverMentionsAReference(t *testing.T) {
	spec, ok := For(TextCorrection)
	require.True(t, ok)

	assert.NotContains(t, spec.Judge, "REFERENCE")
	assert.Contains(t, spec.Judge, "You do not have a reference")
}
