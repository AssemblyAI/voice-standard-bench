package prompts

// correctionSpec frames correction as constrained rewriting — restore
// intended, readable text without changing what was said — the same contract
// used by the OCR/ASR post-correction literature (see HIPE-OCRepair's task
// definition as applied in ../../jobs/text_correction.py: "restore the
// intended wording only — do not modernize, paraphrase, or add content").
// Applied to ASR output the analogous errors are disfluencies, false starts,
// filler words, and missing punctuation/casing rather than misrecognized
// characters, but the constraint is identical: fix the surface form, do not
// touch the content.
//
// Unlike the other two tasks, text_correction has a real answer key:
// aai_transcripts.jsonl carries missed_text, a simulated corrupted version of
// each transcript, alongside the known-true original in text (see
// TranscriptRecord in ../dataset.go). The TASK prompt and the candidate model
// only ever see missed_text — sending the true text as input would hand the
// model the answer.
//
// It takes TWO judge calls, kept apart on purpose:
//
//   - Judge (Rubric: faithfulness, improvement) is reference-free. It only
//     ever sees missed_text and the candidate's output, grading whether the
//     correction is a plausible, faithful cleanup of what the model was
//     shown, with no way to know what the hidden original says. Keeping the
//     answer key out of it keeps these two dimensions independent of the third.
//   - Compare (CompareRubric: meaning_similarity) is reference-based. It
//     sees the true original and the candidate's output, and scores how
//     similar the two are in actual MEANING.
//
// Why a judge reading for meaning rather than a word count against the
// truth: the truth is a raw transcript, and the task prompt asks for things a
// raw transcript doesn't have. The model is told to remove filler words,
// false starts and stutters, which the truth keeps (3% of the words at 3600s
// are um/uh and another 1.5% are immediate repeats), and to leave numbers
// "expressed the way the speaker expressed it" while the truth writes
// numerals where the model is shown the spelled-out form. Measured word for
// word against the truth, a model doing exactly what it was told scores worse
// than one that does nothing. Compare is told what does and doesn't count
// (fillers, stutters, number formatting, punctuation and casing don't; a
// changed word, a wrong name or amount, dropped or added content do), so those
// differences are ignored by definition while the ones that matter -- a
// sound-alike that swaps a word ("week"/"weak"), a skipped passage, text that
// loops -- still cost the model.
//
// A word-level comparison survives only as context (ScoreCorrection in
// ../correction.go): the output's and the input's word error rates against the
// truth, which flag a grossly wrong output (truncated, or a runaway
// generation) and feed the -debug alignment view. They are not scored.
var correctionSpec = Spec{
	Rubric: []string{"faithfulness", "improvement"},

	CompareRubric: []string{"meaning_similarity"},

	Task: `You are cleaning up a raw call transcript for readability.

You will be given the raw transcript of a phone or video call. Produce a corrected version.

Rules:
- Remove filler words (um, uh, like), false starts, and stutters ("I- I think" becomes "I think").
- Fix obvious mis-transcriptions where the intended word is clear from context.
- Fix punctuation, capitalization, and sentence boundaries.
- Do NOT paraphrase, reword, summarize, shorten, or add anything. Every fact, name, number, and claim in the original must still be present, expressed the way the speaker expressed it.
- Do NOT remove disagreements, hedges, or informal phrasing that reflects what was actually meant — only remove noise that carries no meaning.
- If a passage is already clean, leave it unchanged.
- Output ONLY the corrected transcript. No preamble, no explanation of what you changed.`,

	Judge: `You are grading a "cleaned up" call transcript against the original it was produced from. You do not have a reference correction to compare against — grade the correction directly against the original.

Score two dimensions, each from 1 (worst) to 5 (best):

- faithfulness: Does the corrected version preserve every fact, name, number, and claim from the original, expressed with the same meaning? Paraphrasing, summarizing, dropping content, or changing what was said scores low here, even if the prose reads better. Removing filler words and disfluencies is NOT a faithfulness violation — that is the intended job.
- improvement: Is the corrected version actually more readable than the original — filler words and false starts removed, punctuation and casing fixed — without over-correcting into something that no longer reads like a real conversation? A correction that changes nothing when the original clearly needed cleanup scores low here, as does a correction that is technically faithful but left obvious disfluencies untouched.

Think step by step in one or two sentences per dimension, citing specific evidence. Then, as the very last thing in your reply, output a JSON object and nothing after it, with exactly this shape:

{"faithfulness": <1-5>, "improvement": <1-5>, "notes": "<one sentence on the most significant issue found, or \"none\" if the correction is clean>"}`,

	Compare: `You are comparing two versions of the same phone or video call transcript, to decide whether they say the same thing.

REFERENCE is the correct transcript. CANDIDATE is another version of it, produced by someone correcting a copy that had transcription errors. Score how similar the CANDIDATE is to the REFERENCE in actual MEANING, from 1 (worst) to 5 (best).

Judge meaning, not wording. These differences do NOT change the meaning and must NOT lower the score:
- filler words, false starts, and stutters: "um", "uh", "like", a word cut off and restarted ("I- I think"), a word repeated ("the the")
- punctuation, capitalization, and sentence or paragraph breaks
- a number written with digits in one version and with words in the other: "5" and "five", "10%" and "ten percent", "$3,000" and "three thousand dollars", "21st" and "twenty first", "2024" and "twenty twenty four"
- harmless spelling or formatting variants: "okay" and "OK", "all-hands" and "all hands"

These DO change the meaning and lower the score:
- a different word that changes what was said, including a sound-alike ("week" for "weak", "their" for "there", "for" where the reference says "4"), a wrong name, a wrong number or amount, a dropped or added negation ("can" for "can't")
- content that is in the REFERENCE but missing from the CANDIDATE: a dropped sentence, a skipped passage, text that stops early
- content that is in the CANDIDATE but not in the REFERENCE: added commentary or an introduction ("Here is the corrected transcript"), repeated or looping text, invented material
- passages that were summarized, reworded, or reordered so that a detail changes

Scale:
5 - The same meaning throughout. Every fact, name, number, and claim matches; any differences are only the harmless kind above.
4 - One small meaning difference: a single word or short phrase is different or missing, and nothing else is.
3 - A few small differences, or one clear difference in something important (a name, a number, a negation, a decision).
2 - Many differences, or whole passages missing, added, or changed.
1 - The meaning is substantially different: most of the content is missing, replaced, or garbage.

Work through the two texts in order and, in a few sentences, list each place where the meaning differs, ignoring the harmless differences. Then, as the very last thing in your reply, output a JSON object and nothing after it, with exactly this shape:

{"meaning_similarity": <1-5>, "notes": "<one sentence on the most significant meaning difference found, or \"none\" if the candidate says the same thing>"}`,
}
