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
// text_correction is scored by ONE reference-free judge call (Judge, Rubric:
// faithfulness, improvement). It only ever sees missed_text -- the corrupted
// transcript the candidate model was shown -- and the candidate's output,
// grading whether the correction is a faithful, genuine cleanup of what the
// model was shown.
//
// aai_transcripts.jsonl also carries the original transcript in text, but it is
// deliberately NOT used for scoring: it turned out not to be an accurate answer
// key, so comparing against it -- word for word or by a reference-based judge --
// would score the dataset's errors as much as the model's. (It also keeps the
// fillers and stutters the Task prompt tells the model to remove, and writes
// numerals where the model is shown the spelled-out form.) The TASK prompt and
// the candidate model only ever see missed_text.
var correctionSpec = Spec{
	Rubric: []string{"faithfulness", "improvement"},

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
}
