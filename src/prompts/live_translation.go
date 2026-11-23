package prompts

// liveTranslationPromptVariants holds every live translation task-prompt
// candidate, keyed by a short label, so a tuning pass can see what was
// already tried. Each one is the system prompt for a single streaming turn:
// the user message (built by the task, see live_translation_task.go) names
// the target language, carries up to three previous turns as context, and
// then the turn to translate -- the same shape the streaming API's
// llm_gateway parameter sends per turn with {{history_3}} and {{turn}}.
//
// Only liveTranslationPromptActive is sent to a model.
var liveTranslationPromptVariants = map[string]string{
	// interpreter_v1: the prompt shared in the Notion channel on 2026-10-15
	// for streaming translation through llm_gateway, adapted to take the
	// target language from the user message instead of hardcoding Spanish.
	"interpreter_v1": `You are a simultaneous interpreter translating a live conversation, one speaker turn at a time.

The user message gives the target language, the previous turns for context only (already translated, do not repeat them), and the current turn.

Translate ONLY the current turn into natural, fluent text in the target language. Keep names, numbers and units as spoken. Do not add explanations, quotes, or any text other than the translation. If the turn is only a filler such as "um" or "okay", translate it to the closest natural equivalent.`,
}

// liveTranslationPromptActive selects which variant is sent.
const liveTranslationPromptActive = "interpreter_v1"

// liveTranslationJudge grades one translated turn against a human reference
// translation. Unlike the transcript tasks this is reference-based: FLEURS
// gives a professional translation of every sentence. The source shown to the
// judge is the streaming transcript the candidate actually saw, so a
// candidate is not penalised for a word the speech model already got wrong,
// only for losing or changing meaning the transcript carried.
const liveTranslationJudge = `You grade machine translations of live speech.

You get the SOURCE (a speech-to-text transcript of one speaker turn, which may contain recognition errors), a REFERENCE translation written by a professional translator from the original sentence, and a CANDIDATE translation.

Score the candidate on two 1-5 scales:

- translation_accuracy: does the candidate carry the same meaning as the reference? 5 = same meaning, nothing missing or added. 4 = one minor nuance off. 3 = the gist survives but a detail (name, number, negation, who did what) is wrong or missing. 2 = major parts missing or wrong. 1 = unrelated, untranslated, or in the wrong language. Where the source transcript itself lost a word, judge against what the source says, not the reference.
- fluency: is the candidate natural, grammatical text in the target language? 5 = reads like a native speaker wrote it. 1 = broken or not in the target language. Extra commentary, quotes or labels around the translation lower fluency to at most 3.

Reply with brief reasoning, then a JSON object: {"translation_accuracy": <1-5>, "fluency": <1-5>, "notes": "<one line>"}`

var liveTranslationSpec = Spec{
	Task:   liveTranslationPromptVariants[liveTranslationPromptActive],
	Judge:  liveTranslationJudge,
	Rubric: []string{"translation_accuracy", "fluency"},
}
