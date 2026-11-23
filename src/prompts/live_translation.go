package prompts

import (
	"os"
	"sort"
)

// LiveTranslationPrompt is one candidate prompt for translating a streaming
// turn. User is a template: the task fills in {{source}} and {{target}}
// (language names), {{history}} (up to three previous turns, one per line, or
// "(none)") and {{turn}}, the same values the streaming API's llm_gateway
// templates ({{history_3}}, {{turn}}) carry. An empty System sends no system
// message, like a prompt given to llm_gateway as a single user message.
type LiveTranslationPrompt struct {
	System string
	User   string
}

// LiveTranslationPrompts holds every candidate, keyed by a short label, so a
// tuning pass can see what was already tried. Pick one with the
// LIVE_TRANSLATION_PROMPT environment variable (default
// DefaultLiveTranslationPrompt); each run reports one prompt, so compare
// prompts across runs.
var LiveTranslationPrompts = map[string]LiveTranslationPrompt{
	// interpreter_v1: the prompt shared in the Notion channel on 2026-10-15
	// for streaming translation through llm_gateway, with the target language
	// taken from the user message instead of hardcoded.
	"interpreter_v1": {
		System: `You are a simultaneous interpreter translating a live conversation, one speaker turn at a time.

The user message gives the target language, the previous turns for context only (already translated, do not repeat them), and the current turn.

Translate ONLY the current turn into natural, fluent text in the target language. Keep names, numbers and units as spoken. Do not add explanations, quotes, or any text other than the translation. If the turn is only a filler such as "um" or "okay", translate it to the closest natural equivalent.`,
		User: "Target language: {{target}}\n\nPrevious turns (context only):\n{{history}}\n\nCurrent turn:\n{{turn}}",
	},

	// docs_oneliner: the quickstart prompt on the real-time translation docs
	// page, as written: one user message, no context.
	"docs_oneliner": {
		User: "Translate the following {{source}} into natural {{target}} and reply with only the translation: {{turn}}",
	},

	// docs_history: docs_oneliner with the previous turns above the current
	// one, as the docs page suggests for {{history_3}}. Against docs_oneliner
	// it shows what context adds; against interpreter_v1, what the wording adds.
	"docs_history": {
		User: "Previous turns, for context only:\n{{history}}\n\nTranslate the following {{source}} into natural {{target}} and reply with only the translation: {{turn}}",
	},
}

// DefaultLiveTranslationPrompt is the prompt used when LIVE_TRANSLATION_PROMPT
// is unset.
const DefaultLiveTranslationPrompt = "interpreter_v1"

// ActiveLiveTranslationPrompt returns the selected prompt's label and value.
// An unknown label is an error for the caller to report, not a silent default.
func ActiveLiveTranslationPrompt() (name string, prompt LiveTranslationPrompt, ok bool) {
	name = os.Getenv("LIVE_TRANSLATION_PROMPT")
	if name == "" {
		name = DefaultLiveTranslationPrompt
	}

	prompt, ok = LiveTranslationPrompts[name]

	return name, prompt, ok
}

// LiveTranslationPromptNames lists the labels, sorted, for error messages.
func LiveTranslationPromptNames() []string {
	names := make([]string, 0, len(LiveTranslationPrompts))
	for n := range LiveTranslationPrompts {
		names = append(names, n)
	}

	sort.Strings(names)

	return names
}

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

// LiveTranslationTranscriptJudge grades a translated turn that has no
// reference translation (meeting audio) against the human transcript of what
// was said, in the source language. Same rubric names as the reference judge,
// so the scores sit in the same report columns.
const LiveTranslationTranscriptJudge = `You grade machine translations of live speech.

You get the SOURCE (a speech-to-text transcript of one speaker turn, which may contain recognition errors), WHAT WAS SAID (a human transcript of the same moment, in the original language; it may include filler words, false starts and words from other people talking over the speaker), and a CANDIDATE translation into the target language. There is no reference translation.

Score the candidate on two 1-5 scales:

- translation_accuracy: does the candidate carry the meaning of what was said? 5 = same meaning, nothing missing or added. 4 = one minor nuance off. 3 = the gist survives but a detail (name, number, negation, who did what) is wrong or missing. 2 = major parts missing or wrong. 1 = unrelated, untranslated, or in the wrong language. Fillers, false starts and cross-talk may be dropped without penalty. Where the SOURCE itself lost words, judge against what the SOURCE says.
- fluency: is the candidate natural, grammatical text in the target language? 5 = reads like a native speaker wrote it. 1 = broken or not in the target language. Extra commentary, quotes or labels around the translation lower fluency to at most 3.

Reply with brief reasoning, then a JSON object: {"translation_accuracy": <1-5>, "fluency": <1-5>, "notes": "<one line>"}`

// liveTranslationSpec's Task is the active prompt's system and user template
// together, so a replayed run refuses to mix outputs from different prompts
// (replay compares Task between runs).
var liveTranslationSpec = func() Spec {
	_, p, _ := ActiveLiveTranslationPrompt()

	return Spec{
		Task:   p.System + "\n\n---\n\n" + p.User,
		Judge:  liveTranslationJudge,
		Rubric: []string{"translation_accuracy", "fluency"},
	}
}()
