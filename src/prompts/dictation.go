package prompts

// dictationPromptVariants holds every dictation task-prompt candidate tried
// while tuning dictationSpec.Task, keyed by a short label — kept here rather
// than deleted once a winner is picked, so a later tuning pass can see
// exactly what was already tried and why it lost, instead of starting from
// nothing. Every variant is a semantically equivalent restatement of the
// SAME command grammar dictation/synthetic.jsonl's truth_text was generated
// against (comma/period/question mark/exclamation point/colon/semicolon,
// all caps/cap, new paragraph/new line, open/close quote and parenthesis,
// hyphen, apostrophe s, spoken numbers spelled out) — only the wording,
// structure, and emphasis differ between them; changing the actual rules
// would score a candidate against a grammar the dataset was never built for.
//
// Only dictationPromptActive (below) is ever read into dictationSpec.Task at
// runtime — every other entry here is inert data, never sent to a model.
var dictationPromptVariants = map[string]string{
	// original: the starting prompt, unchanged. It states every rule but
	// never tells the model to suppress a preamble/explanation, which a
	// benchmark run (see prompts/README.md's "Dictation" section) showed
	// costing it on exact_match in particular.
	"original": `This is a raw dictation transcript: punctuation, capitalization, and formatting are stated aloud as literal spoken commands rather than applied automatically. Produce the final formatted text by applying these commands: 'comma' -> ','; 'period' -> '.'; 'question mark' -> '?'; 'exclamation point' -> '!'; 'colon' -> ':'; 'semicolon' -> ';'; 'all caps <word>' uppercases that one word ('all caps <word> ... end caps' uppercases the whole span); 'cap <word>' capitalizes just the first letter of that word; 'new paragraph' starts a new paragraph; 'new line' starts a new line; 'open quote' / 'close quote' wrap the enclosed text in double quotes; 'open parenthesis' / 'close parenthesis' wrap the enclosed text in parentheses; 'hyphen' joins the surrounding words with a hyphen; 'apostrophe s' attaches a possessive 's directly to the previous word. Spoken numbers stay spelled out as words in the final text.`,

	// v1_output_only: the smallest possible change -- original, verbatim,
	// plus one paragraph telling the model to reply with nothing but the
	// rewritten text. Isolates the "preamble suppression" hypothesis on its
	// own, with no other wording change to confound it.
	"v1_output_only": `This is a raw dictation transcript: punctuation, capitalization, and formatting are stated aloud as literal spoken commands rather than applied automatically. Produce the final formatted text by applying these commands: 'comma' -> ','; 'period' -> '.'; 'question mark' -> '?'; 'exclamation point' -> '!'; 'colon' -> ':'; 'semicolon' -> ';'; 'all caps <word>' uppercases that one word ('all caps <word> ... end caps' uppercases the whole span); 'cap <word>' capitalizes just the first letter of that word; 'new paragraph' starts a new paragraph; 'new line' starts a new line; 'open quote' / 'close quote' wrap the enclosed text in double quotes; 'open parenthesis' / 'close parenthesis' wrap the enclosed text in parentheses; 'hyphen' joins the surrounding words with a hyphen; 'apostrophe s' attaches a possessive 's directly to the previous word. Spoken numbers stay spelled out as words in the final text.

Output ONLY the final formatted text. Do not include a preamble, an explanation, a restatement of the instructions, or any commentary before or after it. Do not wrap the output in quotes or a code fence.`,

	// v2_structured_list: the same rules restated as a bulleted list instead
	// of one dense semicolon-separated sentence, plus the output-only
	// paragraph. Tests whether visual structure alone helps a model apply
	// each command correctly.
	"v2_structured_list": `You will be given a raw dictation transcript in which punctuation, capitalization, and formatting are stated aloud as literal spoken commands instead of being applied automatically. Rewrite it into final formatted text by replacing each spoken command as follows, leaving every other word exactly as given:

- 'comma' -> ','
- 'period' -> '.'
- 'question mark' -> '?'
- 'exclamation point' -> '!'
- 'colon' -> ':'
- 'semicolon' -> ';'
- 'all caps <word>' -> uppercase that one word; 'all caps <word> ... end caps' -> uppercase the whole span between them
- 'cap <word>' -> capitalize just the first letter of that word
- 'new paragraph' -> start a new paragraph
- 'new line' -> start a new line
- 'open quote' ... 'close quote' -> wrap the enclosed text in double quotes
- 'open parenthesis' ... 'close parenthesis' -> wrap the enclosed text in parentheses
- 'hyphen' -> join the surrounding words with a hyphen
- 'apostrophe s' -> attach a possessive 's directly to the previous word
- Spoken numbers stay spelled out as words -- never convert them to digits.

Output ONLY the final formatted text. Do not include a preamble, an explanation, or any commentary before or after it.`,

	// v3_worked_example: v2_structured_list plus one worked example, and an
	// explicit rule for a case the prose alone leaves implicit -- an
	// all-caps span that spells out single letters (e.g. an acronym) joins
	// them into one word with no spaces, which "uppercases the whole span"
	// doesn't actually say on its own. Tests whether a concrete example (and
	// naming this one edge case) closes a real gap in the instructions,
	// particularly for formatting_command_accuracy.
	"v3_worked_example": `You will be given a raw dictation transcript in which punctuation, capitalization, and formatting are stated aloud as literal spoken commands instead of being applied automatically. Rewrite it into final formatted text by replacing each spoken command as follows, leaving every other word exactly as given:

- 'comma' -> ','
- 'period' -> '.'
- 'question mark' -> '?'
- 'exclamation point' -> '!'
- 'colon' -> ':'
- 'semicolon' -> ';'
- 'all caps <word>' -> uppercase that one word; 'all caps <word> ... end caps' -> uppercase the whole span between them. If the span spells out single letters (e.g. 'all caps c d c end caps'), join them into one uppercase word with no spaces (e.g. 'CDC').
- 'cap <word>' -> capitalize just the first letter of that word
- 'new paragraph' -> start a new paragraph
- 'new line' -> start a new line
- 'open quote' ... 'close quote' -> wrap the enclosed text in double quotes
- 'open parenthesis' ... 'close parenthesis' -> wrap the enclosed text in parentheses
- 'hyphen' -> join the surrounding words with a hyphen
- 'apostrophe s' -> attach a possessive 's directly to the previous word
- Spoken numbers stay spelled out as words -- never convert them to digits.

Example:
Input: Quick update colon the all caps c e o end caps apostrophe s flight lands at nine period
Output: Quick update: the CEO's flight lands at nine.

Output ONLY the final formatted text, matching the style of the example above. Do not include a preamble, an explanation, or any commentary before or after it.`,

	// v4_terse: the opposite bet from v2/v3 -- much shorter and denser
	// rather than more thoroughly spelled out, on the hypothesis that less
	// prose leaves less room to misread or over-think the transformation.
	// Still carries the output-only instruction and the acronym-collapse
	// rule, just compressed.
	"v4_terse": `Rewrite this dictation transcript: replace every spoken formatting command below with its effect, and leave all other words exactly as given. comma->, | period->. | question mark->? | exclamation point->! | colon->: | semicolon->; | all caps X->X in upper case (all caps X...end caps->the whole span in upper case, letters spelled one at a time joined with no spaces, e.g. "c d c"->"CDC") | cap X->capitalize just X's first letter | new paragraph->paragraph break | new line->line break | open quote/close quote->wrap in "double quotes" | open parenthesis/close parenthesis->wrap in (parentheses) | hyphen->join the two surrounding words with a hyphen | apostrophe s->attach a possessive 's to the previous word. Keep spoken numbers spelled out as words, never digits. Reply with ONLY the rewritten text -- no preamble, no explanation.`,

	// v5_negative_emphasis: v2_structured_list plus an explicit warning
	// against the most visible possible failure mode -- leaving a literal
	// command word (e.g. "comma", "all caps") untranslated in the output.
	// Tests whether naming the failure mode directly, rather than only
	// stating the positive transformation rule, reduces it.
	"v5_negative_emphasis": `You will be given a raw dictation transcript in which punctuation, capitalization, and formatting are stated aloud as literal spoken commands instead of being applied automatically. Rewrite it into final formatted text by replacing each spoken command as follows, leaving every other word exactly as given:

- 'comma' -> ','
- 'period' -> '.'
- 'question mark' -> '?'
- 'exclamation point' -> '!'
- 'colon' -> ':'
- 'semicolon' -> ';'
- 'all caps <word>' -> uppercase that one word; 'all caps <word> ... end caps' -> uppercase the whole span between them
- 'cap <word>' -> capitalize just the first letter of that word
- 'new paragraph' -> start a new paragraph
- 'new line' -> start a new line
- 'open quote' ... 'close quote' -> wrap the enclosed text in double quotes
- 'open parenthesis' ... 'close parenthesis' -> wrap the enclosed text in parentheses
- 'hyphen' -> join the surrounding words with a hyphen
- 'apostrophe s' -> attach a possessive 's directly to the previous word
- Spoken numbers stay spelled out as words -- never convert them to digits.

Every spoken command word above (comma, period, question mark, exclamation point, colon, semicolon, all caps, cap, new paragraph, new line, open/close quote, open/close parenthesis, hyphen, apostrophe s) is an INSTRUCTION, never content -- it must never appear literally in your output. Do not add, remove, reorder, or rephrase any other word.

Output ONLY the final formatted text. Do not include a preamble, an explanation, or any commentary before or after it.`,
}

// dictationPromptActive selects which entry of dictationPromptVariants is
// actually sent to candidate models (see dictationSpec.Task below) --
// change this one line to switch prompts, rather than moving text in or out
// of dictationSpec.Task directly, so every variant ever tried stays
// preserved in dictationPromptVariants above.
const dictationPromptActive = "v5_negative_emphasis"

// dictationSpec's Task, unlike the other three tasks it sits alongside in
// specs, is not what's unusual about it — it IS a single fixed system prompt
// sent to every dictation record, exactly the same way summarizationSpec/
// extractionSpec/correctionSpec's Task fields work. What's unusual is
// everything else:
//
//   - dictation/synthetic.jsonl carries its own per-record input.instruction
//     field, but it is deliberately never read (see rawDictationRecord in
//     ../dictation.go) — Task below is the only prompt ever sent, regardless
//     of what that field happens to contain for any given record.
//   - Judge is deliberately left empty, and Rubric does not name LLM-graded
//     dimensions. Every dictation record carries a real answer key
//     (expected.truth_text), so grading is entirely mechanical — word error
//     rate, exact match, and punctuation/capitalization/formatting-command
//     accuracy, all computed against that truth text in code
//     (ScoreDictation in ../dictation.go) — never an LLM call. Rubric is
//     still populated, purely so this task's score names are documented in
//     one place the way every other task's are, but unlike the other three
//     it is never read to build a judge prompt's expected JSON keys
//     (contrast parseJudgeScores in ../judge.go): a record's own "metrics"
//     field (DictationRecord.Metrics) selects which of these are actually
//     computed for it.
//
// Registered in specs (prompts.go) so prompts.For(Dictation) resolves —
// needed for -tasks validation (voicebench.go's parseTasks) — and so
// prompts.All lists it for error messages. Task IS read, by runDictationOne
// in ../dictation.go; Judge is not, since there is no judge call on this
// path at all — see ../dictation.go's package doc comment for the full
// runner, which still bypasses runOne/runModel (the shared path the other
// three tasks use) despite sharing their fixed-Task-prompt shape now, purely
// because of the mechanical-vs-judge grading difference above.
var dictationSpec = Spec{
	Task: dictationPromptVariants[dictationPromptActive],

	Rubric: []string{
		"exact_match",
		"word_error_rate",
		"punctuation_accuracy",
		"capitalization_accuracy",
		"formatting_command_accuracy",
	},
}
