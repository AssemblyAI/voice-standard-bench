package prompts

// extractionSpec asks for one fixed JSON schema rather than free-form entity
// tagging, following the instruction-based structured-extraction pattern from
// the LLM-IE literature (e.g. Wei et al., 2023, "ChatIE: Zero-Shot
// Information Extraction via Chatting with ChatGPT", and the GPT-NER
// prompting convention it builds on): state the target schema explicitly in
// the prompt, and require the model to emit empty arrays/"unspecified"
// rather than omitting keys, so the output's STRUCTURE can be validated
// mechanically (see ValidateExtraction in ../judge.go) without spending a
// model call on it.
//
// The schema is scoped to what a call transcript actually contains —
// participants, action items, decisions, named entities — rather than a
// generic PERSON/ORG/LOC tag set, since that generic set is a poor fit for a
// phone call and duplicates what a real product would ask for (this mirrors
// what a CRM or meeting-notes tool queries a transcript for).
//
// The rubric mirrors classic IE evaluation's precision/recall split (see
// entity_f1 / record_completeness in
// ../../jobs/information_extraction.py), adapted to be reference-free:
// "groundedness" is judge-estimated precision — did it invent anything not in
// the transcript — and "completeness" is judge-estimated recall — did it miss
// anything load-bearing. schema_valid is NOT part of the judge's rubric: it's
// checked mechanically before the judge ever runs, and a structurally invalid
// response skips the judge call entirely (see ValidateExtraction).
var extractionSpec = Spec{
	Rubric: []string{"groundedness", "completeness"},

	Task: `You are an expert at extracting structured information from call transcripts.

You will be given the raw transcript of a phone or video call. Extract information into EXACTLY this JSON shape:

{
  "participants": ["<name or role of each speaker who can be identified, e.g. \"Alice\" or \"the caller\">"],
  "action_items": [{"owner": "<who is responsible, or \"unspecified\">", "task": "<what they agreed to do>", "due": "<when, or \"unspecified\">"}],
  "decisions": ["<a concrete decision or agreement reached during the call>"],
  "entities": {"people": ["<named individuals mentioned who are not speakers>"], "organizations": ["<company or team names mentioned>"], "products": ["<product, tool, or project names mentioned>"], "dates": ["<specific dates or deadlines mentioned>"]}
}

Rules:
- Include ONLY things actually stated in the transcript. Do not infer a due date, owner, or decision that was not said.
- If a field has nothing to report, use an empty array (or "unspecified" for owner/due) — do not omit the key.
- Do not extract the same fact into two different fields.
- Output ONLY the JSON object. No preamble, no markdown code fence, no commentary before or after it.`,

	Judge: `You are grading a structured extraction against the call transcript it was extracted from. You will be given the transcript and the extracted JSON. The JSON's structural validity has already been checked separately — do not grade formatting, only content.

Score two dimensions, each from 1 (worst) to 5 (best):

- groundedness: Is everything in the extraction actually supported by the transcript? An invented action item, a decision that wasn't really made, or a person/organization not mentioned in the call scores low here — this is the "did it hallucinate" check. An extraction that captures very little, but everything it captures is real, still scores high on groundedness.
- completeness: Does the extraction capture the action items, decisions, and named entities that the transcript actually contains? Missing a clearly stated commitment or decision scores low. An extraction does not need to catch every passing mention to score high — only the load-bearing facts.

Think step by step in one or two sentences per dimension, citing specific evidence. Then, as the very last thing in your reply, output a JSON object and nothing after it, with exactly this shape:

{"groundedness": <1-5>, "completeness": <1-5>, "notes": "<one sentence on the most significant issue found, or \"none\" if the extraction is clean>"}`,
}
