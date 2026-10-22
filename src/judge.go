package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/AssemblyAI/voice-standard-bench/internal/jsonrepair"
	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

// judgeMaxTokens bounds the judge's reply. A couple of sentences of
// chain-of-thought per rubric dimension plus a small trailing JSON object
// comfortably fits well under this on every judge model tried.
const judgeMaxTokens = 50000

// JudgeError wraps every RunJudge failure mode — the completion call itself
// failing (non-2xx, timeout, network error), a reply with no JSON object in
// it, or a reply whose JSON doesn't match the rubric — with the raw
// request/response bytes needed to reproduce and inspect it, the same way
// RunCompletion's own CompletionResult does for a candidate call. Request is
// always the verbatim JSON body sent to the judge model (so it shows exactly
// what transcript/candidate output the judge was asked to grade); Response
// is the verbatim raw JSON the gateway returned when there is one (the whole
// ChatCompletion, including the judge's free-text reply, for a parse
// failure) or the gateway's raw error body (for a call that failed
// outright) — see RawAPIError. Callers needing these fields to record a
// failure use errors.As, the same pattern RawAPIError itself uses to find
// the underlying *openai.Error.
type JudgeError struct {
	err      error
	Request  string
	Response string
}

func (e *JudgeError) Error() string { return e.err.Error() }
func (e *JudgeError) Unwrap() error { return e.err }

// RunJudge scores one task's candidate output against its source transcript
// using the task's judge prompt (spec.Judge), via judgeModel — a fixed model
// that must never be one of the candidates under test, or a model would end
// up grading its own (or a sibling's) output. It returns one score per
// rubric dimension named in spec.Rubric.
//
// The judge's reply is free text ending in a JSON object (see every Judge
// prompt in ./prompts), not a response_format-constrained completion: some
// providers behind the gateway don't support structured outputs at all, and
// picking a judge model would then implicitly constrain which providers can
// serve it. Parsing is therefore lenient — extractJSONObject grabs the
// outermost {...} regardless of any reasoning prose or code fence around it
// — and a judge reply that still doesn't parse is reported as a query error,
// not a score of zero: a model that refuses to grade cleanly isn't the same
// as a model that graded harshly. Every failure returned is a *JudgeError
// (see its doc comment) so a caller can save the judge input/output that
// produced it, not just the error message.
func RunJudge(ctx context.Context, gw *Gateway, judgeModel string, spec prompts.Spec, transcript, candidateOutput string) (map[string]float64, error) {
	userMessage := "TRANSCRIPT:\n" + transcript + "\n\nCANDIDATE OUTPUT:\n" + candidateOutput

	scores, _, err := runJudgeCall(ctx, gw, judgeModel, spec.Judge, userMessage, spec.Rubric)

	return scores, err
}

// runJudgeCall sends one judge prompt and parses its reply: the core of RunJudge.
func runJudgeCall(ctx context.Context, gw *Gateway, judgeModel, systemPrompt, userMessage string, rubric []string) (scores map[string]float64, notes string, err error) {
	result, err := gw.RunCompletion(ctx, ModelInfo{ID: judgeModel}, systemPrompt, userMessage, judgeMaxTokens)
	if err != nil {
		return nil, "", &JudgeError{
			err:      fmt.Errorf("judge call failed: %w", err),
			Request:  result.RawRequest,
			Response: RawAPIError(err),
		}
	}

	scores, notes, err = parseJudgeReply(result.Text, rubric)
	if err != nil {
		return nil, "", &JudgeError{
			err:      fmt.Errorf("judge %s: %w", judgeModel, err),
			Request:  result.RawRequest,
			Response: result.RawResponse,
		}
	}

	return scores, notes, nil
}

// parseJudgeScores extracts the outermost JSON object from text and reads
// off exactly the dimensions named in rubric, clamped to the prompts'
// documented 1-5 range. A judge that ignores the range (answers "10" or "0")
// is far more likely to have misread the instructions than to have found a
// real use for an out-of-range score, so this clamps rather than propagating
// a value the rubric never promised downstream into an average.
func parseJudgeScores(text string, rubric []string) (map[string]float64, error) {
	scores, _, err := parseJudgeReply(text, rubric)

	return scores, err
}

// parseJudgeReply is parseJudgeScores plus the judge's free-text "notes"
// field, when it sent one (empty otherwise -- it is optional and never
// scored).
func parseJudgeReply(text string, rubric []string) (map[string]float64, string, error) {
	object, err := extractJSONObject(text)
	if err != nil {
		return nil, "", fmt.Errorf("no JSON object in judge response: %w", err)
	}

	object, err = jsonrepair.JSONRepair(object)

	if err != nil {
		return nil, "", fmt.Errorf("failed to repair judge response json: %w", err)
	}

	var raw map[string]any
	if err := json.Unmarshal([]byte(object), &raw); err != nil {
		return nil, "", fmt.Errorf("judge response is not valid JSON: %w", err)
	}

	scores := make(map[string]float64, len(rubric))

	var missing []string

	for _, dim := range rubric {
		value, ok := raw[dim]
		if !ok {
			missing = append(missing, dim)
			continue
		}

		score, ok := value.(float64)
		if !ok {
			missing = append(missing, dim+" (not numeric)")
			continue
		}

		scores[dim] = clamp(score, 1, 5)
	}

	if len(missing) > 0 {
		return nil, "", fmt.Errorf("missing or invalid rubric field(s): %s", strings.Join(missing, ", "))
	}

	notes, _ := raw["notes"].(string)

	return scores, strings.TrimSpace(notes), nil
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}

	if v > hi {
		return hi
	}

	return v
}

// extractJSONObject returns the outermost {...} span in s. Judge and
// extraction prompts both ask for "only a JSON object, nothing else", but
// models routinely wrap it in a markdown code fence or prepend a sentence of
// reasoning despite that instruction; grabbing between the first '{' and the
// last '}' handles both without separate fence-detection logic.
func extractJSONObject(s string) (string, error) {
	start := strings.IndexByte(s, '{')
	end := strings.LastIndexByte(s, '}')

	if start < 0 || end < 0 || end < start {
		preview := s
		if len(preview) > 200 {
			preview = preview[:200]
		}

		return "", fmt.Errorf("no JSON object found in: %s", preview)
	}

	return s[start : end+1], nil
}

// Extraction is the information-extraction task's target schema, described
// in full inside prompts.InformationExtraction's task prompt. Every field is
// required to be PRESENT in the model's JSON (empty arrays are fine, missing
// keys are not — see ValidateExtraction), matching what the prompt tells the
// candidate model to do.
type Extraction struct {
	Participants []string           `json:"participants"`
	ActionItems  []ActionItem       `json:"action_items"`
	Decisions    []string           `json:"decisions"`
	Entities     ExtractionEntities `json:"entities"`
}

type ActionItem struct {
	Owner string `json:"owner"`
	Task  string `json:"task"`
	Due   string `json:"due"`
}

type ExtractionEntities struct {
	People        []string `json:"people"`
	Organizations []string `json:"organizations"`
	Products      []string `json:"products"`
	Dates         []string `json:"dates"`
}

// extractionRequiredKeys are the top-level keys the task prompt tells the
// model never to omit, even when a field has nothing to report.
var extractionRequiredKeys = []string{"participants", "action_items", "decisions", "entities"}

// ValidateExtraction checks that raw is a JSON object matching Extraction's
// schema with every required key present. A model wrapping its output in a
// markdown code fence, or prefacing it with a sentence of commentary despite
// being told not to, is tolerated the same way judge output is (see
// extractJSONObject) — but a genuinely missing key is not: the task prompt
// is explicit that an empty array, not an absent key, is how "nothing to
// report" is expressed, so a missing key means the model didn't follow the
// schema, not that it found nothing.
//
// This check is deliberately mechanical, with no model call involved: "is
// this valid JSON with these keys" doesn't need a judge's opinion, and a
// structurally invalid response should never reach the judge at all — grading
// the CONTENT of something that isn't even shaped right would produce a
// meaningless groundedness/completeness score.
func ValidateExtraction(raw string) (Extraction, error) {
	object, err := extractJSONObject(raw)
	if err != nil {
		return Extraction{}, fmt.Errorf("not a JSON object: %w", err)
	}

	var keys map[string]json.RawMessage
	if err := json.Unmarshal([]byte(object), &keys); err != nil {
		return Extraction{}, fmt.Errorf("not a JSON object: %w", err)
	}

	var missing []string

	for _, key := range extractionRequiredKeys {
		if _, ok := keys[key]; !ok {
			missing = append(missing, key)
		}
	}

	if len(missing) > 0 {
		return Extraction{}, fmt.Errorf("missing required key(s): %s", strings.Join(missing, ", "))
	}

	var extraction Extraction
	if err := json.Unmarshal([]byte(object), &extraction); err != nil {
		return Extraction{}, fmt.Errorf("does not match extraction schema: %w", err)
	}

	return extraction, nil
}
