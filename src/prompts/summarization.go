package prompts

// summarizationSpec follows SummEval (Fabbri et al., 2021, "SummEval:
// Re-evaluating Summarization Evaluation") for WHAT to grade, and G-Eval
// (Liu et al., 2023, "G-Eval: NLG Evaluation using GPT-4 with Better Human
// Alignment") for HOW to grade it: state the criteria explicitly, ask for a
// short chain-of-thought per dimension, then a numeric score.
//
// SummEval's four human-eval dimensions are coherence, consistency, fluency,
// and relevance. Coherence and fluency are dropped here — modern
// instruction-tuned models essentially never fail either, so across a model
// sweep those two dimensions carry no signal. What's left is consistency
// (renamed "faithfulness": does the summary say only true things) and
// relevance (renamed "coverage": does it say the important things).
// "Conciseness" is added because call-transcript summaries fail open just as
// often as they fail closed — a summary can be faithful and complete while
// still padding out small talk and restating the same point, which
// faithfulness and coverage alone would not catch.
//
// These three names match the reference-free summarization metrics already
// defined on the Python normalization side of voice_bench (see the
// aai_transcripts adapter in ../../jobs/summarization.py), so a result from
// either half of the project uses the same vocabulary.
var summarizationSpec = Spec{
	Rubric: []string{"faithfulness", "coverage", "conciseness"},

	Task: `You are an expert call-transcript summarizer.

You will be given the raw transcript of a phone or video call. Write a summary that a busy person could read in place of the full call.

Rules:
- State only what is actually said in the transcript. Do not infer, assume, or add information that is not present.
- Capture the topics discussed, any decisions made, and any action items or commitments, in that rough order of importance.
- Omit small talk, filler words, and repeated statements.
- Write in plain prose, not bullet points, unless the call covers more than three distinct topics.
- Keep it short: a few sentences for a short call, a short paragraph for a long one. Do not pad the summary to sound more thorough than the call was.
- Output only the summary. No preamble, no "Here is a summary:", no closing remarks.`,

	Judge: `You are grading a call-transcript summary against the call it summarizes. You do not have a reference summary to compare against — grade the summary directly against the source transcript.

Score the summary on three dimensions, each from 1 (worst) to 5 (best):

- faithfulness: Does every claim in the summary actually appear in the transcript? A summary that adds, assumes, or misstates something not said in the call scores low here, even if the rest of the summary is good. A summary that says only true things, but very little of them, can still score high on faithfulness — that failure belongs to coverage instead.
- coverage: Does the summary capture the topics, decisions, and action items that a listener would consider the point of the call? Missing a major topic or a stated commitment scores low. A summary does not need to mention every detail to score high — only the load-bearing ones.
- conciseness: Is the summary free of padding — small talk, filler, repeated statements, or restating the same point in different words? A summary that is accurate and complete but bloated with restatement scores low here even if faithfulness and coverage are both high.

Think step by step about each dimension in one or two sentences, citing what in the transcript supports your score. Then, as the very last thing in your reply, output a JSON object and nothing after it, with exactly this shape:

{"faithfulness": <1-5>, "coverage": <1-5>, "conciseness": <1-5>, "notes": "<one sentence on the most significant issue found, or \"none\" if the summary is clean>"}`,
}
