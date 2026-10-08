// Dictation is the fourth task this benchmark runs. Every record is sent the
// exact SAME fixed system prompt — prompts.dictationSpec.Task
// (prompts/dictation.go), describing the spoken-command grammar being tested
// ("'comma' -> ','; ... 'all caps <word>' uppercases that one word ...") —
// exactly like the other three tasks' prompts.Spec.Task. dictation/
// synthetic.jsonl also carries a per-record input.instruction field, but it
// is deliberately never read (see rawDictationRecord below): that shared
// prompt is the only one ever sent, regardless of what that field happens to
// say for any given record.
//
// What keeps dictation off runOne/runModel's shared pipeline anyway is
// grading. Every record carries a real answer key (expected.truth_text) —
// the known-correct formatting of input.text — so grading is entirely
// mechanical (ScoreDictation, below): word error rate, exact match, and
// punctuation/capitalization/formatting-command accuracy, all computed
// against truth_text in code. There is no judge model call for this task at
// all, unlike every other task's RunJudge (judge.go), which runOne always
// makes one of — that's the one thing dictation still can't share a branch
// of runOne for.
//
// That difference is why this file has its own loader (LoadDictationRecords,
// alongside dataset.go's LoadTranscripts — dictation/synthetic.jsonl's
// schema and filename both differ from aai_transcripts.jsonl's anyway), its
// own runner (runDictationForModel/runDictationOne, alongside runModel/
// runOne in voicebench.go), and its own scorer (ScoreDictation, alongside
// judge.go's RunJudge). What it does NOT need its own version of: Sample,
// Cell, Aggregate, PrintReport, WriteJSONReport, or anything in summary.go's
// per-task quality breakdown — ScoreDictation's five mechanical scores are
// written into the exact same Sample.Judge map[string]float64 an LLM judge
// would fill in, so every one of those already-generic-over-Task functions
// handles a dictation Sample with no change (see summary.go's dimensionRange
// for the one adjustment dictation's scores DID need: telling normalize()
// how to fold word_error_rate's "lower is better" percentage, and the other
// three's plain 0-100 percentages, into its [0,1] convention).
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
)

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

// DictationRecord is the slice of dictation/synthetic.jsonl's schema this
// benchmark needs. Unlike TranscriptRecord (dataset.go), it carries the
// answer key (TruthText) — the other three tasks' records have no
// equivalent — plus Metrics, which selects which of ScoreDictation's five
// scores actually apply to this record. It deliberately does NOT carry the
// dataset's own input.instruction field: every record is sent the same
// shared prompt (prompts.dictationSpec.Task, see runDictationOne) regardless
// of what that field says, so there is nothing useful to keep it for (see
// rawDictationRecord, which doesn't decode it either).
type DictationRecord struct {
	ID        string
	Bucket    string // input.context.audio_length_bucket, e.g. "30s", "120s"
	Input     string // input.text -- the raw dictation transcript (spoken commands included), sent as the user message
	TruthText string // expected.truth_text -- the answer key; never sent to the model, only to ScoreDictation
	Metrics   []string
	Source    string // file this record was read from, for error messages
}

// RecordBucket implements framework.go's Bucketed, letting Runner[DictationRecord]
// restrict a TTFT pass to TTFTBuckets generically.
func (r DictationRecord) RecordBucket() string {
	return r.Bucket
}

// rawDictationRecord decodes only the fields of dictation/synthetic.jsonl
// this tool reads. Unknown fields (schema_version, dataset, split, language,
// domain, tags, source, ...) are ignored by encoding/json, not an error —
// same convention as dataset.go's rawRecord. input.instruction is one such
// deliberately-ignored field: every dictation record is sent the SAME fixed
// prompt (prompts.dictationSpec.Task — see runDictationOne) rather than
// whatever that per-record field happens to say, so it is left undecoded
// rather than read and discarded.
type rawDictationRecord struct {
	ID    string `json:"id"`
	Input struct {
		Text    string `json:"text"`
		Context struct {
			AudioLengthBucket string `json:"audio_length_bucket"`
		} `json:"context"`
	} `json:"input"`
	Expected struct {
		TruthText string `json:"truth_text"`
	} `json:"expected"`
	Metrics []string `json:"metrics"`
}

// LoadDictationRecords reads every file matching glob (e.g.
// ".../normalized/dictation/*.jsonl") into DictationRecords. It mirrors
// LoadTranscripts (dataset.go) in shape — glob, skip-and-count malformed or
// incomplete lines rather than aborting the whole file, sort paths first for
// deterministic load order — but decodes a genuinely different schema
// (truth_text, which LoadTranscripts has no notion of at all) from a
// differently-named file (synthetic.jsonl today, never
// aai_transcripts.jsonl), so it is its own function rather than a mode of
// LoadTranscripts. A record missing its text, bucket, or truth_text is
// skipped and counted: there is nothing to benchmark without input to send
// or a truth to grade against.
func LoadDictationRecords(glob string) (records []DictationRecord, skipped int, err error) {
	paths, err := filepath.Glob(glob)
	if err != nil {
		return nil, 0, fmt.Errorf("invalid glob %q: %w", glob, err)
	}

	if len(paths) == 0 {
		return nil, 0, fmt.Errorf("no files matched %q", glob)
	}

	sort.Strings(paths)

	for _, path := range paths {
		fileRecords, fileSkipped, err := loadDictationFile(path)
		skipped += fileSkipped

		if err != nil {
			return nil, skipped, fmt.Errorf("%s: %w", path, err)
		}

		records = append(records, fileRecords...)
	}

	return records, skipped, nil
}

func loadDictationFile(path string) (records []DictationRecord, skipped int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}

		var rec rawDictationRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			skipped++
			continue
		}

		text := strings.TrimSpace(rec.Input.Text)
		bucket := rec.Input.Context.AudioLengthBucket
		truth := strings.TrimSpace(rec.Expected.TruthText)

		if text == "" || bucket == "" || truth == "" {
			skipped++
			continue
		}

		records = append(records, DictationRecord{
			ID:        rec.ID,
			Bucket:    bucket,
			Input:     text,
			TruthText: truth,
			Metrics:   rec.Metrics,
			Source:    path,
		})
	}

	return records, skipped, scanner.Err()
}

// SampleDictationRecords caps how many records per bucket are benchmarked,
// chosen deterministically — sorted by record ID, then truncated — so repeat
// runs sample the same records and results stay comparable. Mirrors
// SampleByBucket (dataset.go) exactly; kept separate only because it operates
// on DictationRecord rather than TranscriptRecord. max <= 0 means no cap.
func SampleDictationRecords(records []DictationRecord, max int) []DictationRecord {
	if max <= 0 {
		return records
	}

	byBucket := map[string][]DictationRecord{}
	for _, r := range records {
		byBucket[r.Bucket] = append(byBucket[r.Bucket], r)
	}

	buckets := make([]string, 0, len(byBucket))
	for b := range byBucket {
		buckets = append(buckets, b)
	}

	sort.Strings(buckets) // just for deterministic output order; final report order comes from Aggregate, not this

	var sampled []DictationRecord

	for _, bucket := range buckets {
		recs := append([]DictationRecord(nil), byBucket[bucket]...)
		sort.Slice(recs, func(i, j int) bool { return recs[i].ID < recs[j].ID })

		if len(recs) > max {
			recs = recs[:max]
		}

		sampled = append(sampled, recs...)
	}

	return sampled
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

// runDictationForModel runs every dictation record against one model,
// bounded to concurrency simultaneous in-flight requests — mirroring
// runModel's shape for the other three tasks, but with no judgeSem: there is
// no judge call on this path at all (see this file's package doc comment),
// so there is no separate concurrency budget to bound. The dictation
// prompts.Spec is looked up once, outside the per-record loop, since every
// record is sent that exact same system prompt (see runDictationOne).
func runDictationForModel(ctx context.Context, gw *Gateway, model ModelInfo, records []DictationRecord, concurrency int, failures *FailureRecorder) []Sample {
	spec, ok := prompts.For(prompts.Dictation)
	if !ok {
		// Unreachable in practice — prompts/dictation.go always registers
		// dictationSpec in specs — but skip this model's dictation records
		// rather than send every one of them an empty system prompt if that
		// registration is ever removed.
		log.Printf("model %s: no prompts.Spec registered for %q, skipping dictation", model.ID, prompts.Dictation)
		return nil
	}

	if concurrency < 1 {
		concurrency = 1
	}

	sem := make(chan struct{}, concurrency)

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		samples []Sample
	)

	for _, record := range records {
		wg.Add(1)
		sem <- struct{}{}

		go func(record DictationRecord) {
			defer wg.Done()
			defer func() { <-sem }()

			sample := runDictationOne(ctx, gw, model, spec, record, failures)

			mu.Lock()
			samples = append(samples, sample)
			mu.Unlock()
		}(record)
	}

	wg.Wait()

	return samples
}

// runDictationOne runs one dictation record against one model and scores the
// result mechanically against record.TruthText (see ScoreDictation) — never
// a judge call, since a real answer key makes one unnecessary (see this
// file's package doc comment). The candidate's system prompt is spec.Task —
// the SAME fixed prompt sent for every dictation record (see
// runDictationForModel) — never the dataset's own per-record
// input.instruction field, which this file never even decodes (see
// rawDictationRecord). Context-length guarding and failure recording reuse
// the exact same helpers runOne (voicebench.go) uses for the other tasks'
// API-error case — this path just never reaches runOne's schema-validation
// or judge-call steps, neither of which apply here.
func runDictationOne(ctx context.Context, gw *Gateway, model ModelInfo, spec prompts.Spec, record DictationRecord, failures *FailureRecorder) Sample {
	sample := Sample{Model: model.ID, Task: prompts.Dictation, Bucket: record.Bucket, RecordID: record.ID, ReasoningEffort: model.ReasoningEffortLabel()}

	if contextLengthExceeded(model, spec.Task, record.Input) {
		sample.Err = fmt.Errorf("estimated prompt tokens exceed model context length (%d)", model.ContextLength)
		return sample
	}

	result, err := gw.RunCompletion(ctx, model, spec.Task, record.Input, taskMaxTokens(prompts.Dictation, record.Input, model))
	if err != nil {
		sample.Err = err

		if failures != nil {
			recordFailure(failures, model.ID, model.ReasoningEffortLabel(), prompts.Dictation, record.Bucket, record.ID, FailureAPIError, err, result.RawRequest, RawAPIError(err))
		}

		return sample
	}

	sample.TotalLatency = result.TotalLatency
	sample.TimeToFirstToken = result.TimeToFirstToken
	sample.CompletionTokens = result.CompletionTokens
	sample.PromptTokens = result.PromptTokens
	sample.Judge = ScoreDictation(record.TruthText, result.Text, record.Metrics)

	return sample
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

// breakKind records what, if anything, separated a token from the one before
// it in the original text — distinct from "new paragraph"/"new line" as
// spoken COMMANDS (which only ever appear in a record's raw Input, never in
// TruthText or a candidate's output) — this is the STRUCTURAL break those
// commands are supposed to produce in the final formatted text, detected by
// scanning for it directly rather than by re-parsing the command grammar.
type breakKind int

const (
	breakNone breakKind = iota
	breakLine
	breakParagraph
)

// dictationToken is one whitespace-delimited word from a dictation-formatted
// text (TruthText or a candidate's output — never the raw, command-laden
// Input, which this file never tokenizes), split into the pieces each metric
// needs: normalized/surface for word identity and case, leading/trailing for
// attached punctuation, break_ for the paragraph/line break immediately
// before it.
type dictationToken struct {
	normalized string    // lowercase core, compared case-insensitively for alignment (see alignTokens)
	surface    string    // case-preserved core, e.g. "THAI", "Jordan's", "all-hands" -- may contain internal hyphens/apostrophes
	leading    string    // non-alphanumeric runes attached immediately before the core, e.g. `"`, `(`
	trailing   string    // non-alphanumeric runes attached immediately after the core, e.g. `,`, `.`, `.")`
	break_     breakKind // paragraph/line break immediately preceding this token, if any
}

// paragraphBreakRE splits text into paragraphs on any blank line (one or more
// newlines, optionally with in-between whitespace) — the shape "New
// paragraph" produces in every TruthText observed in
// dictation/synthetic.jsonl (e.g. "...priorities.\n\nFirst, we will...").
var paragraphBreakRE = regexp.MustCompile(`\n\s*\n`)

// isWordRune reports whether r can be part of a dictationToken's core.
// Restricting this to letters/digits (never '-' or '\”) is what lets
// splitPunct treat an internal hyphen ("all-hands") or apostrophe ("Jordan's",
// "don't") as part of the core rather than trailing punctuation to strip —
// both only ever appear SURROUNDED by word runes, so they naturally fall
// inside [start, end] below without being listed as word runes themselves.
func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

// splitPunct splits field (one whitespace-delimited token from the original
// text) into leading non-word runes, a core spanning from the first to the
// last word rune (so an internal hyphen or apostrophe stays part of core),
// and trailing non-word runes. A field with no word rune at all — in
// practice never produced by the dictation grammar, which always attaches
// wrapping punctuation to an adjacent word — returns an empty core and the
// whole field as leading; tokenizeText folds that case onto a neighboring
// real token rather than inventing a punctuation-only token kind for a shape
// that doesn't occur.
func splitPunct(field string) (leading, core, trailing string) {
	runes := []rune(field)
	start, end := -1, -1

	for i, r := range runes {
		if isWordRune(r) {
			if start == -1 {
				start = i
			}

			end = i
		}
	}

	if start == -1 {
		return field, "", ""
	}

	return string(runes[:start]), string(runes[start : end+1]), string(runes[end+1:])
}

// tokenizeText splits a dictation-formatted text (TruthText or a candidate's
// output) into dictationTokens, tracking paragraph/line structure alongside
// the usual word/punctuation split. A field with no word rune (see
// splitPunct) is folded onto the previous token's trailing (or, if it's the
// very first thing in the text, held as a leading prefix for the next real
// token) rather than becoming its own token — every punctuation mark ends up
// attributed to some word, matching how the metrics in this file read
// leading/trailing off of dictationTokens, never off a separate punctuation-
// token kind.
func tokenizeText(text string) []dictationToken {
	var tokens []dictationToken

	var pendingLeading string

	pendingBreak := breakNone

	for p, para := range paragraphBreakRE.Split(text, -1) {
		if p > 0 && len(tokens) > 0 {
			pendingBreak = breakParagraph
		}

		for l, line := range strings.Split(para, "\n") {
			if l > 0 && len(tokens) > 0 && pendingBreak == breakNone {
				pendingBreak = breakLine
			}

			for _, field := range strings.Fields(line) {
				leading, core, trailing := splitPunct(field)

				if core == "" {
					if len(tokens) > 0 {
						tokens[len(tokens)-1].trailing += leading + trailing
					} else {
						pendingLeading += leading + trailing
					}

					continue
				}

				tokens = append(tokens, dictationToken{
					normalized: strings.ToLower(core),
					surface:    core,
					leading:    pendingLeading + leading,
					trailing:   trailing,
					break_:     pendingBreak,
				})

				pendingLeading = ""
				pendingBreak = breakNone
			}
		}
	}

	return tokens
}

// ---------------------------------------------------------------------------
// Word alignment (shared by every alignment-based metric below)
// ---------------------------------------------------------------------------

type alignOp int

const (
	opMatch alignOp = iota
	opSub
	opDel // a ref (truth) token with no counterpart in hyp (candidate)
	opIns // a hyp (candidate) token with no counterpart in ref (truth)
)

// tokenAlign is one step of the alignment between ref and hyp: refIdx/hypIdx
// index into the token slices alignTokens was called with, -1 marking the
// side an opDel/opIns has nothing on.
type tokenAlign struct {
	op             alignOp
	refIdx, hypIdx int
}

// alignTokens computes the minimum-edit-distance word-level alignment
// between ref (truth) and hyp (candidate) — standard Levenshtein DP, comparing
// tokens by their normalized (lowercase) form so a pure case difference is a
// match, not a substitution (see dictationToken.normalized) — and returns
// both the edit count (= word_error_rate's numerator) and the full backtrace.
// Every other metric in this file (capitalization/punctuation/
// formatting_command accuracy) reuses this SAME backtrace rather than
// computing its own alignment, so all four scores are judging the model
// against the same understanding of "which candidate word corresponds to
// which truth word", not four independently-drifting comparisons.
func alignTokens(ref, hyp []dictationToken) (edits int, ops []tokenAlign) {
	n, m := len(ref), len(hyp)

	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}

	for i := 0; i <= n; i++ {
		dp[i][0] = i
	}

	for j := 0; j <= m; j++ {
		dp[0][j] = j
	}

	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if ref[i-1].normalized == hyp[j-1].normalized {
				dp[i][j] = dp[i-1][j-1]
				continue
			}

			dp[i][j] = 1 + min3(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
		}
	}

	i, j := n, m

	var reversed []tokenAlign

	for i > 0 || j > 0 {
		switch {
		case i > 0 && j > 0 && ref[i-1].normalized == hyp[j-1].normalized && dp[i][j] == dp[i-1][j-1]:
			reversed = append(reversed, tokenAlign{op: opMatch, refIdx: i - 1, hypIdx: j - 1})
			i--
			j--
		case i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1]+1:
			reversed = append(reversed, tokenAlign{op: opSub, refIdx: i - 1, hypIdx: j - 1})
			i--
			j--
		case i > 0 && dp[i][j] == dp[i-1][j]+1:
			reversed = append(reversed, tokenAlign{op: opDel, refIdx: i - 1, hypIdx: -1})
			i--
		default:
			reversed = append(reversed, tokenAlign{op: opIns, refIdx: -1, hypIdx: j - 1})
			j--
		}
	}

	ops = make([]tokenAlign, len(reversed))
	for k, op := range reversed {
		ops[len(reversed)-1-k] = op
	}

	return dp[n][m], ops
}

func min3(a, b, c int) int {
	m := a
	if b < m {
		m = b
	}

	if c < m {
		m = c
	}

	return m
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

// terminalPunct is the set of characters punctuation_accuracy grades —
// exactly the punctuation-producing spoken commands (comma/period/question
// mark/exclamation point/colon/semicolon). wrapPunct is the set
// formatting_command_accuracy grades instead — quotes and parens, which wrap
// a span rather than terminate one. Every dictation command that produces a
// literal punctuation mark falls into exactly one of the two sets, so a
// single trailing/leading string can be routed to the right metric with a
// simple filter (see filterRunes) rather than the tokenizer needing to know
// which spoken command produced it.
const (
	terminalPunct = ",.?!:;"
	wrapPunct     = `"()`
)

func filterRunes(s, allowed string) string {
	var b strings.Builder

	for _, r := range s {
		if strings.ContainsRune(allowed, r) {
			b.WriteRune(r)
		}
	}

	return b.String()
}

// checkAccumulator tallies a percent-correct score across many (want, got)
// position-by-position comparisons — shared by punctuationAccuracy,
// capitalizationAccuracy, and formattingCommandAccuracy so the three
// metrics' "how do we turn many small comparisons into one percentage"
// logic lives in one place. A position where want and got are BOTH empty is
// skipped entirely rather than counted as a trivial match — most word
// positions carry no wrapping punctuation, hyphen, possessive, or break at
// all, and counting every one of those as "correct" would dilute the score
// toward 100 and hide the real errors among the positions that actually
// carry the feature being graded.
type checkAccumulator struct {
	total, correct int
}

func (c *checkAccumulator) add(want, got string) {
	if want == "" && got == "" {
		return
	}

	c.total++

	if want == got {
		c.correct++
	}
}

func (c *checkAccumulator) percent() (score float64, applicable bool) {
	if c.total == 0 {
		return 0, false
	}

	return 100 * float64(c.correct) / float64(c.total), true
}

// exactMatch reports whether candidate matches truth exactly, ignoring only
// leading/trailing whitespace (a harmless artifact of LLM output formatting,
// not a formatting error the dataset's metrics are meant to catch).
func exactMatch(truth, candidate string) bool {
	return strings.TrimSpace(truth) == strings.TrimSpace(candidate)
}

// wordErrorRate is edits (substitutions + deletions + insertions, from
// alignTokens) over the number of truth words, as a percentage — the
// standard WER definition. It is computed over dictationToken.normalized
// (lowercase, punctuation and paragraph/line structure irrelevant), so a
// capitalization or punctuation mistake never shows up here — those are
// capitalizationAccuracy's and punctuationAccuracy's jobs; this metric is
// purely "did the right words, in the right order, come out."
func wordErrorRate(refCount, edits int) float64 {
	if refCount == 0 {
		return 0
	}

	return 100 * float64(edits) / float64(refCount)
}

// capitalizationAccuracy scores, for every truth word that has a letter to
// capitalize (a pure-number or pure-punctuation token has nothing to
// score), whether the candidate reproduced that EXACT word with the EXACT
// same casing. A truth word the alignment could only substitute or delete
// (op != opMatch) counts as incorrect here too, not merely "not applicable"
// — the word didn't come out right at all, so its capitalization certainly
// didn't either; this is a stricter, unconditional denominator (every
// letter-bearing truth word counts) rather than one that only judges
// casing among words the model already got right, which would silently
// exclude a model's worst errors from its capitalization score.
func capitalizationAccuracy(ref, hyp []dictationToken, ops []tokenAlign) (float64, bool) {
	var acc checkAccumulator

	for _, op := range ops {
		if op.refIdx < 0 {
			continue // a hyp-only insertion has no truth word to score capitalization against
		}

		r := ref[op.refIdx]
		if !hasLetter(r.surface) {
			continue
		}

		var gotSurface string
		if op.op == opMatch {
			gotSurface = hyp[op.hypIdx].surface
		}

		acc.add(r.surface, gotSurface)
	}

	return acc.percent()
}

func hasLetter(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}

	return false
}

// punctuationAccuracy scores, for every truth word carrying a terminal
// punctuation mark (comma/period/question mark/exclamation point/colon/
// semicolon — see terminalPunct), whether the aligned candidate word carries
// the exact same mark(s). A truth word the alignment substituted or deleted
// contributes an empty "got" side, which only counts against the score if
// the truth word DID expect punctuation there (checkAccumulator skips a
// want=="" && got=="" position instead of trivially crediting it) — the
// same stricter, unconditional-among-applicable-positions treatment
// capitalizationAccuracy uses. A hyp-only insertion is skipped: there is no
// truth position to check its punctuation against.
func punctuationAccuracy(ref, hyp []dictationToken, ops []tokenAlign) (float64, bool) {
	var acc checkAccumulator

	for _, op := range ops {
		if op.refIdx < 0 {
			continue
		}

		want := filterRunes(ref[op.refIdx].trailing, terminalPunct)

		var got string
		if op.hypIdx >= 0 {
			got = filterRunes(hyp[op.hypIdx].trailing, terminalPunct)
		}

		acc.add(want, got)
	}

	return acc.percent()
}

// formattingCommandAccuracy scores the dictation commands that are neither
// terminal punctuation nor plain capitalization: quote/parenthesis wrapping
// (leading/trailing " ( )), hyphenated compounds ("all-hands"), possessive
// 's ("Jordan's"), and paragraph/line breaks. Every check is folded into one
// checkAccumulator rather than five separate percentages — the dataset
// names this a single metric, not five — using the same want/got,
// unconditional-among-applicable-positions rule the other two mechanical
// metrics use (see their doc comments): a truth word the model never
// reproduced (substituted or deleted) automatically fails every formatting
// feature IT carried, and a spurious feature the model added where truth had
// none (e.g. an unwanted paragraph break) fails it too, since checkAccumulator
// only skips a position where BOTH sides are empty.
func formattingCommandAccuracy(ref, hyp []dictationToken, ops []tokenAlign) (float64, bool) {
	var acc checkAccumulator

	for _, op := range ops {
		if op.refIdx < 0 {
			continue
		}

		r := ref[op.refIdx]

		var h dictationToken
		if op.hypIdx >= 0 {
			h = hyp[op.hypIdx]
		}

		acc.add(filterRunes(r.leading, wrapPunct), filterRunes(h.leading, wrapPunct))
		acc.add(filterRunes(r.trailing, wrapPunct), filterRunes(h.trailing, wrapPunct))
		acc.add(hyphenMarker(r.surface), hyphenMarker(h.surface))
		acc.add(possessiveMarker(r.surface), possessiveMarker(h.surface))
		acc.add(breakMarker(r.break_), breakMarker(h.break_))
	}

	return acc.percent()
}

func hyphenMarker(surface string) string {
	if strings.ContainsRune(surface, '-') {
		return "-"
	}

	return ""
}

// possessiveMarker reports whether surface ends in a possessive 's (the
// "apostrophe s" command's output), case-insensitively so "CDC's"/"cdc's"
// are treated the same. It does not distinguish a possessive from a
// contraction ("it's", "don't") — both are just "an internal apostrophe" to
// this file's tokenizer (see isWordRune/splitPunct) — but a contraction is
// already part of the word's own spelling and so already graded by
// capitalizationAccuracy's/wordErrorRate's exact-surface comparison; this
// marker exists only to give the "apostrophe s" command itself a place to be
// graded under formatting_command_accuracy specifically.
func possessiveMarker(surface string) string {
	if len(surface) >= 2 && strings.EqualFold(surface[len(surface)-2:], "'s") {
		return "'s"
	}

	return ""
}

func breakMarker(b breakKind) string {
	switch b {
	case breakParagraph:
		return "¶"
	case breakLine:
		return "⏎"
	default:
		return ""
	}
}

// ScoreDictation mechanically scores candidate against truth for exactly the
// metric names listed in metrics (a dictation record's own "metrics" field —
// see DictationRecord), returning them as a Sample.Judge-shaped map so this
// task's results flow through the same Aggregate/summary pipeline every
// LLM-judged task's scores do (see summary.go's dimensionRange for the
// dictation-specific normalization those five names need). An unrecognized
// metric name is silently skipped rather than an error — the dataset's own
// "metrics" field is the source of truth for what to compute, and a name a
// future dataset adds that this version of the code doesn't know how to
// score yet should degrade to "not measured" for that one name, not fail the
// whole sample.
func ScoreDictation(truth, candidate string, metrics []string) map[string]float64 {
	want := make(map[string]bool, len(metrics))
	for _, m := range metrics {
		want[m] = true
	}

	scores := map[string]float64{}

	if want["exact_match"] {
		scores["exact_match"] = 0
		if exactMatch(truth, candidate) {
			scores["exact_match"] = 1
		}
	}

	needsAlignment := want["word_error_rate"] || want["capitalization_accuracy"] ||
		want["punctuation_accuracy"] || want["formatting_command_accuracy"]
	if !needsAlignment {
		return scores
	}

	refTokens := tokenizeText(truth)
	hypTokens := tokenizeText(candidate)
	edits, ops := alignTokens(refTokens, hypTokens)

	if want["word_error_rate"] {
		scores["word_error_rate"] = wordErrorRate(len(refTokens), edits)
	}

	if want["capitalization_accuracy"] {
		if v, ok := capitalizationAccuracy(refTokens, hypTokens, ops); ok {
			scores["capitalization_accuracy"] = v
		}
	}

	if want["punctuation_accuracy"] {
		if v, ok := punctuationAccuracy(refTokens, hypTokens, ops); ok {
			scores["punctuation_accuracy"] = v
		}
	}

	if want["formatting_command_accuracy"] {
		if v, ok := formattingCommandAccuracy(refTokens, hypTokens, ops); ok {
			scores["formatting_command_accuracy"] = v
		}
	}

	return scores
}
