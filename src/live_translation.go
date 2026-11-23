// Live translation: one record per streaming speech turn, translated by an
// LLM Gateway model the way the streaming API's llm_gateway parameter does it
// per turn, and graded against a professional reference translation.
//
// Records come from normalized/live_translation/*.jsonl, built by
// scripts/live_translation/prep.py from FLEURS: clips are joined into
// roughly two-minute streams, streamed once through the notetaker model, and
// each clip's words become one turn. The same clips also go through async
// speech understanding translation, kept on the record as the async baseline
// (see AsyncBaselineModelID).
//
// Buckets are language pairs ("en-es", "de-en"), not audio lengths, so every
// report row breaks down by pair. A dataset built with a streaming mode
// (prep.py --mode) adds it to the bucket ("en-es/max_accuracy").
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

// LiveTranslationRecord is one streaming turn to translate into one target
// language.
type LiveTranslationRecord struct {
	ID             string
	Bucket         string   // "<source>-<target>", e.g. "en-es"
	SourceLanguage string   // ISO 639-1, e.g. "en"
	TargetLanguage string   // ISO 639-1, e.g. "es"
	Turn           string   // the streaming transcript of this turn, as the notetaker model returned it
	History        []string // up to three previous turns of the same stream, oldest first
	SourceTruth    string   // the FLEURS transcription of what was said; never sent to a candidate
	Reference      string   // the professional translation; never sent to a candidate
	AsyncBaseline  string   // async speech understanding translation of the same clip
	Source         string
}

func (r LiveTranslationRecord) RecordBucket() string { return r.Bucket }
func (r LiveTranslationRecord) RecordID() string     { return r.ID }

type rawLiveTranslationRecord struct {
	ID    string `json:"id"`
	Input struct {
		SourceLanguage string   `json:"source_language"`
		TargetLanguage string   `json:"target_language"`
		Turn           string   `json:"turn"`
		History        []string `json:"history"`
		SourceTruth    string   `json:"source_truth"`
		Variant        string   `json:"variant"`
	} `json:"input"`
	Expected struct {
		Reference string `json:"reference"`
	} `json:"expected"`
	Baselines struct {
		AsyncSU string `json:"async_su_translation"`
	} `json:"baselines"`
}

// LoadLiveTranslationRecords reads every file matching glob. A line that
// doesn't parse or lacks a turn, languages or reference is skipped and
// counted, not fatal, the same convention as LoadDictationRecords. A turn the
// speech model returned empty is kept: dropping it would hide exactly the
// recognition misses the benchmark should show.
func LoadLiveTranslationRecords(glob string) (records []LiveTranslationRecord, skipped int, err error) {
	paths, err := filepath.Glob(glob)
	if err != nil {
		return nil, 0, fmt.Errorf("invalid glob %q: %w", glob, err)
	}

	if len(paths) == 0 {
		return nil, 0, fmt.Errorf("no files matched %q (build the dataset with scripts/live_translation/prep.py)", glob)
	}

	sort.Strings(paths)

	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return nil, skipped, err
		}

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}

			var raw rawLiveTranslationRecord
			if json.Unmarshal([]byte(line), &raw) != nil || raw.ID == "" || raw.Input.SourceLanguage == "" ||
				raw.Input.TargetLanguage == "" || raw.Expected.Reference == "" {
				skipped++
				continue
			}

			bucket := raw.Input.SourceLanguage + "-" + raw.Input.TargetLanguage
			if raw.Input.Variant != "" {
				bucket += "/" + raw.Input.Variant // e.g. en-es/max_accuracy, so a streaming mode gets its own report rows
			}

			records = append(records, LiveTranslationRecord{
				ID:             raw.ID,
				Bucket:         bucket,
				SourceLanguage: raw.Input.SourceLanguage,
				TargetLanguage: raw.Input.TargetLanguage,
				Turn:           raw.Input.Turn,
				History:        raw.Input.History,
				SourceTruth:    raw.Input.SourceTruth,
				Reference:      raw.Expected.Reference,
				AsyncBaseline:  raw.Baselines.AsyncSU,
				Source:         path,
			})
		}

		closeErr := f.Close()

		if err := scanner.Err(); err != nil {
			return nil, skipped, fmt.Errorf("reading %s: %w", path, err)
		}

		if closeErr != nil {
			return nil, skipped, closeErr
		}
	}

	return records, skipped, nil
}

// SampleLiveTranslationRecords keeps up to max records per language pair,
// picked by a hash of the record ID so the sample is stable across runs and
// doesn't favour the first streams. max <= 0 keeps everything.
func SampleLiveTranslationRecords(records []LiveTranslationRecord, max int) []LiveTranslationRecord {
	if max <= 0 {
		return records
	}

	byBucket := map[string][]LiveTranslationRecord{}
	for _, r := range records {
		byBucket[r.Bucket] = append(byBucket[r.Bucket], r)
	}

	buckets := make([]string, 0, len(byBucket))
	for b := range byBucket {
		buckets = append(buckets, b)
	}

	sort.Strings(buckets)

	var out []LiveTranslationRecord

	for _, b := range buckets {
		group := byBucket[b]
		sort.Slice(group, func(i, j int) bool { return idHash(group[i].ID) < idHash(group[j].ID) })

		if len(group) > max {
			group = group[:max]
		}

		out = append(out, group...)
	}

	return out
}

func idHash(id string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(id))

	return h.Sum64()
}

// languageNames maps the ISO codes the dataset uses to the names the prompt
// gives the model. An unknown code is passed through as is.
var languageNames = map[string]string{
	"en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
	"it": "Italian", "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "hi": "Hindi",
	"nl": "Dutch", "pl": "Polish", "ru": "Russian", "tr": "Turkish", "ar": "Arabic",
}

func languageName(code string) string {
	if name, ok := languageNames[code]; ok {
		return name
	}

	return code
}

// liveTranslationUserMessage builds the per-turn user message: the target
// language, the previous turns as context, then the turn. It mirrors what the
// streaming API's llm_gateway template sends with {{history_3}} and {{turn}}.
func liveTranslationUserMessage(r LiveTranslationRecord) string {
	var b strings.Builder

	fmt.Fprintf(&b, "Target language: %s\n\n", languageName(r.TargetLanguage))

	b.WriteString("Previous turns (context only):\n")

	if len(r.History) == 0 {
		b.WriteString("(none)\n")
	}

	for _, h := range r.History {
		b.WriteString(h)
		b.WriteString("\n")
	}

	b.WriteString("\nCurrent turn:\n")
	b.WriteString(r.Turn)

	return b.String()
}

// ChrF returns the character n-gram F-score (Popovic, 2015) of hypothesis
// against reference on a 0-100 scale: n-grams of 1 to 6 characters with
// whitespace removed, precision and recall averaged over n, and recall
// weighted twice as heavily as precision (beta = 2), the sacrebleu defaults.
// It works on characters, so it scores Japanese or Chinese output without a
// tokenizer. An n with no n-grams on either side is left out of the average,
// so a short sentence isn't scored zero for having no 6-grams.
func ChrF(hypothesis, reference string) float64 {
	const maxN = 6
	const beta = 2.0

	hyp := chrfChars(hypothesis)
	ref := chrfChars(reference)

	if len(ref) == 0 {
		if len(hyp) == 0 {
			return 100
		}

		return 0
	}

	var precSum, recSum float64

	counted := 0

	for n := 1; n <= maxN; n++ {
		hc, ht := charNgrams(hyp, n)
		rc, rt := charNgrams(ref, n)

		if ht == 0 && rt == 0 {
			continue
		}

		counted++

		if ht == 0 || rt == 0 {
			continue // contributes zero precision and recall
		}

		match := 0

		for g, c := range hc {
			if rcount := rc[g]; rcount < c {
				match += rcount
			} else {
				match += c
			}
		}

		precSum += float64(match) / float64(ht)
		recSum += float64(match) / float64(rt)
	}

	if counted == 0 {
		return 0
	}

	p := precSum / float64(counted)
	r := recSum / float64(counted)

	if p == 0 && r == 0 {
		return 0
	}

	b2 := beta * beta

	return 100 * (1 + b2) * p * r / (b2*p + r)
}

func chrfChars(s string) []rune {
	out := make([]rune, 0, len(s))

	for _, c := range s {
		if !unicode.IsSpace(c) {
			out = append(out, c)
		}
	}

	return out
}

func charNgrams(chars []rune, n int) (counts map[string]int, total int) {
	counts = map[string]int{}

	for i := 0; i+n <= len(chars); i++ {
		counts[string(chars[i:i+n])]++
		total++
	}

	return counts, total
}
