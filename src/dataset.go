package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// TranscriptRecord is the slice of the voice_bench normalized record schema
// (../schema.py) this benchmark actually needs: the transcript text, the
// audio-length bucket it was tagged with, and — for text_correction only —
// the simulated corrupted counterpart of that text (MissedText). Every other
// field in the normalized JSONL — expected, tags, source, metrics — is
// irrelevant here and deliberately left undecoded.
type TranscriptRecord struct {
	ID     string
	Text   string
	Bucket string // input.context.audio_length_bucket, e.g. "30s", "3600s"
	Source string // file this record was read from, for error messages
	// MissedText is a simulated corrupted transcript — Text is the known-true
	// original it was derived from. Only text_correction/aai_transcripts.jsonl
	// carries this field today; it is empty for every other record, including
	// a handful of text_correction rows the simulation didn't cover (see
	// taskInput in main.go, which is where a missing MissedText becomes a
	// per-sample error, not a load-time one — MissedText being empty is a
	// normal, expected shape for a summarization/extraction record, so
	// LoadTranscripts must not treat it as a defect the way it does an empty
	// Text or Bucket).
	MissedText string
}

// RecordBucket implements framework.go's Bucketed, letting Runner[TranscriptRecord]
// restrict a TTFT pass to TTFTBuckets generically.
func (r TranscriptRecord) RecordBucket() string {
	return r.Bucket
}

// RecordID implements framework.go's Record, letting Runner[TranscriptRecord]
// find an earlier run's stashed result for this record (see RunConfig.Replay).
func (r TranscriptRecord) RecordID() string {
	return r.ID
}

// rawRecord decodes only the fields of the normalized schema this tool reads.
// Unknown fields (expected, tags, source, metrics, ...) are ignored by
// encoding/json, not an error.
type rawRecord struct {
	ID         string `json:"id"`
	MissedText string `json:"missed_text"`
	Input      struct {
		Text    string `json:"text"`
		Context struct {
			AudioLengthBucket string `json:"audio_length_bucket"`
		} `json:"context"`
	} `json:"input"`
}

// LoadTranscripts reads every file matching glob and returns their records
// grouped first by the TASK whose normalized folder they came from, then by
// audio-length bucket ("30s", "120s", "900s", "3600s", ...) within that task.
//
// The glob is the ONLY thing that decides what counts as a benchmark input.
// Each task's glob is
// "normalized/<task>/aai_transcripts.jsonl" — one folder per task
// (summarization/, information_extraction/, text_correction/). A record's
// task is taken from the name of the folder its aai_transcripts.jsonl sits
// in, which matches prompts.Summarization / prompts.InformationExtraction /
// prompts.TextCorrection exactly. Keeping task as the outer grouping key —
// rather than merging every matched file into one flat bucket map — is what
// keeps, say, text_correction's transcripts from ever being used to
// benchmark summarization just because both files matched the same glob.
// Any other file in a normalized folder (unified.jsonl,
// unified_reference_free.jsonl, mederrbench.jsonl, ...) is never matched and
// never loaded, per the instruction to reference only aai_transcripts.jsonl.
//
// A line that fails to parse, or parses but is missing text or a bucket, is
// skipped and counted rather than aborting the whole file — one bad line in a
// multi-thousand-line export should not sink every model's benchmark run. An
// unreadable FILE (missing, permission denied, truncated mid-record) is
// still a hard error: unlike a bad line, that is not an expected shape of
// noise in this data.
func LoadTranscripts(glob string) (byTaskBucket map[string]map[string][]TranscriptRecord, skipped int, err error) {
	paths, err := filepath.Glob(glob)
	if err != nil {
		return nil, 0, fmt.Errorf("invalid glob %q: %w", glob, err)
	}

	if len(paths) == 0 {
		return nil, 0, fmt.Errorf("no files matched %q", glob)
	}

	sort.Strings(paths) // deterministic load order, so sampling below is reproducible

	byTaskBucket = map[string]map[string][]TranscriptRecord{}

	for _, path := range paths {
		task := filepath.Base(filepath.Dir(path))

		byBucket, ok := byTaskBucket[task]
		if !ok {
			byBucket = map[string][]TranscriptRecord{}
			byTaskBucket[task] = byBucket
		}

		fileSkipped, err := loadFile(path, byBucket)
		skipped += fileSkipped

		if err != nil {
			return nil, skipped, fmt.Errorf("%s: %w", path, err)
		}
	}

	return byTaskBucket, skipped, nil
}

func loadFile(path string, byBucket map[string][]TranscriptRecord) (skipped int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// The 3600s bucket alone runs to ~90KB of transcript text per line;
	// bufio's default 64KB max token size would truncate-error on those.
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}

		var rec rawRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			skipped++
			continue
		}

		text := strings.TrimSpace(rec.Input.Text)
		bucket := rec.Input.Context.AudioLengthBucket

		if text == "" || bucket == "" {
			skipped++
			continue
		}

		byBucket[bucket] = append(byBucket[bucket], TranscriptRecord{
			ID:         rec.ID,
			Text:       text,
			Bucket:     bucket,
			Source:     path,
			MissedText: strings.TrimSpace(rec.MissedText),
		})
	}

	return skipped, scanner.Err()
}

// SampleByBucket returns up to max records per bucket, chosen deterministically
// — sorted by record ID, then truncated — so repeat runs sample the same
// transcripts and results stay comparable across models and across runs.
// max <= 0 means no cap (use every loaded record).
func SampleByBucket(byBucket map[string][]TranscriptRecord, max int) map[string][]TranscriptRecord {
	if max <= 0 {
		return byBucket
	}

	sampled := make(map[string][]TranscriptRecord, len(byBucket))

	for bucket, records := range byBucket {
		sorted := append([]TranscriptRecord(nil), records...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

		if len(sorted) > max {
			sorted = sorted[:max]
		}

		sampled[bucket] = sorted
	}

	return sampled
}

// SampleByTaskAndBucket applies SampleByBucket within each task
// independently, so a cap on how many transcripts of one task's bucket are
// used never pulls in — or is affected by the count of — any other task's
// records, same as LoadTranscripts never mixes them in the first place.
func SampleByTaskAndBucket(byTaskBucket map[string]map[string][]TranscriptRecord, max int) map[string]map[string][]TranscriptRecord {
	sampled := make(map[string]map[string][]TranscriptRecord, len(byTaskBucket))

	for task, byBucket := range byTaskBucket {
		sampled[task] = SampleByBucket(byBucket, max)
	}

	return sampled
}

// TaskGroup is every transcript one task should be run against, for one
// audio-length bucket — the unit runModel iterates over. Building this list
// once up front (see GroupByTaskAndBucket) keeps "what work exists" a fixed,
// inspectable data structure rather than something the run loop reconstructs
// by nesting a task loop inside a bucket loop inside a record loop at run
// time; the Records slice itself is untouched (same TranscriptRecords
// LoadTranscripts produced), only regrouped.
type TaskGroup struct {
	Task    string
	Bucket  string
	Records []TranscriptRecord
}

// GroupByTaskAndBucket lays byTaskBucket's records out as one TaskGroup per
// (task, bucket) pair: tasks in the order given, and within each task,
// buckets in duration order (see Buckets), drawing EACH task's groups only
// from that same task's own entry in byTaskBucket — never another task's
// transcripts, even if both loaded from the same benchmark run. Every model
// benchmarked walks this same slice in this same order, so which (task,
// bucket, record) combination gets requested first is deterministic and
// identical across models. A bucket with no records for a task is omitted
// rather than producing an empty group; a task absent from byTaskBucket
// entirely (its normalized folder had no aai_transcripts.jsonl) contributes
// no groups at all.
func GroupByTaskAndBucket(byTaskBucket map[string]map[string][]TranscriptRecord, tasks []string) []TaskGroup {
	var groups []TaskGroup

	for _, task := range tasks {
		byBucket := byTaskBucket[task]

		for _, bucket := range Buckets(byBucket) {
			records := byBucket[bucket]
			if len(records) == 0 {
				continue
			}

			groups = append(groups, TaskGroup{
				Task:    task,
				Bucket:  bucket,
				Records: records,
			})
		}
	}

	return groups
}

// Buckets returns the bucket names present in byBucket, ordered by duration
// when a name parses as "<seconds>s" — which is every bucket name every
// voice_bench source has produced so far — falling back to lexical order for
// anything that doesn't. Lexical order alone would sort "120s" before "30s"
// before "900s", which is duration-nonsense for a report meant to show
// performance scaling with size.
func Buckets(byBucket map[string][]TranscriptRecord) []string {
	names := make([]string, 0, len(byBucket))
	for name := range byBucket {
		names = append(names, name)
	}

	sort.Slice(names, func(i, j int) bool {
		si, oki := bucketSeconds(names[i])
		sj, okj := bucketSeconds(names[j])

		if oki && okj {
			return si < sj
		}

		return names[i] < names[j]
	})

	return names
}

func bucketSeconds(bucket string) (seconds int, ok bool) {
	if !strings.HasSuffix(bucket, "s") {
		return 0, false
	}

	n, err := strconv.Atoi(strings.TrimSuffix(bucket, "s"))
	if err != nil {
		return 0, false
	}

	return n, true
}
