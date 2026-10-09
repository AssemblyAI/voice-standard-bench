package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeFile is a small helper: create path (and its parent dirs) with the
// given contents.
func writeFile(t *testing.T, path, contents string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte(contents), 0o644))
}

func recordLine(id, bucket, text string) string {
	return `{"id":"` + id + `","input":{"text":"` + text + `","context":{"audio_length_bucket":"` + bucket + `"}}}` + "\n"
}

// recordLineWithMissedText builds a text_correction-shaped record: missed_text
// is a TOP-LEVEL key, a sibling of "input" — not nested inside it — matching
// the real aai_transcripts.jsonl layout for that task.
func recordLineWithMissedText(id, bucket, text, missedText string) string {
	return `{"id":"` + id + `","missed_text":"` + missedText + `","input":{"text":"` + text +
		`","context":{"audio_length_bucket":"` + bucket + `"}}}` + "\n"
}

// TestLoadTranscripts exercises the loader against a directory laid out like
// the real voice_bench/normalized tree: one aai_transcripts.jsonl per task
// folder, sitting next to files that must NOT be read (unified.jsonl,
// unified_reference_free.jsonl) — matching the instruction to reference only
// aai_transcripts.jsonl. Table-driven per the fields/args/assert structure
// TestLLMGatewayFramework_processUserPrompt uses, adapted to this package's
// one real dependency: the filesystem.
func TestLoadTranscripts(t *testing.T) {
	type args struct {
		// build populates a temp root and returns the glob to load.
		build func(t *testing.T, root string) string
	}

	tests := []struct {
		name    string
		args    args
		wantErr bool
		assert  func(t *testing.T, byTaskBucket map[string]map[string][]TranscriptRecord, skipped int)
	}{
		{
			name: "loads only aai_transcripts.jsonl, ignoring sibling normalized files, keyed by task",
			args: args{
				build: func(t *testing.T, root string) string {
					writeFile(t, filepath.Join(root, "summarization", "aai_transcripts.jsonl"),
						recordLine("r1", "30s", "hello world")+recordLine("r2", "120s", "a longer call"))
					// Must be ignored: real files that sit alongside
					// aai_transcripts.jsonl in a normalized task folder.
					writeFile(t, filepath.Join(root, "summarization", "unified.jsonl"),
						recordLine("should-not-load", "30s", "this is a decoy"))
					writeFile(t, filepath.Join(root, "summarization", "unified_reference_free.jsonl"),
						recordLine("should-not-load-2", "30s", "another decoy"))
					return filepath.Join(root, "*", "aai_transcripts.jsonl")
				},
			},
			assert: func(t *testing.T, byTaskBucket map[string]map[string][]TranscriptRecord, skipped int) {
				assert.Equal(t, 0, skipped)
				require.Contains(t, byTaskBucket, "summarization")
				byBucket := byTaskBucket["summarization"]
				require.Len(t, byBucket["30s"], 1)
				require.Len(t, byBucket["120s"], 1)
				assert.Equal(t, "r1", byBucket["30s"][0].ID)
				assert.Equal(t, "hello world", byBucket["30s"][0].Text)
			},
		},
		{
			name: "keeps each task's transcripts separate — one task's records never leak into another's",
			args: args{
				build: func(t *testing.T, root string) string {
					writeFile(t, filepath.Join(root, "summarization", "aai_transcripts.jsonl"),
						recordLine("s1", "30s", "summarization transcript"))
					// text_correction has no aai_transcripts.jsonl today —
					// the glob must still work, and must pick this one up
					// too once it exists, with no code change — but as its
					// OWN task's data, never merged into summarization's.
					writeFile(t, filepath.Join(root, "text_correction", "aai_transcripts.jsonl"),
						recordLine("c1", "30s", "correction transcript"))
					return filepath.Join(root, "*", "aai_transcripts.jsonl")
				},
			},
			assert: func(t *testing.T, byTaskBucket map[string]map[string][]TranscriptRecord, skipped int) {
				require.Len(t, byTaskBucket, 2)

				require.Len(t, byTaskBucket["summarization"]["30s"], 1)
				assert.Equal(t, "s1", byTaskBucket["summarization"]["30s"][0].ID)

				require.Len(t, byTaskBucket["text_correction"]["30s"], 1)
				assert.Equal(t, "c1", byTaskBucket["text_correction"]["30s"][0].ID)
			},
		},
		{
			name: "a malformed line is skipped and counted, not fatal",
			args: args{
				build: func(t *testing.T, root string) string {
					path := filepath.Join(root, "summarization", "aai_transcripts.jsonl")
					writeFile(t, path,
						recordLine("good", "30s", "fine")+
							"{not valid json\n"+
							`{"id":"no-text","input":{"context":{"audio_length_bucket":"30s"}}}`+"\n"+
							`{"id":"no-bucket","input":{"text":"hi","context":{}}}`+"\n")
					return path
				},
			},
			assert: func(t *testing.T, byTaskBucket map[string]map[string][]TranscriptRecord, skipped int) {
				assert.Equal(t, 3, skipped, "bad-json, missing-text, and missing-bucket lines all count")
				byBucket := byTaskBucket["summarization"]
				require.Len(t, byBucket["30s"], 1)
				assert.Equal(t, "good", byBucket["30s"][0].ID)
			},
		},
		{
			name: "no file matches the glob is an error, not an empty result",
			args: args{
				build: func(t *testing.T, root string) string {
					return filepath.Join(root, "nothing-here", "*.jsonl")
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			glob := tt.args.build(t, root)

			byTaskBucket, skipped, err := LoadTranscripts(glob)

			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			tt.assert(t, byTaskBucket, skipped)
		})
	}
}

func TestSampleByBucket(t *testing.T) {
	byBucket := map[string][]TranscriptRecord{
		"30s": {
			{ID: "c"}, {ID: "a"}, {ID: "b"},
		},
	}

	t.Run("caps per bucket after sorting by id, for reproducible sampling", func(t *testing.T) {
		sampled := SampleByBucket(byBucket, 2)
		require.Len(t, sampled["30s"], 2)
		assert.Equal(t, "a", sampled["30s"][0].ID)
		assert.Equal(t, "b", sampled["30s"][1].ID)
	})

	t.Run("max <= 0 means no cap and returns the input unchanged", func(t *testing.T) {
		sampled := SampleByBucket(byBucket, 0)
		assert.Len(t, sampled["30s"], 3)
	})

	t.Run("max larger than the bucket is a no-op, not a panic", func(t *testing.T) {
		sampled := SampleByBucket(byBucket, 100)
		assert.Len(t, sampled["30s"], 3)
	})
}

func TestSampleByTaskAndBucket(t *testing.T) {
	byTaskBucket := map[string]map[string][]TranscriptRecord{
		"summarization": {
			"30s": {{ID: "c"}, {ID: "a"}, {ID: "b"}},
		},
		"text_correction": {
			"30s": {{ID: "z"}},
		},
	}

	sampled := SampleByTaskAndBucket(byTaskBucket, 2)

	t.Run("caps each task's bucket independently, same rule as SampleByBucket", func(t *testing.T) {
		require.Len(t, sampled["summarization"]["30s"], 2)
		assert.Equal(t, "a", sampled["summarization"]["30s"][0].ID)
		assert.Equal(t, "b", sampled["summarization"]["30s"][1].ID)
	})

	t.Run("a cap on one task's bucket doesn't touch another task's records", func(t *testing.T) {
		require.Len(t, sampled["text_correction"]["30s"], 1)
		assert.Equal(t, "z", sampled["text_correction"]["30s"][0].ID)
	})
}

func TestBuckets_ordersByDuration(t *testing.T) {
	byBucket := map[string][]TranscriptRecord{
		"900s": nil, "30s": nil, "3600s": nil, "120s": nil,
	}

	got := Buckets(byBucket)

	assert.Equal(t, []string{"30s", "120s", "900s", "3600s"}, got,
		"lexical order would put \"120s\" and \"3600s\" before \"30s\" and \"900s\" — duration order is the point")
}

func TestBuckets_fallsBackToLexicalForUnparseableNames(t *testing.T) {
	byBucket := map[string][]TranscriptRecord{
		"unknown": nil, "30s": nil, "also-unknown": nil,
	}

	got := Buckets(byBucket)

	// "30s" parses and sorts by its 30-second duration; the two non-numeric
	// names fall back to lexical order relative to each other. The exact
	// interleaving isn't the contract — not panicking, and "30s" being
	// treated as a real duration, are.
	assert.Contains(t, got, "30s")
	assert.Len(t, got, 3)
}

func TestGroupByTaskAndBucket(t *testing.T) {
	byTaskBucket := map[string]map[string][]TranscriptRecord{
		"summarization": {
			"900s": {{ID: "c"}},
			"30s":  {{ID: "a"}, {ID: "b"}},
			"120s": {{ID: "d"}},
		},
		"text_correction": {
			"30s": {{ID: "z"}},
		},
	}

	t.Run("one group per (task, bucket) pair, tasks outer and buckets in duration order", func(t *testing.T) {
		groups := GroupByTaskAndBucket(byTaskBucket, []string{"summarization", "text_correction"})

		require.Len(t, groups, 4)

		var got [][2]string
		for _, g := range groups {
			got = append(got, [2]string{g.Task, g.Bucket})
		}

		assert.Equal(t, [][2]string{
			{"summarization", "30s"}, {"summarization", "120s"}, {"summarization", "900s"},
			{"text_correction", "30s"},
		}, got)
	})

	t.Run("a group's records are exactly that TASK's own records for that bucket, untouched", func(t *testing.T) {
		groups := GroupByTaskAndBucket(byTaskBucket, []string{"summarization"})

		require.Len(t, groups, 3)
		assert.Equal(t, byTaskBucket["summarization"]["30s"], groups[0].Records)
		assert.Equal(t, byTaskBucket["summarization"]["120s"], groups[1].Records)
		assert.Equal(t, byTaskBucket["summarization"]["900s"], groups[2].Records)
	})

	t.Run("one task's records never appear in another task's groups, even for a shared bucket name", func(t *testing.T) {
		groups := GroupByTaskAndBucket(byTaskBucket, []string{"summarization", "text_correction"})

		for _, g := range groups {
			if g.Bucket != "30s" {
				continue
			}

			ids := make([]string, len(g.Records))
			for i, r := range g.Records {
				ids[i] = r.ID
			}

			if g.Task == "summarization" {
				assert.ElementsMatch(t, []string{"a", "b"}, ids)
			} else {
				assert.ElementsMatch(t, []string{"z"}, ids)
			}
		}
	})

	t.Run("a bucket with no records is omitted rather than producing an empty group", func(t *testing.T) {
		groups := GroupByTaskAndBucket(map[string]map[string][]TranscriptRecord{
			"summarization": {
				"30s":  {{ID: "a"}},
				"120s": nil,
			},
		}, []string{"summarization"})

		require.Len(t, groups, 1)
		assert.Equal(t, "30s", groups[0].Bucket)
	})

	t.Run("a task absent from byTaskBucket entirely contributes no groups", func(t *testing.T) {
		groups := GroupByTaskAndBucket(byTaskBucket, []string{"information_extraction"})
		assert.Empty(t, groups)
	})

	t.Run("no tasks means no groups", func(t *testing.T) {
		assert.Empty(t, GroupByTaskAndBucket(byTaskBucket, nil))
	})
}

// TestLoadTranscripts_missedText locks down decoding of missed_text — the
// simulated corrupted transcript that gives text_correction a real reference
// (TranscriptRecord.Text is the known-true original). It is a TOP-LEVEL key
// in the real data, a sibling of "input", not nested inside it — easy to get
// wrong, since every other field this loader reads lives under "input".
func TestLoadTranscripts_missedText(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "text_correction", "aai_transcripts.jsonl")
	writeFile(t, path,
		recordLineWithMissedText("with-errors", "30s", "the true original text", "the tru original text")+
			recordLine("no-missed-text-field", "30s", "a record from before the field existed"))

	byTaskBucket, skipped, err := LoadTranscripts(filepath.Join(root, "*", "aai_transcripts.jsonl"))
	require.NoError(t, err)
	assert.Equal(t, 0, skipped, "a record with no missed_text field must load, not be skipped")

	records := byTaskBucket["text_correction"]["30s"]
	require.Len(t, records, 2)

	byID := map[string]TranscriptRecord{}
	for _, r := range records {
		byID[r.ID] = r
	}

	assert.Equal(t, "the tru original text", byID["with-errors"].MissedText)
	assert.Equal(t, "the true original text", byID["with-errors"].Text,
		"Text must stay the known-true original even though a corrupted counterpart exists")

	assert.Empty(t, byID["no-missed-text-field"].MissedText,
		"missing missed_text must decode to empty, not error or skip the record — it's a normal shape for non-text_correction data")
}
