// Append/update-canonical mode (see RunAppendCanonical below): merges
// one result set -- a directory holding a standard run's report-*.json
// and/or a TTFT run's ttft-report*.json (see report.go/ttft.go) -- into a
// single running "canonical" result set per benchmark schema version, so
// results/canonical/<version> always reflects the latest data for every
// (model, reasoning effort, task, bucket) combination appended to it,
// however many separate runs -- or -buckets partial runs (see
// RunConfig.Buckets in framework.go) -- it took to build up that coverage.
// This mode does no benchmarking of its own: it only reads/merges/writes
// JSON already on disk.
//
// ReviewAndMergeIntoCanonical (below) is the run-and-merge flow's second half:
// it runs AFTER framework_pass.go's standard and TTFT passes have already staged fresh output, and wraps
// RunAppendCanonical with a human review-and-approve gate in between staging
// and merging.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultCanonicalVersion is the benchmark schema version append-canonical targets
// when -canonical-version isn't set.
const DefaultCanonicalVersion = "0.1.3"

// CanonicalVersionDirName converts a dotted version ("0.1.2") into its folder
// name ("0-1-2") -- a dot reads as a file extension to a casual `ls`, a dash
// doesn't.
func CanonicalVersionDirName(version string) string {
	return strings.ReplaceAll(version, ".", "-")
}

// RunAppendCanonical merges every report-*.json and ttft-report*.json found
// directly under sourceDir into canonicalRoot/<version dir>'s own running
// report-<judge>.json / ttft-report.json files, creating the version dir on
// first use. Unlike a normal run's results/ output (a fresh, uniquely-named
// subfolder every call -- see NewRunDir), canonical files live directly
// in the version dir and accumulate in place across however many
// append-canonical calls it takes to cover every model -- see
// mergeJSONSamples/mergeCells for the per-(model, reasoning effort, task,
// bucket) replace semantics that make repeat appends, including ones
// restricted to a subset of buckets, safe.
//
// The merge works entirely on the already-serialized jsonSample/Cell (and
// TTFT equivalent) values straight from disk -- it never reconstitutes a
// Sample/TTFTSample (with a time.Duration latency) from them. An earlier
// version did: it rebuilt Samples from the canonical file's already-rounded
// seconds, recomputed derived fields like TokensPerSecond from that
// rebuilt (and therefore already lossy) Duration, and re-ran
// Aggregate/AggregateTTFT over the result. That recompute-on-every-append
// meant a (model, task) row's numbers could drift a little on EVERY append,
// even for rows the current sourceDir never touched, purely from
// float-through-Duration-and-back noise compounding call after call. Since
// this mode's whole job is merging already-computed results, not producing
// new ones, an untouched row must come out byte-for-byte identical to how
// it went in -- so rows are copied through verbatim, and only a
// (model, task) pair actually present in the new sourceDir ever changes.
func RunAppendCanonical(canonicalRoot, version, sourceDir string) error {
	versionDir := filepath.Join(canonicalRoot, CanonicalVersionDirName(version))

	if err := os.MkdirAll(versionDir, 0o777); err != nil {
		return fmt.Errorf("creating canonical version dir %s: %w", versionDir, err)
	}

	reportsMerged, err := mergeReports(versionDir, sourceDir)
	if err != nil {
		return err
	}

	ttftMerged, err := mergeTTFTReports(versionDir, sourceDir)
	if err != nil {
		return err
	}

	if reportsMerged == 0 && ttftMerged == 0 {
		return fmt.Errorf("no report-*.json or ttft-report*.json files found under %s", sourceDir)
	}

	return nil
}

// mergeReports merges every report-*.json under sourceDir into versionDir,
// one output file per judge model (report-<judge>.json), and returns how
// many source files it found. A source report with no "judge" field is
// rejected rather than guessed at (see summary.go's judgeFor for why a wrong
// guess is worse than an error): it would silently misattribute real data to
// a judge that never produced it.
func mergeReports(versionDir, sourceDir string) (int, error) {
	paths, err := filepath.Glob(filepath.Join(sourceDir, "report-*.json"))
	if err != nil {
		return 0, fmt.Errorf("globbing %s for report-*.json: %w", sourceDir, err)
	}

	for _, path := range paths {
		var incoming jsonReport
		if err := readJSON(path, &incoming); err != nil {
			return 0, err
		}

		if strings.TrimSpace(incoming.Judge) == "" {
			return 0, fmt.Errorf("%s: empty %q field, refusing to guess which canonical file to merge into", path, "judge")
		}

		canonicalPath := filepath.Join(versionDir, fmt.Sprintf("report-%s.json", incoming.Judge))

		var existing jsonReport
		if err := readJSONIfExists(canonicalPath, &existing); err != nil {
			return 0, err
		}

		out := jsonReport{
			Judge:       incoming.Judge,
			GeneratedAt: time.Now().UTC(),
			Samples:     mergeJSONSamples(existing.Samples, incoming.Samples),
			Aggregates:  mergeCells(existing.Aggregates, incoming.Aggregates),
		}

		if err := writeJSON(canonicalPath, out); err != nil {
			return 0, err
		}

		log.Printf("append-canonical: %s -> %s (%d sample(s) total)", path, canonicalPath, len(out.Samples))
	}

	return len(paths), nil
}

// mergeTTFTReports merges every ttft-report*.json under sourceDir into
// versionDir's single running ttft-report.json -- the TTFT path carries no
// judge (see ttft.go's package doc comment), so unlike report-*.json there's
// only ever one canonical file for it. Returns how many source files it found;
// 0 is not an error here, since a source dir from a standard (non-TTFT) run
// legitimately has none.
func mergeTTFTReports(versionDir, sourceDir string) (int, error) {
	paths, err := filepath.Glob(filepath.Join(sourceDir, "ttft-report*.json"))
	if err != nil {
		return 0, fmt.Errorf("globbing %s for ttft-report*.json: %w", sourceDir, err)
	}

	if len(paths) == 0 {
		return 0, nil
	}

	var incomingSamples []ttftJSONSample

	var incomingCells []TTFTCell

	for _, path := range paths {
		var r ttftJSONReport
		if err := readJSON(path, &r); err != nil {
			return 0, err
		}

		incomingSamples = append(incomingSamples, r.Samples...)
		incomingCells = append(incomingCells, r.Aggregates...)
	}

	canonicalPath := filepath.Join(versionDir, "ttft-report.json")

	var existing ttftJSONReport
	if err := readJSONIfExists(canonicalPath, &existing); err != nil {
		return 0, err
	}

	out := ttftJSONReport{
		GeneratedAt: time.Now().UTC(),
		Samples:     mergeTTFTJSONSamples(existing.Samples, incomingSamples),
		Aggregates:  mergeTTFTCells(existing.Aggregates, incomingCells),
	}

	if err := writeJSON(canonicalPath, out); err != nil {
		return 0, err
	}

	log.Printf("append-canonical: %d ttft source file(s) -> %s (%d sample(s) total)", len(paths), canonicalPath, len(out.Samples))

	return len(paths), nil
}

// mergeJSONSamples returns existing with every sample sharing a (model,
// reasoning effort, task, bucket) key present in incoming dropped, followed
// by all of incoming verbatim -- so appending a rerun of one (model,
// reasoning effort, task, bucket) quadruple fully REPLACES its old data,
// however many records the old run sampled for that bucket, rather than
// merging record-by-record (two runs of the same quadruple aren't
// guaranteed to sample the same records, so a record-level merge could
// leave stale records from an old sampling mixed in with a new one), and
// every untouched sample is passed through byte-for-byte -- see this file's
// package doc comment for why that matters. Keying on bucket (rather than
// stopping at (model, reasoning effort, task)) is what makes a
// -buckets rerun (see RunConfig.Buckets in framework.go) safe to
// append: a run scoped to just "3600s" must replace ONLY that bucket's rows,
// never delete "30s"/"120s"/"900s"'s already-merged data for the same
// model/task just because this particular run never touched them. Keying on
// reasoning effort too means re-running the same model at a NEW effort adds
// a row instead of replacing the old effort's data.
func mergeJSONSamples(existing, incoming []jsonSample) []jsonSample {
	type key struct{ model, reasoningEffort, task, bucket string }

	replace := make(map[key]bool, len(incoming))
	for _, s := range incoming {
		replace[key{s.Model, s.ReasoningEffort, s.Task, s.Bucket}] = true
	}

	merged := make([]jsonSample, 0, len(existing)+len(incoming))

	for _, s := range existing {
		if !replace[key{s.Model, s.ReasoningEffort, s.Task, s.Bucket}] {
			merged = append(merged, s)
		}
	}

	return append(merged, incoming...)
}

// mergeCells mirrors mergeJSONSamples for a report's Aggregates -- same
// (model, reasoning effort, task, bucket) replace key, so a Cell for a
// bucket this run didn't cover survives untouched right alongside the
// samples that back it. incoming's own precomputed Cells are trusted as-is
// (they came straight from the source run's own Aggregate() call), never
// rebuilt from the merged sample set -- rebuilding here is exactly the
// recompute this file's doc comment says to avoid.
func mergeCells(existing, incoming []Cell) []Cell {
	type key struct{ model, reasoningEffort, task, bucket string }

	replace := make(map[key]bool, len(incoming))
	for _, c := range incoming {
		replace[key{c.Model, c.ReasoningEffort, c.Task, c.Bucket}] = true
	}

	merged := make([]Cell, 0, len(existing)+len(incoming))

	for _, c := range existing {
		if !replace[key{c.Model, c.ReasoningEffort, c.Task, c.Bucket}] {
			merged = append(merged, c)
		}
	}

	return append(merged, incoming...)
}

// mergeTTFTJSONSamples mirrors mergeJSONSamples for the TTFT path's sample
// type, including the (model, reasoning effort, task, bucket) replace key.
func mergeTTFTJSONSamples(existing, incoming []ttftJSONSample) []ttftJSONSample {
	type key struct{ model, reasoningEffort, task, bucket string }

	replace := make(map[key]bool, len(incoming))
	for _, s := range incoming {
		replace[key{s.Model, s.ReasoningEffort, s.Task, s.Bucket}] = true
	}

	merged := make([]ttftJSONSample, 0, len(existing)+len(incoming))

	for _, s := range existing {
		if !replace[key{s.Model, s.ReasoningEffort, s.Task, s.Bucket}] {
			merged = append(merged, s)
		}
	}

	return append(merged, incoming...)
}

// mergeTTFTCells mirrors mergeCells for the TTFT path's cell type, including
// the (model, reasoning effort, task, bucket) replace key.
func mergeTTFTCells(existing, incoming []TTFTCell) []TTFTCell {
	type key struct{ model, reasoningEffort, task, bucket string }

	replace := make(map[key]bool, len(incoming))
	for _, c := range incoming {
		replace[key{c.Model, c.ReasoningEffort, c.Task, c.Bucket}] = true
	}

	merged := make([]TTFTCell, 0, len(existing)+len(incoming))

	for _, c := range existing {
		if !replace[key{c.Model, c.ReasoningEffort, c.Task, c.Bucket}] {
			merged = append(merged, c)
		}
	}

	return append(merged, incoming...)
}

func readJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}

	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("parsing %s: %w", path, err)
	}

	return nil
}

// readJSONIfExists is readJSON, except a missing file leaves v at its zero
// value instead of erroring -- the common case the first time a given judge
// or the TTFT path is appended to a canonical version.
func readJSONIfExists(path string, v any) error {
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return nil
	}

	return readJSON(path, v)
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("encoding %s: %w", path, err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}

	return nil
}

// ReviewAndMergeIntoCanonical implements the run-and-merge flow's second half
// (see framework_pass.go): reportDir and ttftDir are the directories
// the standard and TTFT passes just staged their output
// into -- either may be "" when that pass produced nothing to write (e.g. an
// empty dataset match, or no records in TTFTBuckets). It prints a
// combined review summary over the staged data (RunSummary, see summary.go)
// for a human to sanity-check, then either merges into canonical immediately
// (autoApprove, for scripted/CI use) or asks for interactive confirmation on
// stdin first -- mirroring `terraform apply`'s plan-then-confirm shape,
// since the whole point of this mode is a human looking at real numbers
// before they become part of the canonical result set. Declining leaves both
// staged directories on disk untouched -- nothing is deleted, so
// either of them can still be merged later via RunAppendCanonical.
func ReviewAndMergeIntoCanonical(canonicalRoot, version, reportDir, ttftDir string, autoApprove bool) error {
	if reportDir == "" && ttftDir == "" {
		return errors.New("nothing staged to review -- both the standard and TTFT passes produced no output")
	}

	versionDir := filepath.Join(canonicalRoot, CanonicalVersionDirName(version))

	fmt.Println("\n=== staged review summary ===")

	if reportDir == "" {
		// RunSummary requires at least one report-*.json (see loadReports);
		// a TTFT-only stage has no judged samples for it to build a
		// quality/latency table from.
		fmt.Printf("no standard benchmark samples staged -- review the raw ttft files directly: %s\n", ttftDir)
	} else if err := RunSummary(reportDir, reportDir, ttftDir, false); err != nil {
		return fmt.Errorf("summarizing staged results: %w", err)
	}

	fmt.Printf("\nstaged report dir:      %s\n", displayPathOrNone(reportDir))
	fmt.Printf("staged ttft report dir: %s\n", displayPathOrNone(ttftDir))
	fmt.Printf("merge target:           canonical version %s at %s\n", version, versionDir)

	if !autoApprove && !confirmYes(fmt.Sprintf("\nmerge the above into canonical version %s? type 'yes' to proceed: ", version)) {
		fmt.Println("not merged -- staged results were left on disk at the dirs above; pass -approve-merge next time to skip this prompt")
		return nil
	}

	if reportDir != "" {
		if err := RunAppendCanonical(canonicalRoot, version, reportDir); err != nil {
			return fmt.Errorf("merging staged report(s): %w", err)
		}
	}

	if ttftDir != "" {
		if err := RunAppendCanonical(canonicalRoot, version, ttftDir); err != nil {
			return fmt.Errorf("merging staged ttft report(s): %w", err)
		}
	}

	if err := refreshCanonicalSummary(versionDir); err != nil {
		return fmt.Errorf("refreshing canonical summary: %w", err)
	}

	log.Printf("merged staged results into canonical version %s at %s", version, versionDir)

	return nil
}

// refreshCanonicalSummary re-runs the summary code (summary.go) over
// versionDir's own just-merged report-<judge>.json files, updating its
// summary.json/summary.csv/use_cases.json IN PLACE (isCanonical) so they
// reflect the merge immediately rather than going stale until someone
// re-runs it by hand (see RunRefreshCanonicalSummary, -refresh-canonical-summary).
// Skipped entirely -- logged, not an error -- if
// versionDir has no report-*.json yet: RunSummary requires at least one (see
// loadReports), and a canonical version that has only ever received TTFT data
// has no judged samples to build a quality/latency table from. versionDir's
// own ttft-report.json, if the merge (or an earlier append) ever produced
// one, is folded in the same way; loadTTFTReports errors on a glob with zero
// matches too, so an all-report, no-TTFT canonical version must not pass
// itself as the TTFT results dir either.
func refreshCanonicalSummary(versionDir string) error {
	hasReports, err := anyFileMatches(versionDir, "report-*.json")
	if err != nil {
		return err
	}

	if !hasReports {
		log.Printf("no report-*.json under %s yet -- skipping canonical summary refresh", versionDir)
		return nil
	}

	hasTTFT, err := anyFileMatches(versionDir, "ttft-report*.json")
	if err != nil {
		return err
	}

	ttftDir := versionDir
	if !hasTTFT {
		ttftDir = ""
	}

	return RunSummary(versionDir, versionDir, ttftDir, true)
}

// RunRefreshCanonicalSummary implements -refresh-canonical-summary: recomputes
// version's summary.json/summary.csv/use_cases.json from whatever it
// already has staged -- no gateway, ASSEMBLY_AI_TOKEN, or benchmark run needed.
// Thin wrapper around refreshCanonicalSummary for standalone use.
func RunRefreshCanonicalSummary(canonicalRoot, version string) error {
	if strings.TrimSpace(version) == "" {
		version = DefaultCanonicalVersion
	}

	return refreshCanonicalSummary(filepath.Join(canonicalRoot, CanonicalVersionDirName(version)))
}

func anyFileMatches(dir, pattern string) (bool, error) {
	matches, err := filepath.Glob(filepath.Join(dir, pattern))
	if err != nil {
		return false, fmt.Errorf("globbing %s for %s: %w", dir, pattern, err)
	}

	return len(matches) > 0, nil
}

func displayPathOrNone(dir string) string {
	if dir == "" {
		return "(none)"
	}

	return dir
}

// confirmYes prints prompt, reads one line from stdin, and reports whether
// it was exactly "yes" (case-insensitive, surrounding whitespace trimmed) --
// mirroring `terraform apply`'s confirmation prompt. Anything else --
// including a bare Enter, "y", or EOF because stdin isn't a terminal (e.g.
// piped input, a CI job that forgot -approve-merge) -- reads as "no" rather
// than being guessed at, so a merge only ever happens on an explicit,
// unambiguous yes.
func confirmYes(prompt string) bool {
	fmt.Print(prompt)

	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil && line == "" {
		return false
	}

	return strings.EqualFold(strings.TrimSpace(line), "yes")
}
