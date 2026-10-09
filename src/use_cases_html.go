// HTML rendering of use_cases.json (see use_cases.go) into a standalone,
// dependency-free report a human can open straight in a browser: one card
// per use case, grouped by audio-length bucket, with its ranked models'
// quality/cost/latency rendered as readable bars and badges instead of raw
// JSON. writeUseCasesHTML is called from RunSummary (summary.go) right after
// writeUseCases writes use_cases.json for the canonical version dir, so the two
// files never drift apart. It reads use_cases.json back off disk rather than
// taking a useCasesReport parameter directly, so the same regeneration path
// also works standalone -- e.g. after hand-editing use_cases.go and
// re-running -refresh-canonical-summary, or against a use_cases.json that was
// written by an earlier binary.
package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

//go:embed use_cases.html.tmpl
var useCasesHTMLTemplateSrc string

var useCasesHTMLTemplate = template.Must(template.New("use_cases.html.tmpl").Parse(useCasesHTMLTemplateSrc))

// useCasesPageView is the root data useCasesHTMLTemplate renders. Every
// display-formatting decision (number formatting, color tiers, human bucket
// labels) is made here in Go rather than in the template itself -- the
// template only ranges and branches on already-formatted fields.
type useCasesPageView struct {
	GeneratedAt   string
	Buckets       []bucketView
	TotalUseCases int
	TotalModels   int
	Tags          []tagCountView
}

// tagCountView backs the toolbar's tag filter chips, most-used tag first.
type tagCountView struct {
	Tag   string
	Count int
}

type bucketView struct {
	Bucket string
	// Label is a human duration ("30 seconds", "1 hour") derived from
	// bucketSeconds -- see bucketDurationLabel.
	Label string
	// Kind is "real-time-adjacent"/"batch / async"/"" -- see
	// bucketKindLabel, which reuses deriveUseCases' own thresholds so this
	// label can never disagree with which use cases the bucket actually got.
	Kind     string
	UseCases []useCaseCardView
}

type useCaseCardView struct {
	Bucket      string
	Title       string
	Description string
	Tags        []string
	// TagsAttr is Tags space-joined for the card's data-tags attribute, the
	// hook the page's tag-filter JS matches against.
	TagsAttr string
	Models   []modelRowView
}

type modelRowView struct {
	Rank  int
	Model string
	// ReasoningEffortLabel mirrors useCaseModel.ReasoningEffort -- "default"
	// when the model was benchmarked without -reasoning-effort or didn't
	// support it, never blank -- rendered next to LatencyLabel in the
	// template.
	ReasoningEffortLabel string
	QualityScore         int
	QualityClass         string
	HasTaskQuality       bool
	TaskQualityScore     int
	TaskQualityClass     string
	CostLabel            string
	LatencyLabel         string
}

// buildUseCasesPageView converts a parsed use_cases.json into template-ready
// data: use cases regrouped by bucket (ascending duration, mirroring
// deriveUseCases' own bucket order), plus the header/toolbar's summary
// counts and tag chips.
func buildUseCasesPageView(report useCasesReport) useCasesPageView {
	var bucketOrder []string

	byBucket := map[string][]useCase{}

	for _, uc := range report.UseCases {
		if _, seen := byBucket[uc.Bucket]; !seen {
			bucketOrder = append(bucketOrder, uc.Bucket)
		}

		byBucket[uc.Bucket] = append(byBucket[uc.Bucket], uc)
	}

	sort.Slice(bucketOrder, func(i, j int) bool {
		si, oki := bucketSeconds(bucketOrder[i])
		sj, okj := bucketSeconds(bucketOrder[j])

		if oki && okj {
			return si < sj
		}

		return bucketOrder[i] < bucketOrder[j]
	})

	models := map[string]bool{}
	tagCounts := map[string]int{}
	buckets := make([]bucketView, 0, len(bucketOrder))

	for _, bucket := range bucketOrder {
		cards := make([]useCaseCardView, 0, len(byBucket[bucket]))

		for _, uc := range byBucket[bucket] {
			for _, tag := range uc.Tags {
				tagCounts[tag]++
			}

			cards = append(cards, useCaseCardView{
				Bucket:      uc.Bucket,
				Title:       uc.Title,
				Description: uc.Description,
				Tags:        uc.Tags,
				TagsAttr:    strings.Join(uc.Tags, " "),
				Models:      modelRowViews(uc.Models, models),
			})
		}

		buckets = append(buckets, bucketView{
			Bucket:   bucket,
			Label:    bucketDurationLabel(bucket),
			Kind:     bucketKindLabel(bucket),
			UseCases: cards,
		})
	}

	return useCasesPageView{
		GeneratedAt:   report.GeneratedAt.Format("Jan 2, 2006 15:04 MST"),
		Buckets:       buckets,
		TotalUseCases: len(report.UseCases),
		TotalModels:   len(models),
		Tags:          sortedTagCounts(tagCounts),
	}
}

// modelRowViews formats one use case's ranked models, recording every model
// name it sees into seen -- buildUseCasesPageView's running set behind the
// header's total-unique-models stat.
func modelRowViews(models []useCaseModel, seen map[string]bool) []modelRowView {
	out := make([]modelRowView, len(models))

	for i, m := range models {
		seen[m.Model] = true

		out[i] = modelRowView{
			Rank:                 m.Rank,
			Model:                m.Model,
			ReasoningEffortLabel: m.ReasoningEffort,
			QualityScore:         m.QualityScore,
			QualityClass:         qualityClass(m.QualityScore),
			HasTaskQuality:       m.TaskQualityScore > 0,
			TaskQualityScore:     m.TaskQualityScore,
			TaskQualityClass:     qualityClass(m.TaskQualityScore),
			CostLabel:            formatCostPerQuality(m.CostPerQuality),
			LatencyLabel:         formatLatencyP50(m.LatencyP50S),
		}
	}

	return out
}

// sortedTagCounts orders tags by frequency (most-used use cases first, tag
// name breaking ties) so the toolbar's tag chips surface the most useful
// filters first rather than in arbitrary map order.
func sortedTagCounts(counts map[string]int) []tagCountView {
	out := make([]tagCountView, 0, len(counts))
	for tag, n := range counts {
		out = append(out, tagCountView{Tag: tag, Count: n})
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}

		return out[i].Tag < out[j].Tag
	})

	return out
}

// qualityClass buckets a 1-100 quality score into the same tiers use_cases.go
// gates eligibility on (qualityFloorPremium/qualityFloorAcceptable), so a
// card's color-coding can never disagree with why that model qualified for
// the use case it's listed under. Every model that reaches this page has
// already cleared at least qualityFloorAcceptable on some score (see
// overallQualityAcceptable and each archetype's own gate in use_cases.go),
// so "q-low" is defensive rather than expected to ever render.
func qualityClass(score int) string {
	switch {
	case float64(score) >= qualityFloorPremium:
		return "q-high"
	case float64(score) >= qualityFloorAcceptable:
		return "q-mid"
	default:
		return "q-low"
	}
}

// formatCostPerQuality renders a cost_per_quality value the same
// shortest-round-trip way encoding/json itself would, suffixed with "¢" --
// e.g. 0.1162 -> "0.1162¢", 5 -> "5¢". Zero (no cost data) renders
// as "n/a" rather than "0¢", matching summary.go's costPerQualityString
// convention for the same field.
func formatCostPerQuality(v float64) string {
	if v == 0 {
		return "n/a"
	}

	return strconv.FormatFloat(v, 'f', -1, 64) + "¢"
}

// formatLatencyP50 renders sub-second p50s in milliseconds -- "561 ms" reads
// faster than "0.56 s" for exactly the low-latency use cases where the
// difference is the point -- and everything else in seconds.
func formatLatencyP50(seconds float64) string {
	if seconds <= 0 {
		return "n/a"
	}

	if seconds < 1 {
		return fmt.Sprintf("%d ms", int(math.Round(seconds*1000)))
	}

	return fmt.Sprintf("%.2f s", seconds)
}

// bucketDurationLabel renders a bucket's seconds as a short human duration --
// "30s" -> "30 seconds", "3600s" -> "1 hour" -- derived from bucketSeconds
// rather than a hardcoded bucket-to-label table, so a new bucket the
// normalization side adds later renders correctly with no change here (the
// same "discovered, not hardcoded" convention tasksIn/taskColumns/Buckets
// already use elsewhere in this package). Falls back to the raw bucket name
// when it doesn't parse as "<seconds>s".
func bucketDurationLabel(bucket string) string {
	seconds, ok := bucketSeconds(bucket)
	if !ok {
		return bucket
	}

	switch {
	case seconds < 60:
		return pluralize(seconds, "second")
	case seconds < 3600:
		return pluralize(seconds/60, "minute")
	default:
		return pluralize(seconds/3600, "hour")
	}
}

func pluralize(n int, unit string) string {
	if n == 1 {
		return "1 " + unit
	}

	return strconv.Itoa(n) + " " + unit + "s"
}

// bucketKindLabel tags a bucket as real-time-adjacent or batch/async using
// deriveUseCases' own thresholds (realtimeMaxBucketS/batchMinBucketS), so
// this label can never disagree with which latency- or throughput-flavored
// use cases that bucket actually received.
func bucketKindLabel(bucket string) string {
	seconds, ok := bucketSeconds(bucket)
	if !ok {
		return ""
	}

	switch {
	case seconds <= realtimeMaxBucketS:
		return "real-time-adjacent"
	case seconds >= batchMinBucketS:
		return "batch / async"
	default:
		return ""
	}
}

// renderUseCasesHTML executes useCasesHTMLTemplate over report and writes the
// result to w.
func renderUseCasesHTML(w io.Writer, report useCasesReport) error {
	return useCasesHTMLTemplate.Execute(w, buildUseCasesPageView(report))
}

// writeUseCasesHTML reads dir/use_cases.json and renders it to
// dir/use_cases.html: a standalone report a human can open directly in a
// browser, no server or build step required. See this file's package doc
// comment for why it reads the JSON back off disk instead of taking a
// useCasesReport parameter.
func writeUseCasesHTML(dir string) error {
	jsonPath := filepath.Join(dir, "use_cases.json")

	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return fmt.Errorf("reading %s: %w", jsonPath, err)
	}

	var report useCasesReport
	if err := json.Unmarshal(data, &report); err != nil {
		return fmt.Errorf("parsing %s: %w", jsonPath, err)
	}

	var buf bytes.Buffer
	if err := renderUseCasesHTML(&buf, report); err != nil {
		return fmt.Errorf("rendering use_cases.html: %w", err)
	}

	htmlPath := filepath.Join(dir, "use_cases.html")
	if err := os.WriteFile(htmlPath, buf.Bytes(), 0o644); err != nil {
		return err
	}

	fmt.Println("wrote", htmlPath)

	return nil
}
