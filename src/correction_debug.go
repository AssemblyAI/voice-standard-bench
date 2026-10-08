package main

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// -debug review of one text_correction sample.
//
// text_correction's truth-based quality score is a judge's verdict on how
// similar the output is to the truth in meaning, and when a verdict like that
// looks wrong the only way to know why is to read the texts it was given. This
// file stacks them -- the TRUTH, the INPUT the model was shown (the corrupted
// missed_text), and the model's OUTPUT -- one word per column, so the same
// column reads top to bottom as "what it should say / what the model was
// given / what the model said":
//
//	TRUTH   we  agreed  to  ship    the  budget    review  on   ·     friday
//	INPUT   we  agreed  to  [shop]  the  [budjet]  review  on  [um]   friday
//	OUTPUT  we  agreed  to  ship    the  [budjet]  review  on   ·     friday
//
// A cell in [brackets] differs from the truth WORD FOR WORD -- the INPUT row's
// brackets are the corruption the model was handed, the OUTPUT row's are what
// it left or introduced -- and · is "no word here" ([·] is a truth word that
// row is missing). That is deliberately stricter than the score: a filler, a
// stutter, or "5" against "five" is bracketed here but is exactly what the
// meaning judge is told to look past, so a bracket is something to LOOK at, not
// a mark against the model. Each row is aligned to the truth by the same
// minimum-edit alignment behind truth_wer and input_wer (alignWords over
// exactly the Keys they compare), so the bracket count in a row IS that row's
// edit count. Words are shown as written, case and punctuation intact; only a
// difference in the underlying word gets brackets, so a restyled "Friday,"
// against "friday" reads as different text but not as a difference. The header
// above the rows prints every score -- the judge's meaning_similarity among
// them -- and the judge's own one-sentence note on the most significant meaning
// difference it found, so the verdict can be set directly against the words.

const (
	// debugViewWidth is the line width the aligned rows wrap at.
	debugViewWidth = 120

	// debugGap is the cell text for "no word here".
	debugGap = "·"

	// debugCollapseMinRun is how many consecutive columns identical in all
	// three rows (down to case and punctuation) get folded into one
	// "… N identical words …" marker, keeping debugCollapseContext columns
	// either side so the neighbourhood of every difference stays readable. A
	// 3,600-second transcript is ~10,000 words that are 99.7% the same; printed
	// whole, its handful of differences would be buried. Shorter runs -- the
	// whole of a 30-second record -- print in full.
	debugCollapseMinRun  = 40
	debugCollapseContext = 8

	// debugLabelWidth is the width of the "TRUTH " / "INPUT " / "OUTPUT " row
	// labels, so every row's first column starts in the same place.
	debugLabelWidth = 7

	// debugColumnGap separates adjacent columns.
	debugColumnGap = "  "

	// debugPreviewWords is how many leading words of the output are shown when
	// a pair of texts is too large to align at all.
	debugPreviewWords = 40
)

// viewCell is one word of one row of the view.
type viewCell struct {
	text string
	// wrong marks a cell that differs from the truth: it renders in [brackets]
	// and is one edit in the score.
	wrong bool
}

func (c viewCell) display() string {
	if c.wrong {
		return "[" + c.text + "]"
	}

	return c.text
}

// viewColumn is one aligned column: the truth's word and what each of the two
// other rows has opposite it.
type viewColumn struct {
	truth, input, output viewCell
	// firstWord/lastWord are the 1-based truth word numbers this column
	// covers: the same word for an ordinary column, a range for a collapsed
	// run, and 0 for a column that holds only inserted words (no truth word).
	firstWord, lastWord int
	// same is set when all three rows show identical text, making the column
	// a candidate for folding into a collapsed run.
	same bool
}

// correctionView is the three texts aligned into columns, plus the edit count
// each of the two non-truth rows carries.
type correctionView struct {
	columns     []viewColumn
	truthWords  int
	inputEdits  int // [bracketed] cells in the INPUT row
	outputEdits int // [bracketed] cells in the OUTPUT row
}

// buildCorrectionView aligns input and output each to truth and merges the two
// alignments into shared columns. ok is false only when a pair of texts is too
// large to align (see alignMaxCells).
func buildCorrectionView(truth, input, output string) (view correctionView, ok bool) {
	truthWords, inputWords, outputWords := correctionWords(truth), correctionWords(input), correctionWords(output)

	inputOps, _, inputOK := alignWords(wordKeys(truthWords), wordKeys(inputWords))
	outputOps, _, outputOK := alignWords(wordKeys(truthWords), wordKeys(outputWords))

	if !inputOK || !outputOK {
		return correctionView{truthWords: len(truthWords)}, false
	}

	inputRow := alignRowToTruth(len(truthWords), inputOps)
	outputRow := alignRowToTruth(len(truthWords), outputOps)

	view.truthWords = len(truthWords)

	for i := 0; i <= len(truthWords); i++ {
		// Words either row inserted between truth word i-1 and i get columns
		// of their own, with a gap in the truth row.
		view.columns = append(view.columns, insertionColumns(inputWords, outputWords, inputRow.gaps[i], outputRow.gaps[i])...)

		if i == len(truthWords) {
			break
		}

		key := truthWords[i].Key

		view.columns = append(view.columns, viewColumn{
			truth:     viewCell{text: truthWords[i].Surface},
			input:     alignedCell(inputWords, inputRow.at[i], key),
			output:    alignedCell(outputWords, outputRow.at[i], key),
			firstWord: i + 1,
			lastWord:  i + 1,
		})
	}

	for i := range view.columns {
		col := &view.columns[i]

		if col.input.wrong {
			view.inputEdits++
		}

		if col.output.wrong {
			view.outputEdits++
		}

		col.same = !col.input.wrong && !col.output.wrong &&
			col.truth.text == col.input.text && col.truth.text == col.output.text
	}

	return view, true
}

func wordKeys(words []correctionWord) []string {
	keys := make([]string, len(words))
	for i, w := range words {
		keys[i] = w.Key
	}

	return keys
}

// rowAlignment is one row's alignment to the truth, reshaped from edit-script
// order into per-truth-word lookups.
type rowAlignment struct {
	// at[i] is the index of the row word aligned to truth word i (equal or
	// substituted), or -1 when the row has no word there.
	at []int
	// gaps[i] holds the row words inserted between truth word i-1 and truth
	// word i; gaps[len(truth)] trails the last truth word.
	gaps [][]int
}

func alignRowToTruth(truthWords int, ops []tokenAlign) rowAlignment {
	row := rowAlignment{at: make([]int, truthWords), gaps: make([][]int, truthWords+1)}

	var pending []int

	for _, op := range ops {
		if op.op == opIns {
			pending = append(pending, op.hypIdx)
			continue
		}

		// Match, substitution, and deletion each consume one truth word.
		row.gaps[op.refIdx] = pending
		pending = nil

		if op.op == opDel {
			row.at[op.refIdx] = -1
		} else {
			row.at[op.refIdx] = op.hypIdx
		}
	}

	row.gaps[truthWords] = pending

	return row
}

// alignedCell is a row's cell opposite one truth word with the given Key.
func alignedCell(words []correctionWord, idx int, truthKey string) viewCell {
	if idx < 0 {
		return viewCell{text: debugGap, wrong: true}
	}

	return viewCell{text: words[idx].Surface, wrong: words[idx].Key != truthKey}
}

// insertionColumns lays out the words the INPUT row (inputIdx) and the OUTPUT
// row (outputIdx) each inserted at one spot between truth words. The two
// lists are aligned to each other so a filler the model left in place sits in
// the same column as the input's, rather than the lists being zipped by
// position.
func insertionColumns(inputWords, outputWords []correctionWord, inputIdx, outputIdx []int) []viewColumn {
	if len(inputIdx) == 0 && len(outputIdx) == 0 {
		return nil
	}

	gap := viewCell{text: debugGap}

	column := func(in, out int) viewColumn {
		col := viewColumn{truth: gap, input: gap, output: gap}

		if in >= 0 {
			col.input = viewCell{text: inputWords[in].Surface, wrong: true}
		}

		if out >= 0 {
			col.output = viewCell{text: outputWords[out].Surface, wrong: true}
		}

		return col
	}

	inputKeys, outputKeys := make([]string, len(inputIdx)), make([]string, len(outputIdx))

	for i, idx := range inputIdx {
		inputKeys[i] = inputWords[idx].Key
	}

	for i, idx := range outputIdx {
		outputKeys[i] = outputWords[idx].Key
	}

	ops, _, ok := alignWords(inputKeys, outputKeys)
	if !ok { // can't happen at these sizes; fall back to one column per word
		var cols []viewColumn
		for _, in := range inputIdx {
			cols = append(cols, column(in, -1))
		}

		for _, out := range outputIdx {
			cols = append(cols, column(-1, out))
		}

		return cols
	}

	cols := make([]viewColumn, 0, len(ops))

	for _, op := range ops {
		in, out := -1, -1

		if op.refIdx >= 0 {
			in = inputIdx[op.refIdx]
		}

		if op.hypIdx >= 0 {
			out = outputIdx[op.hypIdx]
		}

		cols = append(cols, column(in, out))
	}

	return cols
}

// collapseIdentical folds every run of at least minRun consecutive same
// columns into one marker column, leaving context columns of the run on each
// side. The marker records the truth words it stands in for, so the word
// numbers in the block headers stay true.
func collapseIdentical(columns []viewColumn, minRun, context int) []viewColumn {
	out := make([]viewColumn, 0, len(columns))

	for i := 0; i < len(columns); {
		if !columns[i].same {
			out = append(out, columns[i])
			i++

			continue
		}

		end := i
		for end < len(columns) && columns[end].same {
			end++
		}

		run := columns[i:end]

		if len(run) < minRun || len(run) <= 2*context+1 {
			out = append(out, run...)
		} else {
			hidden := run[context : len(run)-context]
			first, last := hidden[0].firstWord, hidden[len(hidden)-1].lastWord

			out = append(out, run[:context]...)
			out = append(out, viewColumn{
				truth:     viewCell{text: fmt.Sprintf("… %d identical words …", last-first+1)},
				input:     viewCell{text: "…"},
				output:    viewCell{text: "…"},
				firstWord: first,
				lastWord:  last,
			})
			out = append(out, run[len(run)-context:]...)
		}

		i = end
	}

	return out
}

// formatCorrectionView renders view as wrapped, column-aligned TRUTH / INPUT /
// OUTPUT blocks no wider than width, long identical runs folded.
func formatCorrectionView(view correctionView, width int) string {
	columns := collapseIdentical(view.columns, debugCollapseMinRun, debugCollapseContext)

	var b strings.Builder

	for start := 0; start < len(columns); {
		// Greedily fill one block: as many columns as fit, at least one.
		end := start
		used := debugLabelWidth

		var widths []int

		for end < len(columns) {
			w := columnWidth(columns[end])

			need := w
			if end > start {
				need += len(debugColumnGap)
			}

			if end > start && used+need > width {
				break
			}

			widths = append(widths, w)
			used += need
			end++
		}

		b.WriteString(blockHeader(columns[start:end], columns[:start]))

		for _, row := range []struct {
			label string
			cell  func(viewColumn) viewCell
		}{
			{"TRUTH  ", func(c viewColumn) viewCell { return c.truth }},
			{"INPUT  ", func(c viewColumn) viewCell { return c.input }},
			{"OUTPUT ", func(c viewColumn) viewCell { return c.output }},
		} {
			line := row.label

			for i, col := range columns[start:end] {
				if i > 0 {
					line += debugColumnGap
				}

				line += padRight(row.cell(col).display(), widths[i])
			}

			b.WriteString(strings.TrimRight(line, " "))
			b.WriteByte('\n')
		}

		b.WriteByte('\n')

		start = end
	}

	return b.String()
}

func columnWidth(c viewColumn) int {
	w := 0

	for _, cell := range []viewCell{c.truth, c.input, c.output} {
		if n := utf8.RuneCountInString(cell.display()); n > w {
			w = n
		}
	}

	return w
}

func padRight(s string, width int) string {
	if n := utf8.RuneCountInString(s); n < width {
		return s + strings.Repeat(" ", width-n)
	}

	return s
}

// blockHeader labels a block with the truth words it covers, so a difference
// found on screen can be located in the transcript. before is every column
// ahead of the block, used only when the block holds nothing but inserted words.
func blockHeader(block, before []viewColumn) string {
	first, last := 0, 0

	for _, col := range block {
		if col.firstWord == 0 {
			continue
		}

		if first == 0 {
			first = col.firstWord
		}

		last = col.lastWord
	}

	if first == 0 {
		after := 0

		for _, col := range before {
			if col.lastWord > after {
				after = col.lastWord
			}
		}

		return fmt.Sprintf("(inserted words after truth word %d)\n", after)
	}

	if first == last {
		return fmt.Sprintf("(truth word %d)\n", first)
	}

	return fmt.Sprintf("(truth words %d-%d)\n", first, last)
}

// formatDebugSample is everything -debug prints for one judged text_correction
// sample: which sample it is, the scores it got, the meaning judge's note on
// them (meaningNotes, when it sent one), and the aligned view of the three texts
// those scores were given.
func formatDebugSample(model ModelInfo, record TranscriptRecord, input, output string, correction CorrectionResult, scores map[string]float64, meaningNotes string) string {
	var b strings.Builder

	b.WriteString(strings.Repeat("=", debugViewWidth))
	b.WriteByte('\n')
	fmt.Fprintf(&b, "text_correction debug  model=%s  reasoning_effort=%s  bucket=%s  record=%s\n",
		model.ID, model.ReasoningEffortLabel(), record.Bucket, record.ID)
	fmt.Fprintf(&b, "scores:  %s\n", formatScores(scores))

	if note := strings.Join(strings.Fields(meaningNotes), " "); note != "" {
		fmt.Fprintf(&b, "judge:   %s\n", note)
	}

	fmt.Fprintf(&b, "TRUTH    %d words -- the original text, the answer key (never shown to the model)\n", correction.TruthWords)
	fmt.Fprintf(&b, "INPUT    %s from the truth, word for word (%.2f%% WER) -- the corrupted text the model was shown\n", pluralize(correction.InputEdits, "edit"), correction.InputWER())
	fmt.Fprintf(&b, "OUTPUT   %s from the truth, word for word (%.2f%% WER) -- the model's corrected text\n", pluralize(correction.OutputEdits, "edit"), correction.TruthWER())
	b.WriteString("legend   [word] differs from the truth word for word (the judge scores meaning, so fillers, stutters and number formatting show here without costing the model)\n")
	b.WriteString("         " + debugGap + " no word here   [" + debugGap + "] a truth word that row is missing\n\n")

	view, ok := buildCorrectionView(record.Text, input, output)
	if !ok {
		fmt.Fprintf(&b, "alignment skipped: the texts are too large or too different to align in memory.\nfirst %d words of the output:\n%s\n\n",
			debugPreviewWords, previewWords(output, debugPreviewWords))

		return b.String()
	}

	if view.inputEdits != correction.InputEdits || view.outputEdits != correction.OutputEdits {
		fmt.Fprintf(&b, "WARNING: the view below counts %d/%d edits (input/output) but the score counted %d/%d -- the alignment and the score disagree\n\n",
			view.inputEdits, view.outputEdits, correction.InputEdits, correction.OutputEdits)
	}

	b.WriteString(formatCorrectionView(view, debugViewWidth))

	return b.String()
}

// formatScores renders a score map as space-separated name=value pairs in
// name order, four significant digits.
func formatScores(scores map[string]float64) string {
	names := make([]string, 0, len(scores))
	for name := range scores {
		names = append(names, name)
	}

	sort.Strings(names)

	parts := make([]string, len(names))
	for i, name := range names {
		parts[i] = fmt.Sprintf("%s=%.4g", name, scores[name])
	}

	return strings.Join(parts, "  ")
}

func previewWords(text string, n int) string {
	fields := strings.Fields(text)
	if len(fields) > n {
		fields = append(fields[:n:n], "…")
	}

	return strings.Join(fields, " ")
}
