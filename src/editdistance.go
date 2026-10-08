package main

// Word-level Levenshtein distance, sized for the longest transcripts this
// benchmark scores (the 3600s bucket runs to ~15,000 words). A full
// (len(a)+1) x (len(b)+1) matrix of that is ~225M cells -- gigabytes as the
// [][]int alignTokens (dictation.go) builds, which also needs the whole
// matrix to backtrace -- and every correction sample needs two of these
// distances, so editDistance computes only the DISTANCE (no alignment), in
// O(len) memory, and in time proportional to the number of edits rather than
// to len(a)*len(b): a near-identical pair of transcripts (a real correction
// differs from its truth in under a few percent of its words) finishes in a
// few passes over a narrow diagonal band.
//
// alignWords runs the very same banded DP but also keeps its moves, so the
// -debug review (correction_debug.go) can SHOW the alignment the score was
// computed from. It stores one 2-bit move per in-band cell rather than a
// matrix of ints, which is what lets a 10,000-word transcript with a couple of
// thousand edits fit in tens of megabytes.

// editDistanceInitialBand is the first diagonal half-width editDistance tries
// before widening. A correction's distance from the truth is usually tens of
// words, so most pairs resolve on the first or second pass; a garbage
// candidate (truncated, or a runaway generation many times the truth's
// length) just walks the doubling up to the full matrix.
const editDistanceInitialBand = 16

// editDistanceInf stands in for "outside the band" in the DP rows. It is
// well below the int32 maximum so adding 1 to it can never overflow.
const editDistanceInf = int32(1 << 29)

// alignMaxCells bounds the trace table alignWords will build, in DP cells (a
// quarter of that many bytes). It sits well above any real transcript pair --
// 10,400 words against a candidate 19% wrong needs ~42M -- and exists only so
// a pathological pair can't allocate without limit; past it alignWords reports
// ok=false and the caller falls back to something that doesn't need an
// alignment. A var, not a const, so tests can exercise that fallback.
var alignMaxCells = 1 << 28

// The moves a trace table records. Match and substitution are the same move
// (diagonal) -- which of the two it was is just whether the tokens are equal.
const (
	moveDiag uint8 = iota
	moveDel        // a[i-1] has no counterpart in b: came from (i-1, j)
	moveIns        // b[j-1] has no counterpart in a: came from (i, j-1)
)

// traceTable is a rows x width grid of 2-bit moves, packed four to a byte.
// Cells start as moveDiag (0) and set only ever ORs a move in, so each cell
// may be written once.
type traceTable struct {
	bits  []byte
	width int
}

func newTraceTable(rows, width int) *traceTable {
	return &traceTable{bits: make([]byte, (rows*width+3)/4), width: width}
}

func (t *traceTable) set(row, col int, move uint8) {
	idx := row*t.width + col
	t.bits[idx>>2] |= move << (uint(idx&3) * 2)
}

func (t *traceTable) get(row, col int) uint8 {
	idx := row*t.width + col

	return (t.bits[idx>>2] >> (uint(idx&3) * 2)) & 3
}

// editDistance returns the Levenshtein distance between a and b: the fewest
// single-token insertions, deletions, and substitutions that turn one into
// the other. Tokens are compared for exact string equality -- any
// normalization (case, punctuation) is the caller's job; see
// correctionTokens.
func editDistance(a, b []string) int {
	ids := make(map[string]int32, len(a)+len(b))

	return editDistanceIDs(internTokens(ids, a), internTokens(ids, b))
}

// internTokens maps every token to a small integer id, shared across calls
// through ids, so the hot DP loops compare int32s instead of strings.
func internTokens(ids map[string]int32, tokens []string) []int32 {
	out := make([]int32, len(tokens))

	for i, token := range tokens {
		id, ok := ids[token]
		if !ok {
			id = int32(len(ids))
			ids[token] = id
		}

		out[i] = id
	}

	return out
}

func editDistanceIDs(a, b []int32) int {
	a, b = trimCommonEnds(a, b)

	// Keep a the shorter side: it drives the rows, b the columns.
	if len(a) > len(b) {
		a, b = b, a
	}

	n, m := len(a), len(b)
	if n == 0 {
		return m
	}

	band := initialBand(n, m)

	for {
		if band > m {
			band = m // wider than the matrix buys nothing; and m always suffices
		}

		if distance, ok := bandedEditDistance(a, b, band); ok {
			return distance
		}

		band *= 2
	}
}

// trimCommonEnds drops the common prefix and suffix of a and b. They never
// need an edit, and trimming them is what makes a lightly edited long
// transcript cheap.
func trimCommonEnds(a, b []int32) ([]int32, []int32) {
	for len(a) > 0 && len(b) > 0 && a[0] == b[0] {
		a, b = a[1:], b[1:]
	}

	for len(a) > 0 && len(b) > 0 && a[len(a)-1] == b[len(b)-1] {
		a, b = a[:len(a)-1], b[:len(b)-1]
	}

	return a, b
}

// initialBand is the first half-width to try for sequences of n <= m tokens.
// The distance is at least the length difference, so the band must be at
// least that wide for the end cell to be reachable at all.
func initialBand(n, m int) int {
	band := m - n
	if band < editDistanceInitialBand {
		band = editDistanceInitialBand
	}

	return band
}

// bandedEditDistance computes the edit distance between a and b (len(a) <=
// len(b)) restricted to the diagonal band |i-j| <= band, and reports whether
// the result is exact. Every optimal alignment of cost d stays within d
// diagonals of the main one, so a result <= band IS the true distance; a
// result above band only says the true distance is larger, and the caller
// retries with a wider band (Ukkonen's cutoff). Uses two rows of len(b)+1.
func bandedEditDistance(a, b []int32, band int) (distance int, exact bool) {
	return bandedEditDP(a, b, band, nil)
}

// bandedEditDP is bandedEditDistance's DP, optionally recording every in-band
// cell's move into trace (rows len(a), width 2*band+1: row i-1, column
// j-i+band for cell (i, j)). trace == nil skips the recording.
func bandedEditDP(a, b []int32, band int, trace *traceTable) (distance int, exact bool) {
	n, m := len(a), len(b)
	if m-n > band {
		return 0, false
	}

	prev := make([]int32, m+1)
	cur := make([]int32, m+1)

	// Row 0: distance i=0 to j is j insertions, inside the band only.
	for j := range prev {
		prev[j] = editDistanceInf
	}

	for j := 0; j <= m && j <= band; j++ {
		prev[j] = int32(j)
	}

	for i := 1; i <= n; i++ {
		lo := i - band
		if lo < 1 {
			lo = 1
		}

		hi := i + band
		if hi > m {
			hi = m
		}

		// The cell just left of the band (or column 0, which is i
		// deletions) seeds this row's left edge.
		switch {
		case lo > 1:
			cur[lo-1] = editDistanceInf
		case i <= band:
			cur[0] = int32(i)
		default:
			cur[0] = editDistanceInf
		}

		ai := a[i-1]

		for j := lo; j <= hi; j++ {
			best := prev[j-1] // substitute or match
			if ai != b[j-1] {
				best++
			}

			move := moveDiag

			if del := prev[j] + 1; del < best { // a[i-1] has no counterpart
				best, move = del, moveDel
			}

			if ins := cur[j-1] + 1; ins < best { // b[j-1] has no counterpart
				best, move = ins, moveIns
			}

			cur[j] = best

			if trace != nil && move != moveDiag {
				trace.set(i-1, j-i+band, move)
			}
		}

		// The cell just right of the band is read by the next row's
		// deletion step, so it must read as unreachable, not stale.
		if hi < m {
			cur[hi+1] = editDistanceInf
		}

		prev, cur = cur, prev
	}

	result := int(prev[m])

	return result, result <= band
}

// alignWords returns an optimal word alignment of a (the reference) to b (the
// hypothesis): a minimum-cost edit script, in order, whose cost is exactly
// editDistance(a, b). It reuses dictation.go's alignOp/tokenAlign vocabulary:
// opMatch/opSub pair a[refIdx] with b[hypIdx], opDel is an a token with no
// counterpart in b (hypIdx -1), opIns a b token with no counterpart in a
// (refIdx -1). ok is false only when the trace table would exceed
// alignMaxCells; the distance is still available from editDistance.
func alignWords(a, b []string) (ops []tokenAlign, distance int, ok bool) {
	ids := make(map[string]int32, len(a)+len(b))
	ai, bi := internTokens(ids, a), internTokens(ids, b)

	prefix := 0
	for prefix < len(ai) && prefix < len(bi) && ai[prefix] == bi[prefix] {
		prefix++
	}

	suffix := 0
	for suffix < len(ai)-prefix && suffix < len(bi)-prefix && ai[len(ai)-1-suffix] == bi[len(bi)-1-suffix] {
		suffix++
	}

	core, distance, ok := alignCore(ai[prefix:len(ai)-suffix], bi[prefix:len(bi)-suffix])
	if !ok {
		return nil, 0, false
	}

	ops = make([]tokenAlign, 0, prefix+len(core)+suffix)

	for i := 0; i < prefix; i++ {
		ops = append(ops, tokenAlign{op: opMatch, refIdx: i, hypIdx: i})
	}

	for _, op := range core {
		if op.refIdx >= 0 {
			op.refIdx += prefix
		}

		if op.hypIdx >= 0 {
			op.hypIdx += prefix
		}

		ops = append(ops, op)
	}

	for k := suffix; k > 0; k-- {
		ops = append(ops, tokenAlign{op: opMatch, refIdx: len(a) - k, hypIdx: len(b) - k})
	}

	return ops, distance, true
}

// alignCore aligns a to b after their common ends were trimmed, widening the
// band exactly as editDistanceIDs does but with a trace table so the path can
// be walked back.
func alignCore(a, b []int32) (ops []tokenAlign, distance int, ok bool) {
	switch {
	case len(a) == 0:
		for j := range b {
			ops = append(ops, tokenAlign{op: opIns, refIdx: -1, hypIdx: j})
		}

		return ops, len(b), true
	case len(b) == 0:
		for i := range a {
			ops = append(ops, tokenAlign{op: opDel, refIdx: i, hypIdx: -1})
		}

		return ops, len(a), true
	}

	// The DP wants the shorter side as its rows (smaller trace table); flip
	// the result back afterwards.
	swapped := len(a) > len(b)
	if swapped {
		a, b = b, a
	}

	n, m := len(a), len(b)
	band := initialBand(n, m)

	for {
		if band > m {
			band = m
		}

		width := 2*band + 1
		if n*width > alignMaxCells {
			return nil, 0, false
		}

		trace := newTraceTable(n, width)

		if d, exact := bandedEditDP(a, b, band, trace); exact {
			return walkTrace(a, b, trace, band, swapped), d, true
		}

		band *= 2
	}
}

// walkTrace follows trace back from the bottom-right cell to the origin and
// returns the path as forward-ordered ops over (a, b) -- or, when swapped,
// over the caller's original (b, a) orientation.
func walkTrace(a, b []int32, trace *traceTable, band int, swapped bool) []tokenAlign {
	var reversed []tokenAlign

	i, j := len(a), len(b)

	for i > 0 || j > 0 {
		switch {
		case i == 0:
			reversed = append(reversed, tokenAlign{op: opIns, refIdx: -1, hypIdx: j - 1})
			j--
		case j == 0:
			reversed = append(reversed, tokenAlign{op: opDel, refIdx: i - 1, hypIdx: -1})
			i--
		default:
			switch trace.get(i-1, j-i+band) {
			case moveDel:
				reversed = append(reversed, tokenAlign{op: opDel, refIdx: i - 1, hypIdx: -1})
				i--
			case moveIns:
				reversed = append(reversed, tokenAlign{op: opIns, refIdx: -1, hypIdx: j - 1})
				j--
			default:
				op := opSub
				if a[i-1] == b[j-1] {
					op = opMatch
				}

				reversed = append(reversed, tokenAlign{op: op, refIdx: i - 1, hypIdx: j - 1})
				i--
				j--
			}
		}
	}

	ops := make([]tokenAlign, len(reversed))

	for k, op := range reversed {
		if swapped {
			// The DP ran with the sides exchanged: what it called a deletion
			// (an a token with no b counterpart) is, in the caller's terms,
			// an insertion, and the indices trade places.
			switch op.op {
			case opDel:
				op = tokenAlign{op: opIns, refIdx: -1, hypIdx: op.refIdx}
			case opIns:
				op = tokenAlign{op: opDel, refIdx: op.hypIdx, hypIdx: -1}
			default:
				op.refIdx, op.hypIdx = op.hypIdx, op.refIdx
			}
		}

		ops[len(reversed)-1-k] = op
	}

	return ops
}
