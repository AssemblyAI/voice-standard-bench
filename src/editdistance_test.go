package main

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// naiveEditDistance is the textbook full-matrix Levenshtein -- slow and
// memory-hungry, which is exactly why editDistance doesn't use it, but
// obviously correct, so it is the oracle the banded implementation is checked
// against.
func naiveEditDistance(a, b []string) int {
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)

	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(a); i++ {
		cur[0] = i

		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}

			cur[j] = min(prev[j-1]+cost, prev[j]+1, cur[j-1]+1)
		}

		prev, cur = cur, prev
	}

	return prev[len(b)]
}

func words(s string) []string { return strings.Fields(s) }

func TestEditDistance(t *testing.T) {
	tests := []struct {
		name string
		a, b string
		want int
	}{
		{"identical", "the quick brown fox", "the quick brown fox", 0},
		{"both empty", "", "", 0},
		{"empty against non-empty is every token inserted", "", "a b c", 3},
		{"non-empty against empty is every token deleted", "a b c", "", 3},
		{"one substitution", "the quick brown fox", "the quick red fox", 1},
		{"one insertion", "the quick brown fox", "the quick brown furry fox", 1},
		{"one deletion", "the quick brown fox", "the brown fox", 1},
		{"completely different", "apple banana cherry", "dog elephant fox", 3},
		{"classic kitten/sitting shape", "k i t t e n", "s i t t i n g", 3},
		{"a swap of two adjacent tokens costs two, not zero", "a b", "b a", 2},
		{"order matters: the same words reversed", "one two three four", "four three two one", 4},
		{"a repeated token against a single one", "a a a a", "a", 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, editDistance(words(tt.a), words(tt.b)))
			assert.Equal(t, tt.want, editDistance(words(tt.b), words(tt.a)), "the distance is symmetric")
			assert.Equal(t, tt.want, naiveEditDistance(words(tt.a), words(tt.b)), "and the oracle agrees")
		})
	}
}

// TestEditDistance_matchesNaiveOnRandomInputs is the property test that earns
// the banded implementation its trust: for thousands of random pairs -- tiny
// alphabets so alignments are genuinely ambiguous, lengths from empty up to
// well past editDistanceInitialBand so the band has to widen -- it must agree
// with the full-matrix oracle exactly.
func TestEditDistance_matchesNaiveOnRandomInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	randomTokens := func(n, alphabet int) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = fmt.Sprintf("t%d", rng.Intn(alphabet))
		}

		return out
	}

	for trial := 0; trial < 4000; trial++ {
		alphabet := 1 + rng.Intn(5)
		a := randomTokens(rng.Intn(90), alphabet)
		b := randomTokens(rng.Intn(90), alphabet)

		want := naiveEditDistance(a, b)
		got := editDistance(a, b)

		require.Equal(t, want, got, "trial %d: a=%v b=%v", trial, a, b)
	}
}

// TestEditDistance_matchesNaiveOnNearIdenticalInputs covers the shape real
// corrections have: a long sequence with a handful of scattered edits, where
// the first band attempt succeeds and trimming the shared prefix/suffix does
// most of the work.
func TestEditDistance_matchesNaiveOnNearIdenticalInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(2))

	for trial := 0; trial < 300; trial++ {
		n := 50 + rng.Intn(400)
		a := make([]string, n)

		for i := range a {
			a[i] = fmt.Sprintf("w%d", rng.Intn(60))
		}

		b := append([]string(nil), a...)

		edits := rng.Intn(12)

		for e := 0; e < edits; e++ {
			if len(b) == 0 {
				break
			}

			pos := rng.Intn(len(b))

			switch rng.Intn(3) {
			case 0:
				b[pos] = "edited"
			case 1:
				b = append(b[:pos], b[pos+1:]...)
			default:
				b = append(b[:pos], append([]string{"inserted"}, b[pos:]...)...)
			}
		}

		require.Equal(t, naiveEditDistance(a, b), editDistance(a, b), "trial %d", trial)
	}
}

// TestEditDistance_longTranscriptWithKnownEdits pins an exact answer at the
// size of the largest bucket (~15,000 words). Every token is unique, and
// every edit swaps one for a fresh token no other position carries, so the
// distance can only be the number of edits -- and a full-matrix
// implementation would need gigabytes to say so.
func TestEditDistance_longTranscriptWithKnownEdits(t *testing.T) {
	const n = 15000

	a := make([]string, n)
	for i := range a {
		a[i] = fmt.Sprintf("word%d", i)
	}

	b := append([]string(nil), a...)

	rng := rand.New(rand.NewSource(3))
	for _, pos := range rng.Perm(n)[:60] {
		b[pos] = fmt.Sprintf("fresh%d", pos)
	}

	assert.Equal(t, 60, editDistance(a, b))
}

// TestEditDistance_garbageCandidateWidensToTheFullMatrix: a runaway output
// many times the truth's length has a distance near its own length, which
// forces the band all the way out -- and must still be exact, not a cutoff.
func TestEditDistance_garbageCandidateWidensToTheFullMatrix(t *testing.T) {
	truth := words("the meeting is at three")
	runaway := words(strings.Repeat("and then the meeting is at three ", 40))

	assert.Equal(t, naiveEditDistance(truth, runaway), editDistance(truth, runaway))
	assert.GreaterOrEqual(t, editDistance(truth, runaway), len(runaway)-len(truth), "never below the length difference")
}

func TestBandedEditDistance_reportsWhenTheBandIsTooNarrow(t *testing.T) {
	intern := func(tokens []string) []int32 {
		out := make([]int32, len(tokens))

		for i, tok := range tokens {
			out[i] = int32(len(tok)*31) + int32(tok[0]) // enough to tell the test tokens apart
		}

		return out
	}

	a := intern(words("aa bb cc dd ee ff"))
	b := intern(words("gg hh ii jj kk ll"))

	t.Run("a distance above the band is reported as inexact", func(t *testing.T) {
		_, exact := bandedEditDistance(a, b, 2)
		assert.False(t, exact, "all six differ, which a half-width of 2 cannot prove")
	})

	t.Run("a distance within the band is exact", func(t *testing.T) {
		got, exact := bandedEditDistance(a, b, 6)
		assert.True(t, exact)
		assert.Equal(t, 6, got)
	})

	t.Run("a length difference beyond the band cannot even reach the end cell", func(t *testing.T) {
		_, exact := bandedEditDistance(a[:1], b, 2)
		assert.False(t, exact)
	})
}

// checkAlignment verifies everything an alignment must satisfy to be trusted
// as THE alignment behind a score: its cost is the optimal distance, every
// reference and hypothesis token is accounted for exactly once and in order,
// and a "match" really is a match while a "substitution" really isn't.
func checkAlignment(t *testing.T, a, b []string, ops []tokenAlign, distance int) {
	t.Helper()

	require.Equal(t, naiveEditDistance(a, b), distance, "alignWords' distance is the optimum")

	cost, nextRef, nextHyp := 0, 0, 0

	for _, op := range ops {
		switch op.op {
		case opMatch, opSub:
			require.Equal(t, nextRef, op.refIdx, "reference tokens are consumed in order, none skipped")
			require.Equal(t, nextHyp, op.hypIdx, "hypothesis tokens are consumed in order, none skipped")

			if op.op == opMatch {
				require.Equal(t, a[op.refIdx], b[op.hypIdx], "a match pairs equal tokens")
			} else {
				require.NotEqual(t, a[op.refIdx], b[op.hypIdx], "a substitution pairs different tokens")

				cost++
			}

			nextRef++
			nextHyp++
		case opDel:
			require.Equal(t, nextRef, op.refIdx)
			require.Equal(t, -1, op.hypIdx)

			nextRef++
			cost++
		case opIns:
			require.Equal(t, -1, op.refIdx)
			require.Equal(t, nextHyp, op.hypIdx)

			nextHyp++
			cost++
		default:
			t.Fatalf("unknown op %v", op.op)
		}
	}

	require.Equal(t, len(a), nextRef, "every reference token appears")
	require.Equal(t, len(b), nextHyp, "every hypothesis token appears")
	require.Equal(t, distance, cost, "the ops add up to the reported distance")
}

func TestAlignWords(t *testing.T) {
	tests := []struct {
		name string
		a, b string
		want []tokenAlign
	}{
		{"both empty", "", "", []tokenAlign{}},
		{"identical", "a b c", "a b c", []tokenAlign{
			{opMatch, 0, 0}, {opMatch, 1, 1}, {opMatch, 2, 2},
		}},
		{"one substitution in the middle", "the quick brown fox", "the slow brown fox", []tokenAlign{
			{opMatch, 0, 0}, {opSub, 1, 1}, {opMatch, 2, 2}, {opMatch, 3, 3},
		}},
		{"a hypothesis-only token is an insertion", "a c", "a b c", []tokenAlign{
			{opMatch, 0, 0}, {opIns, -1, 1}, {opMatch, 1, 2},
		}},
		{"a reference-only token is a deletion", "a b c", "a c", []tokenAlign{
			{opMatch, 0, 0}, {opDel, 1, -1}, {opMatch, 2, 1},
		}},
		{"against an empty hypothesis everything is deleted", "a b", "", []tokenAlign{
			{opDel, 0, -1}, {opDel, 1, -1},
		}},
		{"against an empty reference everything is inserted", "", "a b", []tokenAlign{
			{opIns, -1, 0}, {opIns, -1, 1},
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, b := words(tt.a), words(tt.b)

			got, distance, ok := alignWords(a, b)

			require.True(t, ok)
			assert.Equal(t, tt.want, got)
			checkAlignment(t, a, b, got, distance)
		})
	}
}

// TestAlignWords_isAnOptimalAlignmentOnRandomInputs holds alignWords to the
// same standard as editDistance: for thousands of random pairs -- both
// orientations (the shorter side may be either), ambiguous tiny alphabets,
// lengths that force the band to widen -- the script it returns must be a
// valid, in-order, optimal alignment.
func TestAlignWords_isAnOptimalAlignmentOnRandomInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(11))

	randomTokens := func(n, alphabet int) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = fmt.Sprintf("t%d", rng.Intn(alphabet))
		}

		return out
	}

	for trial := 0; trial < 3000; trial++ {
		alphabet := 1 + rng.Intn(5)
		a := randomTokens(rng.Intn(80), alphabet)
		b := randomTokens(rng.Intn(80), alphabet)

		ops, distance, ok := alignWords(a, b)

		require.True(t, ok, "trial %d", trial)
		checkAlignment(t, a, b, ops, distance)
		require.Equal(t, editDistance(a, b), distance, "trial %d: the alignment and the score's distance can never disagree", trial)
	}
}

func TestAlignWords_longTranscriptWithKnownEdits(t *testing.T) {
	const n = 15000

	a := make([]string, n)
	for i := range a {
		a[i] = fmt.Sprintf("word%d", i)
	}

	b := append([]string(nil), a...)

	rng := rand.New(rand.NewSource(5))
	for _, pos := range rng.Perm(n)[:60] {
		b[pos] = fmt.Sprintf("fresh%d", pos)
	}

	ops, distance, ok := alignWords(a, b)

	require.True(t, ok)
	assert.Equal(t, 60, distance)
	assert.Len(t, ops, n, "60 substitutions among 14,940 matches, nothing inserted or deleted")

	subs := 0

	for _, op := range ops {
		if op.op == opSub {
			subs++
		}
	}

	assert.Equal(t, 60, subs)
}

// TestAlignWords_aRunawayCandidateStillAligns: output many times the truth's
// length is the very case worth looking at in the debug view, and it forces
// the band out to the whole matrix.
func TestAlignWords_aRunawayCandidateStillAligns(t *testing.T) {
	truth := words("the meeting is at three")
	runaway := words(strings.Repeat("and then the meeting is at three ", 40))

	ops, distance, ok := alignWords(truth, runaway)

	require.True(t, ok)
	checkAlignment(t, truth, runaway, ops, distance)
}

// TestAlignWords_givesUpInsteadOfAllocatingWithoutBound pins the memory
// guard: a pair whose trace table would exceed alignMaxCells reports ok=false
// (the caller falls back to no alignment) rather than allocating it, while a
// pair that needs no table at all -- identical, or trimmed to nothing -- still
// aligns however small the cap.
func TestAlignWords_givesUpInsteadOfAllocatingWithoutBound(t *testing.T) {
	original := alignMaxCells
	alignMaxCells = 100

	t.Cleanup(func() { alignMaxCells = original })

	a := make([]string, 200)
	b := make([]string, 200)

	for i := range a {
		a[i] = fmt.Sprintf("a%d", i)
		b[i] = fmt.Sprintf("b%d", i) // nothing in common, so no trimming and a wide band
	}

	_, _, ok := alignWords(a, b)
	assert.False(t, ok, "200x200 of nothing in common needs far more than 100 cells")

	ops, distance, ok := alignWords(a, a)
	require.True(t, ok)
	assert.Equal(t, 0, distance)
	assert.Len(t, ops, len(a))
}

func TestTraceTable_roundTripsEveryMove(t *testing.T) {
	const rows, width = 7, 11

	table := newTraceTable(rows, width)
	want := map[[2]int]uint8{}

	// A deterministic scatter of all three moves (moveDiag is the zero
	// default, so it is "set" by leaving it alone).
	for r := 0; r < rows; r++ {
		for c := 0; c < width; c++ {
			move := uint8((r*3 + c*5) % 3)
			want[[2]int{r, c}] = move

			if move != moveDiag {
				table.set(r, c, move)
			}
		}
	}

	for r := 0; r < rows; r++ {
		for c := 0; c < width; c++ {
			require.Equal(t, want[[2]int{r, c}], table.get(r, c), "cell (%d, %d)", r, c)
		}
	}
}
