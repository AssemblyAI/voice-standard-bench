package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/AssemblyAI/voice-standard-bench/src/prompts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractJSONObject(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    string
		wantErr bool
	}{
		{
			name:  "bare JSON object",
			input: `{"a":1}`,
			want:  `{"a":1}`,
		},
		{
			name:  "wrapped in a markdown code fence, the model's own preamble ignored",
			input: "Here is my evaluation:\n```json\n{\"a\":1}\n```",
			want:  `{"a":1}`,
		},
		{
			name:  "chain-of-thought reasoning before the object, as every judge prompt actually asks for",
			input: "Faithfulness: the summary is accurate. Coverage: it misses the deadline.\n\n{\"a\":1}",
			want:  `{"a":1}`,
		},
		{
			name:    "no braces at all",
			input:   "I refuse to grade this.",
			wantErr: true,
		},
		{
			name:    "a stray closing brace before any opening one",
			input:   "}{",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractJSONObject(tt.input)

			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestParseJudgeScores(t *testing.T) {
	rubric := []string{"faithfulness", "coverage"}

	tests := []struct {
		name    string
		text    string
		rubric  []string
		want    map[string]float64
		wantErr bool
	}{
		{
			name:   "clean JSON with every rubric field",
			text:   `{"faithfulness": 4, "coverage": 5, "notes": "fine"}`,
			rubric: rubric,
			want:   map[string]float64{"faithfulness": 4, "coverage": 5},
		},
		{
			name:   "reasoning prose before the JSON is tolerated",
			text:   "The summary is mostly faithful but misses the follow-up date.\n\n{\"faithfulness\": 4, \"coverage\": 3, \"notes\": \"missed follow-up\"}",
			rubric: rubric,
			want:   map[string]float64{"faithfulness": 4, "coverage": 3},
		},
		{
			name:    "missing a rubric dimension is an error, not a silent zero",
			text:    `{"faithfulness": 4, "notes": "fine"}`,
			rubric:  rubric,
			wantErr: true,
		},
		{
			name:    "a non-numeric score is an error",
			text:    `{"faithfulness": "high", "coverage": 5}`,
			rubric:  rubric,
			wantErr: true,
		},
		{
			name:   "an out-of-range score is clamped into the documented 1-5 range",
			text:   `{"faithfulness": 10, "coverage": 0}`,
			rubric: rubric,
			want:   map[string]float64{"faithfulness": 5, "coverage": 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseJudgeScores(tt.text, tt.rubric)

			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestValidateExtraction(t *testing.T) {
	valid := `{
		"participants": ["Alice", "Bob"],
		"action_items": [{"owner": "Alice", "task": "send the proposal", "due": "Friday"}],
		"decisions": ["use Postgres"],
		"entities": {"people": [], "organizations": ["Acme"], "products": [], "dates": ["Friday"]}
	}`

	tests := []struct {
		name    string
		input   string
		wantErr bool
		assert  func(t *testing.T, e Extraction)
	}{
		{
			name:  "valid schema with every field populated",
			input: valid,
			assert: func(t *testing.T, e Extraction) {
				assert.Equal(t, []string{"Alice", "Bob"}, e.Participants)
				require.Len(t, e.ActionItems, 1)
				assert.Equal(t, "Friday", e.ActionItems[0].Due)
				assert.Equal(t, []string{"Acme"}, e.Entities.Organizations)
			},
		},
		{
			name: "empty arrays for every field is valid -- \"nothing to report\" is not \"malformed\"",
			input: `{"participants": [], "action_items": [], "decisions": [],
				"entities": {"people": [], "organizations": [], "products": [], "dates": []}}`,
			assert: func(t *testing.T, e Extraction) {
				assert.Empty(t, e.Participants)
			},
		},
		{
			name:  "wrapped in a code fence despite the prompt saying not to",
			input: "```json\n" + valid + "\n```",
			assert: func(t *testing.T, e Extraction) {
				assert.Equal(t, []string{"Alice", "Bob"}, e.Participants)
			},
		},
		{
			name:    "missing the entities key entirely is invalid, even though every other key is present",
			input:   `{"participants": [], "action_items": [], "decisions": []}`,
			wantErr: true,
		},
		{
			name:    "action_items shaped as strings instead of objects is invalid",
			input:   `{"participants": [], "action_items": ["do the thing"], "decisions": [], "entities": {}}`,
			wantErr: true,
		},
		{
			name:    "not JSON at all",
			input:   "I couldn't find anything to extract.",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ValidateExtraction(tt.input)

			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			tt.assert(t, got)
		})
	}
}

// TestRunJudge_endToEnd exercises RunJudge -> RunCompletion -> real JSON
// response parsing -> parseJudgeScores as one path against an httptest
// server, rather than unit-testing parseJudgeScores alone — this is the
// integration seam that matters: a correct parser wired to a broken
// transcript/candidate message format would still fail in production.
func TestRunJudge_endToEnd(t *testing.T) {
	var receivedUserMessage string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		receivedUserMessage = string(body)

		content := "The summary is accurate and covers the key point.\n\n" +
			`{"faithfulness": 5, "coverage": 4, "conciseness": 4, "notes": "none"}`
		writeChatCompletion(w, content, 30, 20)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	scores, err := RunJudge(context.Background(), gw, "judge-model", spec, "the source transcript", "a candidate summary")
	require.NoError(t, err)

	assert.Equal(t, map[string]float64{"faithfulness": 5, "coverage": 4, "conciseness": 4}, scores)
	assert.Contains(t, receivedUserMessage, "the source transcript", "judge must receive the transcript")
	assert.Contains(t, receivedUserMessage, "a candidate summary", "judge must receive the candidate output")
}

func TestRunJudge_unparseableReplyIsAQueryErrorNotAZeroScore(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatCompletion(w, "I am not able to grade this content.", 10, 8)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	_, err := RunJudge(context.Background(), gw, "judge-model", spec, "transcript", "summary")
	require.Error(t, err, "a judge reply with no JSON object must surface as an error, not a fabricated score")
}

// TestRunJudge_failureModesCarryRequestAndResponseForReview locks in that
// every RunJudge failure mode -- the completion call itself failing
// (non-2xx), a reply with no JSON object, and a reply whose JSON is missing
// a rubric field -- comes back as a *JudgeError carrying the exact judge
// input (Request, the verbatim JSON body naming the transcript/candidate
// output that was sent) and output (Response, the gateway's raw JSON body)
// that produced it, so a failure recorder can save them for review instead
// of only the error string.
func TestRunJudge_failureModesCarryRequestAndResponseForReview(t *testing.T) {
	spec, ok := prompts.For(prompts.Summarization)
	require.True(t, ok)

	t.Run("non-2xx response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":{"code":"bad_gateway","message":"upstream exploded","param":null,"type":"api_error"}}`))
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)

		_, err := RunJudge(context.Background(), gw, "judge-model", spec, "the source transcript", "a candidate summary")
		require.Error(t, err)

		var judgeErr *JudgeError
		require.ErrorAs(t, err, &judgeErr)
		assert.Contains(t, judgeErr.Request, "the source transcript", "Request must show what the judge was asked to grade")
		assert.JSONEq(t, `{"code":"bad_gateway","message":"upstream exploded","param":null,"type":"api_error"}`, judgeErr.Response,
			"Response must carry the gateway's raw error body")
	})

	t.Run("no JSON object in the reply", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeChatCompletion(w, "I refuse to grade this.", 10, 8)
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)

		_, err := RunJudge(context.Background(), gw, "judge-model", spec, "the source transcript", "a candidate summary")
		require.Error(t, err)

		var judgeErr *JudgeError
		require.ErrorAs(t, err, &judgeErr)
		assert.Contains(t, judgeErr.Request, "the source transcript", "Request must show what the judge was asked to grade")
		assert.Contains(t, judgeErr.Response, "I refuse to grade this.", "Response must carry the judge's raw reply")
	})

	t.Run("JSON missing a rubric field", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeChatCompletion(w, `{"faithfulness": 4, "notes": "fine"}`, 10, 8)
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)

		_, err := RunJudge(context.Background(), gw, "judge-model", spec, "the source transcript", "a candidate summary")
		require.Error(t, err)

		var judgeErr *JudgeError
		require.ErrorAs(t, err, &judgeErr)
		assert.Contains(t, judgeErr.Request, "the source transcript", "Request must show what the judge was asked to grade")
		assert.Contains(t, judgeErr.Response, "faithfulness", "Response must carry the judge's raw reply")
	})
}

func TestParseJudgeReply_returnsTheJudgesNotes(t *testing.T) {
	rubric := []string{"faithfulness"}

	t.Run("notes ride along with the scores", func(t *testing.T) {
		scores, notes, err := parseJudgeReply("One word differs.\n"+`{"faithfulness": 4, "notes": "  two words were dropped  "}`, rubric)

		require.NoError(t, err)
		assert.Equal(t, map[string]float64{"faithfulness": 4}, scores)
		assert.Equal(t, "two words were dropped", notes, "trimmed")
	})

	t.Run("notes are optional", func(t *testing.T) {
		_, notes, err := parseJudgeReply(`{"faithfulness": 5}`, rubric)

		require.NoError(t, err)
		assert.Empty(t, notes)
	})

	t.Run("a notes field that is not text is ignored, never an error", func(t *testing.T) {
		scores, notes, err := parseJudgeReply(`{"faithfulness": 5, "notes": 12}`, rubric)

		require.NoError(t, err)
		assert.Equal(t, 5.0, scores["faithfulness"])
		assert.Empty(t, notes)
	})

	t.Run("a missing rubric field is still an error", func(t *testing.T) {
		_, _, err := parseJudgeReply(`{"notes": "no score"}`, rubric)

		assert.Error(t, err)
	})
}
