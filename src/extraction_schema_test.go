package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type extractionSchemaCase struct {
	name string
	raw  string
}

func invalidExtractionSchemaCases(t *testing.T) []extractionSchemaCase {
	t.Helper()
	var cases []extractionSchemaCase
	add := func(name string, mutate func(map[string]any)) {
		var value map[string]any
		require.NoError(t, json.Unmarshal([]byte(validExtraction), &value))
		mutate(value)
		raw, err := json.Marshal(value)
		require.NoError(t, err)
		cases = append(cases, extractionSchemaCase{name, string(raw)})
	}

	// Each required field is tested independently so another malformed field
	// cannot mask a missing validation check.
	for _, path := range []string{
		"participants", "action_items", "decisions", "entities",
		"action_items[0].owner", "action_items[0].task", "action_items[0].due",
		"entities.people", "entities.organizations", "entities.products", "entities.dates",
	} {
		for _, variant := range []string{"missing", "null", "wrong type"} {
			add(path+"/"+variant, func(value map[string]any) {
				parent, key := value, path
				if strings.HasPrefix(path, "entities.") {
					parent = value["entities"].(map[string]any)
					key = strings.TrimPrefix(path, "entities.")
				} else if strings.HasPrefix(path, "action_items[0].") {
					parent = value["action_items"].([]any)[0].(map[string]any)
					key = strings.TrimPrefix(path, "action_items[0].")
				}
				switch variant {
				case "missing":
					delete(parent, key)
				case "null":
					parent[key] = nil
				default:
					parent[key] = 42
				}
			})
		}
	}

	for _, path := range []string{
		"participants", "decisions", "entities.people",
		"entities.organizations", "entities.products", "entities.dates",
	} {
		for _, element := range []any{nil, 42} {
			add(fmt.Sprintf("%s/element %v", path, element), func(value map[string]any) {
				parent, key := value, path
				if strings.HasPrefix(path, "entities.") {
					parent = value["entities"].(map[string]any)
					key = strings.TrimPrefix(path, "entities.")
				}
				parent[key] = []any{"valid first element", element}
			})
		}
	}
	for _, element := range []any{nil, "not an object", map[string]any{"task": "send proposal"}} {
		add(fmt.Sprintf("action_items/second element %v", element), func(value map[string]any) {
			value["action_items"] = append(value["action_items"].([]any), element)
		})
	}
	add("empty entities object", func(value map[string]any) {
		value["entities"] = map[string]any{}
	})
	add("task-only action and empty entities", func(value map[string]any) {
		value["action_items"] = []any{map[string]any{"task": "send proposal"}}
		value["entities"] = map[string]any{}
	})
	return cases
}

func TestValidateExtractionRequiredSchema(t *testing.T) {
	for _, tc := range invalidExtractionSchemaCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ValidateExtraction(tc.raw)
			require.Error(t, err, "must reject %s", tc.raw)
		})
	}
}

func TestValidateExtractionSchemaControls(t *testing.T) {
	empty := `{"participants":[],"action_items":[],"decisions":[],"entities":{"people":[],"organizations":[],"products":[],"dates":[]}}`
	unspecified := strings.ReplaceAll(strings.ReplaceAll(validExtraction, `"owner": "Alice"`, `"owner": "unspecified"`), `"due": "Friday"`, `"due": "unspecified"`)
	for name, raw := range map[string]string{
		"populated":                 validExtraction,
		"empty arrays":              empty,
		"unspecified owner and due": unspecified,
		"code fence":                "```json\n" + unspecified + "\n```",
		"commentary":                "Here is the extraction:\n" + empty,
		// Presence and type, not content restrictions or unknown-key policy.
		"empty strings": strings.ReplaceAll(unspecified, `"unspecified"`, `""`),
		"extra field":   strings.Replace(empty, `"participants":[]`, `"extra":true,"participants":[]`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ValidateExtraction(raw)
			require.NoError(t, err)
		})
	}
}

func TestInformationExtractionTaskRunJudgeRequiredSchema(t *testing.T) {
	for _, tc := range invalidExtractionSchemaCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			var judgeCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				judgeCalls.Add(1)
				writeChatCompletion(w, `{"groundedness":5,"completeness":5,"notes":"unexpected"}`, 20, 10)
			}))
			defer server.Close()
			task := newTestInformationExtractionTask(t, NewGateway(server.URL, "test-key", 5*time.Second))
			judged, err := task.RunJudge(context.Background(), ModelInfo{ID: "candidate"},
				TranscriptRecord{ID: "record", Bucket: "30s", Text: "Please send the proposal."},
				"the-judge", CompletionResult{Text: tc.raw}, nil)
			require.NoError(t, err)
			assert.True(t, judged.SchemaInvalid)
			assert.Equal(t, map[string]float64{"groundedness": 1, "completeness": 1}, judged.Judge)
			assert.Equal(t, int32(0), judgeCalls.Load(), "invalid schema must never reach the gateway")
		})
	}
}
