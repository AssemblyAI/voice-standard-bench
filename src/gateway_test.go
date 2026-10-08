package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGateway_ListModels exercises GET /v1/models against a real HTTP
// server (httptest), matching the actual response shape from
// pkg/api/routers/v1/schemas/models.go rather than a mocked interface —
// this is a direct HTTP call (see ListModels' doc comment on why it bypasses
// the SDK), so testing it as HTTP is testing the real code path.
func TestGateway_ListModels(t *testing.T) {
	type fields struct {
		handler http.HandlerFunc
	}

	tests := []struct {
		name    string
		fields  fields
		wantErr bool
		assert  func(t *testing.T, models []ModelInfo)
	}{
		{
			name: "parses id/name/creator/context/max-completion-tokens, and sends no Authorization header",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					assert.Equal(t, "/v1/models", r.URL.Path)
					assert.Equal(t, http.MethodGet, r.Method)
					// GET /v1/models is genuinely unauthenticated (see
					// NewGateway's doc comment) -- the client must not need
					// to attach a key for this call to work.
					assert.Empty(t, r.Header.Get("Authorization"))
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprint(w, `{"data":[
						{"id":"model-a","name":"Model A","creator":"Acme","context_length":128000,"top_provider":{"max_completion_tokens":4096}},
						{"id":"model-b","name":"Model B","creator":"Acme","context_length":32000,"top_provider":{"max_completion_tokens":2048}}
					]}`)
				},
			},
			assert: func(t *testing.T, models []ModelInfo) {
				require.Len(t, models, 2)
				assert.Equal(t, ModelInfo{
					ID: "model-a", Name: "Model A", Creator: "Acme",
					ContextLength: 128000, MaxCompletionTokens: 4096,
				}, models[0])
			},
		},
		{
			name: "parses global pricing, ignoring regional us/eu overrides",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					fmt.Fprint(w, `{"data":[
						{"id":"model-a","name":"Model A","pricing":{"global":{"prompt":3,"completions":15},"us":{"prompt":99,"completions":99}}}
					]}`)
				},
			},
			assert: func(t *testing.T, models []ModelInfo) {
				require.Len(t, models, 1)
				assert.Equal(t, 3.0, models[0].PromptPricePerMillion)
				assert.Equal(t, 15.0, models[0].CompletionPricePerMillion, "regional us/eu pricing must never override global")
			},
		},
		{
			name: "a model with no pricing block decodes to zero, not an error",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					fmt.Fprint(w, `{"data":[{"id":"model-a","name":"Model A"}]}`)
				},
			},
			assert: func(t *testing.T, models []ModelInfo) {
				require.Len(t, models, 1)
				assert.Zero(t, models[0].PromptPricePerMillion)
				assert.Zero(t, models[0].CompletionPricePerMillion)
			},
		},
		{
			name: "parses supported_parameters",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					fmt.Fprint(w, `{"data":[
						{"id":"model-a","name":"Model A","supported_parameters":["max_tokens","reasoning_effort"]},
						{"id":"model-b","name":"Model B"}
					]}`)
				},
			},
			assert: func(t *testing.T, models []ModelInfo) {
				require.Len(t, models, 2)
				assert.Equal(t, []string{"max_tokens", "reasoning_effort"}, models[0].SupportedParameters)
				assert.True(t, models[0].SupportsParameter("reasoning_effort"))
				assert.Empty(t, models[1].SupportedParameters, "a model with no supported_parameters block decodes to nil, not an error")
				assert.False(t, models[1].SupportsParameter("reasoning_effort"))
			},
		},
		{
			name: "a model with an empty id is dropped, not benchmarked as \"\"",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					fmt.Fprint(w, `{"data":[{"id":"","name":"nameless"},{"id":"real","name":"Real"}]}`)
				},
			},
			assert: func(t *testing.T, models []ModelInfo) {
				require.Len(t, models, 1)
				assert.Equal(t, "real", models[0].ID)
			},
		},
		{
			name: "non-200 status is an error, not an empty list",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
					fmt.Fprint(w, "internal error")
				},
			},
			wantErr: true,
		},
		{
			name: "malformed JSON body is an error",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					fmt.Fprint(w, `{not json`)
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(tt.fields.handler)
			defer server.Close()

			gw := NewGateway(server.URL, "", 5*time.Second)
			models, err := gw.ListModels(context.Background())

			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			tt.assert(t, models)
		})
	}
}

// TestModelInfo_Cost locks in the $/1M-tokens convention (schemas.PricingData's
// own unit — see ModelInfo's doc comment): cost = tokens * price / 1e6.
func TestModelInfo_Cost(t *testing.T) {
	tests := []struct {
		name                        string
		model                       ModelInfo
		promptTokens, completionTok int
		want                        float64
	}{
		{
			name:         "Claude Sonnet-shaped pricing: $3/1M prompt, $15/1M completion",
			model:        ModelInfo{PromptPricePerMillion: 3, CompletionPricePerMillion: 15},
			promptTokens: 1_000_000, completionTok: 1_000_000,
			want: 18,
		},
		{
			name:         "small, realistic token counts",
			model:        ModelInfo{PromptPricePerMillion: 3, CompletionPricePerMillion: 15},
			promptTokens: 301, completionTok: 35,
			want: 301*3.0/1e6 + 35*15.0/1e6,
		},
		{
			name:  "zero tokens costs zero regardless of pricing",
			model: ModelInfo{PromptPricePerMillion: 3, CompletionPricePerMillion: 15},
			want:  0,
		},
		{
			name:         "unpriced model (GET /v1/models reported no pricing) costs zero",
			model:        ModelInfo{},
			promptTokens: 500, completionTok: 500,
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, tt.model.Cost(tt.promptTokens, tt.completionTok), 1e-9)
		})
	}
}

// TestModelInfo_ReasoningEffortLabel locks in the label every
// Sample/Cell/summary/use-case row records in its reasoning-effort column:
// reasoningEffortDefaultLabel ("default") when ReasoningEffort is unset --
// so a benchmark run that never touches -reasoning-effort still gets an
// explicit value in that column, never blank -- and the actual value when
// set.
func TestModelInfo_ReasoningEffortLabel(t *testing.T) {
	t.Run("ReasoningEffort unset returns the default label", func(t *testing.T) {
		assert.Equal(t, "default", ModelInfo{ID: "gpt-6-luna"}.ReasoningEffortLabel())
	})

	t.Run("ReasoningEffort set returns the actual value", func(t *testing.T) {
		effort := "none"
		assert.Equal(t, "none", ModelInfo{ID: "gpt-6-luna", ReasoningEffort: &effort}.ReasoningEffortLabel())
	})
}

// writeChatCompletion writes a minimal, valid chat.completion response body —
// the shape RunCompletion's synchronous call decodes (openai-go's
// ChatCompletion) — with the given content and token usage, AND the
// Content-Type header that decode requires.
//
// That header is not decoration: unlike the streaming path this tool used to
// use (which accepted any content-type and fell back to a generic
// line-scanning SSE decoder regardless), a synchronous response is decoded
// straight into a Go struct, and openai-go's request layer refuses to do
// that for anything but "application/json" — an httptest handler that omits
// it gets Go's content sniffer instead, which calls a bare "{...}" body
// "text/plain" and makes the SDK fail with a decode error that has nothing
// to do with whatever the test is actually checking. Routing every fixture
// through this one function means that mistake can't recur one handler at a
// time — see the "no choices" test below, which still sets it manually
// because it needs a body shape this helper doesn't parameterize.
func writeChatCompletion(w http.ResponseWriter, content string, promptTokens, completionTokens int) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w,
		`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
		content, promptTokens, completionTokens, promptTokens+completionTokens,
	)
}

// TestGateway_RunCompletion exercises synchronous completions against a real
// HTTP server, mirroring TestLLMGatewayFramework_runStreamCompletion's
// structure (table of scenarios, each asserting both the returned value and
// side effects) but against genuine HTTP instead of a mocked sender, since
// RunCompletion has no sender interface to mock — it IS the HTTP client.
func TestGateway_RunCompletion(t *testing.T) {
	type fields struct {
		handler http.HandlerFunc
	}

	tests := []struct {
		name    string
		fields  fields
		wantErr bool
		assert  func(t *testing.T, result CompletionResult)
	}{
		{
			name: "returns completion text and usage from a synchronous JSON response",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					writeChatCompletion(w, "Hello, world", 50, 2)
				},
			},
			assert: func(t *testing.T, result CompletionResult) {
				assert.Equal(t, "Hello, world", result.Text)
				assert.Equal(t, 50, result.PromptTokens)
				assert.Equal(t, 2, result.CompletionTokens)
				assert.Greater(t, result.TotalLatency, time.Duration(0))

				// RawResponse must be the WHOLE response body verbatim --
				// not just the Text this tool extracted from it -- so a
				// failed-sample dump downstream (see failures.go) can show
				// exactly what the gateway actually returned: usage
				// included, not just the one field this struct otherwise
				// surfaces.
				assert.JSONEq(t,
					`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m",`+
						`"choices":[{"index":0,"message":{"role":"assistant","content":"Hello, world"},"finish_reason":"stop"}],`+
						`"usage":{"prompt_tokens":50,"completion_tokens":2,"total_tokens":52}}`,
					result.RawResponse)
			},
		},
		{
			name: "TimeToFirstToken always equals TotalLatency for a synchronous call",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					// A real delay before the single response is written --
					// there is no earlier moment a synchronous client could
					// observe, so both fields must reflect it identically,
					// not just both happen to be zero.
					time.Sleep(50 * time.Millisecond)
					writeChatCompletion(w, "delayed", 1, 1)
				},
			},
			assert: func(t *testing.T, result CompletionResult) {
				assert.GreaterOrEqual(t, result.TotalLatency, 50*time.Millisecond)
				assert.Equal(t, result.TotalLatency, result.TimeToFirstToken,
					"a synchronous response has no observable moment earlier than the full reply")
			},
		},
		{
			name: "a response with no choices is an error, not a garbage result",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					// Content-Type set explicitly (not via writeChatCompletion,
					// whose body shape always carries a choice) so this
					// genuinely exercises the no-choices check in RunCompletion,
					// rather than accidentally erroring earlier on a
					// content-type mismatch and passing for the wrong reason.
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`)
				},
			},
			wantErr: true,
		},
		{
			name: "a non-200 status fails the call rather than returning a garbage result",
			fields: fields{
				handler: func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
					fmt.Fprint(w, `{"message":"upstream provider errored"}`)
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(tt.fields.handler)
			defer server.Close()

			gw := NewGateway(server.URL, "test-key", 5*time.Second)
			result, err := gw.RunCompletion(context.Background(), ModelInfo{ID: "test-model"}, "system prompt", "user message", 128)

			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			tt.assert(t, result)
		})
	}
}

func TestGateway_RunCompletion_sendsAuthorizationHeader(t *testing.T) {
	var gotAuth atomic.Value

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth.Store(r.Header.Get("Authorization"))
		writeChatCompletion(w, "ok", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "secret-token", 5*time.Second)
	_, err := gw.RunCompletion(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 8)
	require.NoError(t, err)

	assert.Equal(t, "secret-token", gotAuth.Load(),
		"the bare token must be sent as-is -- the gateway strips an optional \"Bearer \" prefix but doesn't require one")
}

// TestGateway_RunCompletion_doesNotRetry locks in the rate-limit-integrity
// decision documented on NewGateway: the SDK's default of 2 client-side
// retries must be disabled, or a request that succeeds on a retried attempt
// would silently report that attempt's latency as the model's.
func TestGateway_RunCompletion_doesNotRetry(t *testing.T) {
	var calls atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"message":"rate limited"}`)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)
	_, err := gw.RunCompletion(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 8)

	require.Error(t, err)
	assert.Equal(t, int32(1), calls.Load(), "a 429 must be reported as a failed sample, not retried")
}

// TestGateway_RunCompletion_reasoningEffort locks in the "only set if
// explicitly attached to ModelInfo" contract applyReasoningEffort
// (voicebench.go) relies on: RunCompletion sends model.ReasoningEffort as
// the request's reasoning_effort when set, and omits the field from the
// wire entirely -- not merely as an empty string -- when nil.
func TestGateway_RunCompletion_reasoningEffort(t *testing.T) {
	var gotBody atomic.Value

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody.Store(readBody(t, r))
		writeChatCompletion(w, "ok", 1, 1)
	}))
	defer server.Close()

	gw := NewGateway(server.URL, "test-key", 5*time.Second)

	t.Run("set on the model, sent on the wire", func(t *testing.T) {
		effort := "low"
		_, err := gw.RunCompletion(context.Background(), ModelInfo{ID: "test-model", ReasoningEffort: &effort}, "sys", "user", 8)
		require.NoError(t, err)
		assert.Contains(t, gotBody.Load().(string), `"reasoning_effort":"low"`)
	})

	t.Run("nil on the model, never sent", func(t *testing.T) {
		_, err := gw.RunCompletion(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 8)
		require.NoError(t, err)
		assert.NotContains(t, gotBody.Load().(string), "reasoning_effort")
	})
}

// writeSSEChunk writes one Server-Sent Events "data: <json>\n\n" frame and
// flushes it immediately -- without the flush, an httptest handler's writes
// can sit buffered until the handler returns, which would defeat every test
// below that depends on chunks actually arriving at different times.
func writeSSEChunk(w http.ResponseWriter, jsonBody string) {
	fmt.Fprintf(w, "data: %s\n\n", jsonBody)

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// writeChatCompletionStream writes a minimal, valid chat.completion.chunk
// SSE stream -- the shape RunCompletionStream's streaming call decodes
// (openai-go's ChatCompletionChunk) -- one chunk per element of
// contentChunks (each carrying that piece of the reply as delta.content), a
// final chunk carrying usage with an empty choices list (the
// stream_options.include_usage shape every real provider uses), and the
// "[DONE]" sentinel every real provider ends a stream with.
func writeChatCompletionStream(w http.ResponseWriter, contentChunks []string, promptTokens, completionTokens int) {
	w.Header().Set("Content-Type", "text/event-stream")

	for _, content := range contentChunks {
		writeSSEChunk(w, fmt.Sprintf(
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
				`"choices":[{"index":0,"delta":{"content":%q},"finish_reason":null}]}`,
			content,
		))
	}

	writeSSEChunk(w, fmt.Sprintf(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
		promptTokens, completionTokens, promptTokens+completionTokens,
	))

	fmt.Fprint(w, "data: [DONE]\n\n")

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// TestGateway_RunCompletionStream exercises the streaming call against a
// real HTTP server, the same way TestGateway_RunCompletion does for the
// synchronous one -- proving the one property that call can't have at all:
// a TimeToFirstToken genuinely earlier than TotalLatency.
func TestGateway_RunCompletionStream(t *testing.T) {
	t.Run("TimeToFirstToken reflects the FIRST content chunk, strictly before TotalLatency reflects the last", func(t *testing.T) {
		const firstChunkDelay = 80 * time.Millisecond
		const secondChunkDelay = 80 * time.Millisecond

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")

			time.Sleep(firstChunkDelay)
			writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
				`"choices":[{"index":0,"delta":{"content":"Hello, "},"finish_reason":null}]}`)

			time.Sleep(secondChunkDelay)
			writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
				`"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}`)

			writeSSEChunk(w, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",`+
				`"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":2,"total_tokens":52}}`)

			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)
		result, err := gw.RunCompletionStream(context.Background(), ModelInfo{ID: "test-model"}, "system prompt", "user message", 128)
		require.NoError(t, err)

		assert.Equal(t, "Hello, world", result.Text, "content deltas across every chunk must be concatenated in order")
		assert.Equal(t, 50, result.PromptTokens)
		assert.Equal(t, 2, result.CompletionTokens)

		assert.GreaterOrEqual(t, result.TimeToFirstToken, firstChunkDelay,
			"TTFT must not be observed before the first content chunk actually arrives")
		assert.GreaterOrEqual(t, result.TotalLatency, firstChunkDelay+secondChunkDelay,
			"total latency must reflect the full stream, including the second chunk's delay")
		assert.Less(t, result.TimeToFirstToken, result.TotalLatency,
			"unlike RunCompletion's synchronous call, a streaming call has a real, strictly earlier first-token moment")
	})

	t.Run("a stream that never produces content is an error, not a zero-valued success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Only the usage chunk (empty choices) and [DONE] -- no content
			// chunk ever arrives, so TTFT can never be set.
			writeChatCompletionStream(w, nil, 5, 0)
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)
		_, err := gw.RunCompletionStream(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 128)
		require.Error(t, err)
	})

	t.Run("a non-200 status fails the call rather than returning a garbage result", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"message":"upstream provider errored"}`)
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)
		_, err := gw.RunCompletionStream(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 128)
		require.Error(t, err)
	})

	t.Run("a non-200 status's raw error body is recoverable via RawAPIError, same as the synchronous path", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"code":"rate_limited","message":"slow down","param":"","type":"rate_limit_error"}}`)
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)
		_, err := gw.RunCompletionStream(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 128)
		require.Error(t, err)

		raw := RawAPIError(err)
		require.NotEmpty(t, raw)
		assert.JSONEq(t, `{"code":"rate_limited","message":"slow down","param":"","type":"rate_limit_error"}`, raw)
	})

	t.Run("model.ReasoningEffort is sent when set, omitted from the wire when nil", func(t *testing.T) {
		var gotBody atomic.Value

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotBody.Store(readBody(t, r))
			writeChatCompletionStream(w, []string{"ok"}, 1, 1)
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)

		effort := "low"
		_, err := gw.RunCompletionStream(context.Background(), ModelInfo{ID: "test-model", ReasoningEffort: &effort}, "sys", "user", 128)
		require.NoError(t, err)
		assert.Contains(t, gotBody.Load().(string), `"reasoning_effort":"low"`)

		_, err = gw.RunCompletionStream(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 128)
		require.NoError(t, err)
		assert.NotContains(t, gotBody.Load().(string), "reasoning_effort")
	})
}

// TestRawAPIError exercises RawAPIError directly: it must extract the
// gateway's raw "error" envelope verbatim when the SDK actually returns one
// (a non-2xx response shaped like an OpenAI-compatible error), and return ""
// for any other kind of failure -- one where there was never a response
// body to extract from at all.
func TestRawAPIError(t *testing.T) {
	t.Run("extracts the raw error envelope from a non-2xx response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"code":"bad_request","message":"nope","param":"model","type":"invalid_request_error"}}`)
		}))
		defer server.Close()

		gw := NewGateway(server.URL, "test-key", 5*time.Second)
		_, err := gw.RunCompletion(context.Background(), ModelInfo{ID: "test-model"}, "sys", "user", 8)
		require.Error(t, err)

		raw := RawAPIError(err)
		require.NotEmpty(t, raw)
		assert.JSONEq(t, `{"code":"bad_request","message":"nope","param":"model","type":"invalid_request_error"}`, raw)
	})

	t.Run("returns empty for a failure that never got a response body", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		gw := NewGateway("http://127.0.0.1:0", "test-key", 5*time.Second)
		_, err := gw.RunCompletion(ctx, ModelInfo{ID: "test-model"}, "sys", "user", 8)
		require.Error(t, err)

		assert.Empty(t, RawAPIError(err), "a canceled context never reaches the API, so there is no error body to extract")
	})

	t.Run("returns empty for any plain error unrelated to the API", func(t *testing.T) {
		assert.Empty(t, RawAPIError(fmt.Errorf("some other failure")))
	})
}
