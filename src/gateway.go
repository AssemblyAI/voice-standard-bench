package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// reasoningEffortParam is the value GET /v1/models' supported_parameters
// carries for a model that accepts reasoning_effort (see
// pkg/models/registry.go's own SupportedParameters entries) -- the wire
// parameter name, not the CLI flag's.
const reasoningEffortParam = "reasoning_effort"

// Gateway is a thin client for one llm_gateway deployment: GET /v1/models via
// a direct HTTP call (that route is gateway-specific, not part of the OpenAI
// chat-completions surface), and POST /v1/chat/completions via the official
// openai-go SDK's synchronous (non-streaming) call — the gateway speaks the
// OpenAI chat.completion response shape regardless of which provider serves
// the model (see pkg/models/*/stream.go for the streaming equivalent this
// tool deliberately does not use; see RunCompletion for why), so the SDK's
// request construction and response parsing are exactly the code this tool
// would otherwise have to hand-roll.
type Gateway struct {
	client     openai.Client
	httpClient *http.Client
	baseURL    string
	timeout    time.Duration
}

// NewGateway returns a client for baseURL (e.g. https://llm-gateway.assemblyai.com).
// apiKey is sent verbatim as the Authorization header — the gateway strips an
// optional "Bearer " prefix but doesn't require one (pkg/service/data_access/auth.go),
// and AssemblyAI clients conventionally send the bare token.
//
// The SDK's default of 2 client-side retries is disabled: the gateway
// enforces a per-account, per-model rate limit (pkg/service/rate_limit,
// default 30 req/min) that returns 429 under load, and a client-side retry
// that succeeds on its second attempt would report that attempt's latency,
// not the model's — silently corrupting exactly the numbers this tool exists
// to measure. A 429 or transient 5xx is reported as a failed sample instead
// (see RunCompletion).
func NewGateway(baseURL, apiKey string, timeout time.Duration) *Gateway {
	trimmed := strings.TrimRight(baseURL, "/")

	return &Gateway{
		client: openai.NewClient(
			option.WithBaseURL(trimmed+"/v1/"),
			option.WithHeader("Authorization", apiKey),
			option.WithMaxRetries(0),
		),
		httpClient: &http.Client{Timeout: timeout},
		baseURL:    trimmed,
		timeout:    timeout,
	}
}

// ModelInfo is the slice of GET /v1/models this tool needs to pick candidate
// models, size requests to them, and cost their usage. See
// pkg/api/routers/v1/schemas/models.go for the full response shape —
// available regions and default sampling params are irrelevant here and
// deliberately left undecoded; pricing is decoded (Global only, never the
// US/EU regional overrides — this tool has no notion of which region a
// request lands in) since cost-per-quality (see summary.go) needs it.
type ModelInfo struct {
	ID                  string // what to send as ChatCompletionNewParams.Model
	Name                string
	Creator             string
	ContextLength       int
	MaxCompletionTokens int
	// PromptPricePerMillion/CompletionPricePerMillion are USD per 1M tokens
	// (schemas.PricingData's own unit -- see Cost). Zero for a model GET
	// /v1/models reports no pricing for, same as an absent ContextLength.
	PromptPricePerMillion     float64
	CompletionPricePerMillion float64
	// SupportedParameters lists the chat-completion parameters this model's
	// provider actually accepts (schemas.ModelDetails.SupportedParameters --
	// e.g. "reasoning_effort", "tools", "response_format"). Used to decide
	// whether ReasoningEffort is safe to request at all -- see
	// SupportsParameter and applyReasoningEffort in voicebench.go.
	SupportedParameters []string
	// ReasoningEffort, when non-nil, is sent as the request's
	// reasoning_effort by RunCompletion/RunCompletionStream. Nil means "say
	// nothing" -- the gateway/provider's own default applies -- rather than
	// this tool silently picking a value; it is only ever set by
	// applyReasoningEffort (voicebench.go), and only after confirming
	// SupportedParameters actually lists reasoningEffortParam, never set
	// directly from the -reasoning-effort flag.
	ReasoningEffort *string
}

// SupportsParameter reports whether GET /v1/models advertised support for
// the named chat-completion parameter (e.g. reasoningEffortParam) in this
// model's SupportedParameters list.
func (m ModelInfo) SupportsParameter(name string) bool {
	return slices.Contains(m.SupportedParameters, name)
}

// reasoningEffortDefaultLabel is what ReasoningEffortLabel reports for a
// model with no ReasoningEffort set -- so every Sample/Cell/summary/
// use-case row always carries an explicit reasoning-effort value, never a
// blank column, whether or not -reasoning-effort was ever passed.
const reasoningEffortDefaultLabel = "default"

// ReasoningEffortLabel is what every Sample/Cell/summary/use-case row
// records in its reasoning-effort column for this model: the actual value
// when ReasoningEffort is set, or reasoningEffortDefaultLabel when nil (the
// flag was never passed, or this model doesn't support it -- see
// applyReasoningEffort in voicebench.go). Model and reasoning effort
// together are the unique key everywhere downstream groups by both
// (metrics.go's Aggregate, canonical.go's merge key, summary.go, use_cases.go),
// so the same base model benchmarked at two different efforts lands as two
// distinct rows instead of one overwriting the other.
func (m ModelInfo) ReasoningEffortLabel() string {
	if m.ReasoningEffort == nil {
		return reasoningEffortDefaultLabel
	}

	return *m.ReasoningEffort
}

// Cost returns the USD cost of one completion using promptTokens and
// completionTokens, from m's global per-million-token pricing.
func (m ModelInfo) Cost(promptTokens, completionTokens int) float64 {
	return float64(promptTokens)*m.PromptPricePerMillion/1e6 + float64(completionTokens)*m.CompletionPricePerMillion/1e6
}

type modelsResponse struct {
	Data []struct {
		ID                  string   `json:"id"`
		Name                string   `json:"name"`
		Creator             string   `json:"creator"`
		ContextLen          int      `json:"context_length"`
		SupportedParameters []string `json:"supported_parameters"`
		TopProvider         struct {
			MaxCompletionTokens int `json:"max_completion_tokens"`
		} `json:"top_provider"`
		Pricing struct {
			Global struct {
				Prompt      float64 `json:"prompt"`
				Completions float64 `json:"completions"`
			} `json:"global"`
		} `json:"pricing"`
	} `json:"data"`
}

// ListModels calls the public, unauthenticated GET /v1/models. It does not go
// through the openai-go SDK — unlike chat completions, this route is a
// gateway-specific extension with its own response shape, not part of the
// surface the SDK models — so it's a small direct HTTP call instead.
func (g *Gateway) ListModels(ctx context.Context) ([]ModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, g.baseURL+"/v1/models", nil)
	if err != nil {
		return nil, err
	}

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET /v1/models: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("GET /v1/models: reading body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET /v1/models: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var parsed modelsResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("GET /v1/models: parsing response: %w", err)
	}

	models := make([]ModelInfo, 0, len(parsed.Data))

	for _, m := range parsed.Data {
		if m.ID == "" {
			continue
		}

		models = append(models, ModelInfo{
			ID:                        m.ID,
			Name:                      m.Name,
			Creator:                   m.Creator,
			ContextLength:             m.ContextLen,
			MaxCompletionTokens:       m.TopProvider.MaxCompletionTokens,
			PromptPricePerMillion:     m.Pricing.Global.Prompt,
			CompletionPricePerMillion: m.Pricing.Global.Completions,
			SupportedParameters:       m.SupportedParameters,
		})
	}

	return models, nil
}

// CompletionResult is the raw measurement from one synchronous completion
// call.
type CompletionResult struct {
	Text string
	// RawRequest is the verbatim JSON body sent to the gateway — the same
	// bytes ChatCompletionNewParams.MarshalJSON produces and the SDK puts on
	// the wire (see the comment inline in RunCompletion), not this tool's own
	// reconstruction of it. Set even when the call fails, so a failed sample
	// can be replayed exactly (e.g. via curl) to debug or reproduce it.
	RawRequest string
	// RawResponse is the verbatim JSON response body the gateway returned —
	// the whole ChatCompletion object (id, model, choices with their
	// messages, usage, ...), not just the extracted Text — kept so a failed
	// sample (e.g. information_extraction output that fails schema
	// validation) can be saved and inspected exactly as the model actually
	// returned it, rather than only the one field this tool happened to
	// extract from it. Empty when the call failed before producing one.
	RawResponse      string
	TotalLatency     time.Duration
	TimeToFirstToken time.Duration
	PromptTokens     int
	CompletionTokens int
}

// RunCompletion sends systemPrompt + userMessage to model as a single
// synchronous (non-streaming) HTTP request and measures wall-clock latency
// directly off the wire: TotalLatency starts immediately before the request
// is sent and ends when the full response has been read — what a caller
// waiting on the complete reply actually experiences.
//
// TimeToFirstToken is always equal to TotalLatency. A synchronous call has no
// observable "first token" moment distinct from "the whole response
// arrived" — the client reads one response body, not a sequence of chunks,
// so there is nothing earlier than the complete completion to time against.
// The field is kept, rather than dropped, so Sample/Stats/the report and the
// JSON report schema don't need a special case for "not measurable here"
// versus "measured as zero"; a reader comparing the TTFT and latency columns
// for any row can see for themselves that they're identical, and why.
//
// model.ReasoningEffort is sent as the request's reasoning_effort when set,
// and left off the wire entirely when nil -- model carries it (rather than a
// separate parameter here) so every call site already holding a ModelInfo
// gets this for free, and applyReasoningEffort (voicebench.go) is the only
// place that ever sets it, having already confirmed the model advertises
// support for it.
func (g *Gateway) RunCompletion(ctx context.Context, model ModelInfo, systemPrompt, userMessage string, maxTokens int) (CompletionResult, error) {
	ctx, cancel := context.WithTimeout(ctx, g.timeout)
	defer cancel()

	params := openai.ChatCompletionNewParams{
		Model: model.ID,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(userMessage),
		},
		MaxTokens: openai.Int(int64(maxTokens)),
	}

	if model.ReasoningEffort != nil {
		params.ReasoningEffort = openai.ReasoningEffort(*model.ReasoningEffort)
	}

	// Marshaled independently of the actual request the SDK sends, but from
	// the same params value and via the same ChatCompletionNewParams.MarshalJSON
	// the SDK itself calls to build the wire body (see NewRequestConfig in
	// openai-go's internal/requestconfig) — so this is byte-for-byte what was
	// actually sent, not a hand-rolled reconstruction that could drift from it.
	rawRequest, marshalErr := json.Marshal(params)
	if marshalErr != nil {
		// params is built entirely from this function's own string/int
		// arguments — nothing here can fail to marshal in practice. Treat it
		// as "nothing to show" rather than failing the whole completion call
		// over a request body this tool doesn't even use to build the value
		// actually sent.
		rawRequest = nil
	}

	start := time.Now()

	resp, err := g.client.Chat.Completions.New(ctx, params)

	total := time.Since(start)

	if err != nil {
		return CompletionResult{RawRequest: string(rawRequest)}, fmt.Errorf("model %s: %w", model.ID, err)
	}

	if len(resp.Choices) == 0 {
		return CompletionResult{RawRequest: string(rawRequest)}, fmt.Errorf("model %s: response contained no choices", model.ID)
	}

	return CompletionResult{
		Text:             resp.Choices[0].Message.Content,
		RawRequest:       string(rawRequest),
		RawResponse:      resp.RawJSON(),
		TotalLatency:     total,
		TimeToFirstToken: total,
		PromptTokens:     int(resp.Usage.PromptTokens),
		CompletionTokens: int(resp.Usage.CompletionTokens),
	}, nil
}

// StreamResult is the raw measurement from one streaming completion call —
// the counterpart to CompletionResult for RunCompletionStream. Unlike
// CompletionResult, TimeToFirstToken here is a REAL, independently-observed
// measurement, not a copy of the total latency (see RunCompletionStream).
type StreamResult struct {
	Text string
	// RawRequest is the verbatim JSON body sent to the gateway, the same way
	// CompletionResult.RawRequest is — see RunCompletionStream. There is no
	// RawResponse: a streamed reply is a sequence of SSE chunks, not one JSON
	// response body, so there is no single verbatim document to save the way
	// a synchronous call's RawResponse captures one.
	RawRequest       string
	TimeToFirstToken time.Duration
	TotalLatency     time.Duration
	PromptTokens     int
	CompletionTokens int
}

// RunCompletionStream sends systemPrompt + userMessage to model as a
// streaming HTTP request (POST /v1/chat/completions with stream: true) and
// measures TimeToFirstToken as the actual wall-clock gap between sending the
// request and the first chunk carrying non-empty delta content — genuinely
// earlier than TotalLatency (the gap to the stream's end), unlike
// RunCompletion's synchronous call, which has no earlier moment to measure
// against (see its doc comment). This exists solely for the TTFT-only
// benchmark path (see ttft.go); the standard quality/latency benchmark keeps
// using RunCompletion so its own numbers stay exactly as they've always
// been.
//
// stream_options.include_usage is set so the final chunk carries prompt/
// completion token counts — otherwise, unlike a synchronous ChatCompletion,
// a stream reports no usage at all.
//
// Like RunCompletion, model.ReasoningEffort is sent as reasoning_effort when
// set and left off the wire when nil -- see RunCompletion's doc comment.
func (g *Gateway) RunCompletionStream(ctx context.Context, model ModelInfo, systemPrompt, userMessage string, maxTokens int) (StreamResult, error) {
	ctx, cancel := context.WithTimeout(ctx, g.timeout)
	defer cancel()

	params := openai.ChatCompletionNewParams{
		Model: model.ID,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(userMessage),
		},
		MaxTokens: openai.Int(int64(maxTokens)),
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}

	if model.ReasoningEffort != nil {
		params.ReasoningEffort = openai.ReasoningEffort(*model.ReasoningEffort)
	}

	rawRequest, marshalErr := json.Marshal(params)
	if marshalErr != nil {
		rawRequest = nil
	}

	start := time.Now()

	stream := g.client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	var (
		text    strings.Builder
		ttft    time.Duration
		ttftSet bool
		usage   openai.CompletionUsage
	)

	for stream.Next() {
		chunk := stream.Current()

		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			if !ttftSet {
				ttft = time.Since(start)
				ttftSet = true
			}

			text.WriteString(chunk.Choices[0].Delta.Content)
		}

		// The usage chunk (stream_options.include_usage) carries an empty
		// Choices slice and is not guaranteed to be the LAST chunk relative to
		// this loop's exit, but it IS the only chunk with non-zero usage, so
		// capturing it unconditionally here is equivalent to special-casing
		// "the final chunk" without needing to know which one that is ahead of
		// time.
		if chunk.Usage.TotalTokens > 0 {
			usage = chunk.Usage
		}
	}

	total := time.Since(start)

	if err := stream.Err(); err != nil {
		return StreamResult{RawRequest: string(rawRequest), TotalLatency: total}, fmt.Errorf("model %s: %w", model.ID, err)
	}

	if !ttftSet {
		return StreamResult{RawRequest: string(rawRequest), TotalLatency: total}, fmt.Errorf("model %s: stream produced no content", model.ID)
	}

	return StreamResult{
		Text:             text.String(),
		RawRequest:       string(rawRequest),
		TimeToFirstToken: ttft,
		TotalLatency:     total,
		PromptTokens:     int(usage.PromptTokens),
		CompletionTokens: int(usage.CompletionTokens),
	}, nil
}

// RawAPIError returns the verbatim JSON body of the gateway's error
// response, if err actually came from one — i.e. the closest thing to "the
// LLM's response" a failed completion call still has. openai-go returns a
// *openai.Error (an alias for the SDK's internal apierror.Error — see
// NewGateway's doc comment on when the gateway returns this shape) for any
// non-2xx response; errors.As unwraps RunCompletion's fmt.Errorf("model %s:
// %w", ...) wrapping to find it. Returns "" for any other kind of failure —
// a timeout, a canceled context, a network error that never got a response
// at all — where there genuinely is no response body to save.
func RawAPIError(err error) string {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return apiErr.RawJSON()
	}

	return ""
}
