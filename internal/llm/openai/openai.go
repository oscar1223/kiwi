// Package openai adapts the official OpenAI SDK to llm.Provider.
//
// It targets the chat-completions wire format rather than OpenAI specifically,
// so the same adapter serves Ollama, LM Studio, OpenRouter, Groq and anything
// else that speaks it — which is what keeps Kiwi usable fully offline.
package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"github.com/oscar1223/kiwi/internal/llm"
)

// defaultAzureAPIVersion is used when a profile doesn't specify one. Azure
// OpenAI's API version is a required query parameter, unlike plain OpenAI
// where the wire format has no such versioning.
const defaultAzureAPIVersion = "2024-10-21"

type Provider struct {
	client sdk.Client
	model  string
	name   string
}

type Options struct {
	APIKey  string
	BaseURL string
	Model   string
	// Name labels the provider in the UI (e.g. "ollama", "openrouter").
	Name string
}

// defaultMaxRetries raises the SDK's default of 2. Kiwi's tool loop can burst
// several requests in quick succession (parallel subagents, retried turns),
// which trips provider rate limits well within 2 attempts; the SDK's own
// backoff already respects Retry-After, so widening the budget is enough.
const defaultMaxRetries = 5

func New(opts Options) *Provider {
	reqOpts := []option.RequestOption{option.WithMaxRetries(defaultMaxRetries)}
	if opts.APIKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
	}
	if opts.BaseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(opts.BaseURL))
	}
	name := opts.Name
	if name == "" {
		name = "openai"
	}
	return &Provider{client: sdk.NewClient(reqOpts...), model: opts.Model, name: name}
}

type AzureOptions struct {
	// Resource is the Azure OpenAI resource name, i.e. the "foo" in
	// https://foo.openai.azure.com.
	Resource string
	APIKey   string
	// Deployment is the Azure deployment name. Azure routes by deployment,
	// not by a "model" field in the request body — the SDK's endpoint
	// middleware rewrites the URL path for this, so Deployment still just
	// flows into Provider.model the same way Options.Model does above.
	Deployment string
	// APIVersion defaults to defaultAzureAPIVersion when empty.
	APIVersion string
}

// NewAzure builds a Provider for an Azure OpenAI deployment. Azure's wire
// format is otherwise identical to plain OpenAI chat-completions — only the
// URL shape (resource + deployment + api-version) and the auth header
// differ, both handled by the SDK's azure subpackage.
func NewAzure(opts AzureOptions) *Provider {
	apiVersion := opts.APIVersion
	if apiVersion == "" {
		apiVersion = defaultAzureAPIVersion
	}
	endpoint := fmt.Sprintf("https://%s.openai.azure.com", opts.Resource)
	client := sdk.NewClient(
		option.WithMaxRetries(defaultMaxRetries),
		azure.WithEndpoint(endpoint, apiVersion),
		azure.WithAPIKey(opts.APIKey),
	)
	return &Provider{client: client, model: opts.Deployment, name: "azure"}
}

func (p *Provider) Name() string  { return p.name }
func (p *Provider) Model() string { return p.model }

func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		params, err := p.params(req)
		if err != nil {
			yield(llm.Event{}, err)
			return
		}

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)
		defer stream.Close()

		var acc sdk.ChatCompletionAccumulator
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)
			if len(chunk.Choices) > 0 {
				if text := chunk.Choices[0].Delta.Content; text != "" {
					if !yield(llm.Event{Type: llm.EventTextDelta, Text: text}, nil) {
						return
					}
				}
			}
		}
		if err := stream.Err(); err != nil {
			yield(llm.Event{}, fmt.Errorf("%s: stream: %w", p.name, err))
			return
		}
		if len(acc.Choices) == 0 {
			yield(llm.Event{}, fmt.Errorf("%s: response had no choices", p.name))
			return
		}

		choice := acc.Choices[0].Message
		msg := llm.Message{Role: llm.RoleAssistant, Content: choice.Content}
		for _, tc := range choice.ToolCalls {
			call := llm.ToolCall{
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			}
			// Some OpenAI-compatible servers send "" for a no-argument call;
			// downstream code expects valid JSON.
			if len(call.Input) == 0 {
				call.Input = json.RawMessage("{}")
			}
			msg.ToolCalls = append(msg.ToolCalls, call)
			if !yield(llm.Event{Type: llm.EventToolCall, ToolCall: &call}, nil) {
				return
			}
		}

		yield(llm.Event{
			Type:    llm.EventDone,
			Message: &msg,
			Usage: &llm.Usage{
				InputTokens:  int(acc.Usage.PromptTokens),
				OutputTokens: int(acc.Usage.CompletionTokens),
			},
		}, nil)
	}
}

func (p *Provider) params(req llm.Request) (sdk.ChatCompletionNewParams, error) {
	msgs := make([]sdk.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, sdk.SystemMessage(req.System))
	}

	for _, m := range req.Messages {
		switch m.Role {
		case llm.RoleUser:
			msgs = append(msgs, sdk.UserMessage(m.Content))

		case llm.RoleTool:
			msgs = append(msgs, sdk.ToolMessage(m.Content, m.ToolCallID))

		case llm.RoleAssistant:
			am := sdk.ChatCompletionAssistantMessageParam{}
			if m.Content != "" {
				am.Content.OfString = sdk.String(m.Content)
			}
			for _, tc := range m.ToolCalls {
				args := string(tc.Input)
				if args == "" {
					args = "{}"
				}
				am.ToolCalls = append(am.ToolCalls, sdk.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &sdk.ChatCompletionMessageFunctionToolCallParam{
						ID: tc.ID,
						Function: sdk.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      tc.Name,
							Arguments: args,
						},
					},
				})
			}
			msgs = append(msgs, sdk.ChatCompletionMessageParamUnion{OfAssistant: &am})

		default:
			return sdk.ChatCompletionNewParams{}, fmt.Errorf("%s: unsupported role %q", p.name, m.Role)
		}
	}

	params := sdk.ChatCompletionNewParams{
		Model:    sdk.ChatModel(p.model),
		Messages: msgs,
	}
	if req.MaxTokens > 0 {
		params.MaxTokens = sdk.Int(int64(req.MaxTokens))
	}
	if req.Temperature != nil {
		params.Temperature = sdk.Float(*req.Temperature)
	}
	for _, t := range req.Tools {
		params.Tools = append(params.Tools, sdk.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        t.Name,
			Description: sdk.String(t.Description),
			Parameters:  shared.FunctionParameters(t.Schema),
		}))
	}
	return params, nil
}
