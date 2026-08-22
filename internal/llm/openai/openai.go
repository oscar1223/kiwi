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
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"github.com/oscar1223/kiwi/internal/llm"
)

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

func New(opts Options) *Provider {
	reqOpts := []option.RequestOption{}
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
