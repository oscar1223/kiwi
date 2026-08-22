// Package anthropic adapts the official Anthropic SDK to llm.Provider.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/oscar1223/kiwi/internal/llm"
)

const defaultMaxTokens = 8192

type Provider struct {
	client sdk.Client
	model  string
}

type Options struct {
	APIKey  string
	BaseURL string
	Model   string
}

func New(opts Options) *Provider {
	reqOpts := []option.RequestOption{}
	if opts.APIKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(opts.APIKey))
	}
	if opts.BaseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(opts.BaseURL))
	}
	return &Provider{client: sdk.NewClient(reqOpts...), model: opts.Model}
}

func (p *Provider) Name() string  { return "anthropic" }
func (p *Provider) Model() string { return p.model }

func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		params, err := p.params(req)
		if err != nil {
			yield(llm.Event{}, err)
			return
		}

		stream := p.client.Messages.NewStreaming(ctx, params)
		defer stream.Close()

		var acc sdk.Message
		for stream.Next() {
			event := stream.Current()
			if err := acc.Accumulate(event); err != nil {
				yield(llm.Event{}, fmt.Errorf("anthropic: accumulate: %w", err))
				return
			}
			// Only prose deltas are forwarded live. Tool input arrives as
			// partial JSON and is worthless until assembled, so it is emitted
			// once, complete, at the end.
			if delta := event.AsContentBlockDelta(); delta.Type == "content_block_delta" {
				if text := delta.Delta.Text; text != "" {
					if !yield(llm.Event{Type: llm.EventTextDelta, Text: text}, nil) {
						return
					}
				}
			}
		}
		if err := stream.Err(); err != nil {
			yield(llm.Event{}, fmt.Errorf("anthropic: stream: %w", err))
			return
		}

		msg := llm.Message{Role: llm.RoleAssistant}
		for _, block := range acc.Content {
			switch block.Type {
			case "text":
				msg.Content += block.Text
			case "tool_use":
				call := llm.ToolCall{ID: block.ID, Name: block.Name, Input: block.Input}
				msg.ToolCalls = append(msg.ToolCalls, call)
				if !yield(llm.Event{Type: llm.EventToolCall, ToolCall: &call}, nil) {
					return
				}
			}
		}

		yield(llm.Event{
			Type:    llm.EventDone,
			Message: &msg,
			Usage: &llm.Usage{
				InputTokens:  int(acc.Usage.InputTokens),
				OutputTokens: int(acc.Usage.OutputTokens),
			},
		}, nil)
	}
}

func (p *Provider) params(req llm.Request) (sdk.MessageNewParams, error) {
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	msgs, err := toMessages(req.Messages)
	if err != nil {
		return sdk.MessageNewParams{}, err
	}

	params := sdk.MessageNewParams{
		Model:     sdk.Model(p.model),
		MaxTokens: int64(maxTokens),
		Messages:  msgs,
	}
	if req.System != "" {
		params.System = []sdk.TextBlockParam{{Text: req.System}}
	}
	for _, t := range req.Tools {
		schema := sdk.ToolInputSchemaParam{}
		if props, ok := t.Schema["properties"]; ok {
			schema.Properties = props
		}
		if reqd, ok := t.Schema["required"].([]string); ok {
			schema.Required = reqd
		}
		tool := sdk.ToolUnionParamOfTool(schema, t.Name)
		tool.OfTool.Description = sdk.String(t.Description)
		params.Tools = append(params.Tools, tool)
	}
	return params, nil
}

// toMessages converts Kiwi history to Anthropic's format.
//
// The subtlety: Anthropic expects every tool_result for one assistant turn to
// arrive in a *single* user message. Sending one message per result is
// rejected, so consecutive tool messages are coalesced here.
func toMessages(in []llm.Message) ([]sdk.MessageParam, error) {
	var out []sdk.MessageParam
	var pendingResults []sdk.ContentBlockParamUnion

	flush := func() {
		if len(pendingResults) > 0 {
			out = append(out, sdk.NewUserMessage(pendingResults...))
			pendingResults = nil
		}
	}

	for _, m := range in {
		switch m.Role {
		case llm.RoleTool:
			pendingResults = append(pendingResults,
				sdk.NewToolResultBlock(m.ToolCallID, m.Content, m.IsError))

		case llm.RoleUser:
			flush()
			out = append(out, sdk.NewUserMessage(sdk.NewTextBlock(m.Content)))

		case llm.RoleAssistant:
			flush()
			var blocks []sdk.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, sdk.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var input any
				if len(tc.Input) > 0 {
					if err := json.Unmarshal(tc.Input, &input); err != nil {
						return nil, fmt.Errorf("anthropic: tool call %s input: %w", tc.ID, err)
					}
				}
				blocks = append(blocks, sdk.NewToolUseBlock(tc.ID, input, tc.Name))
			}
			if len(blocks) > 0 {
				out = append(out, sdk.NewAssistantMessage(blocks...))
			}

		default:
			return nil, fmt.Errorf("anthropic: unsupported role %q", m.Role)
		}
	}
	flush()
	return out, nil
}
