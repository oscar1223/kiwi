// Package telemetry sends Kiwi's turns, model calls and tool calls to an
// OpenTelemetry backend, when one is configured.
//
// There is no official Langfuse SDK for Go, so this goes through the OTLP
// exporter Langfuse itself recommends for languages it doesn't ship a native
// client for. That also means kiwi is not actually coupled to Langfuse: any
// OTel-compatible backend (Honeycomb, Grafana Tempo, a local collector) works
// the same way, through the same standard OTEL_EXPORTER_OTLP_* variables.
//
// With nothing configured, tracing is entirely inert: Configure is never
// called, the package-level tracer stays bound to the OTel SDK's default
// no-op provider, and every Start* function below returns a span that costs
// next to nothing and sends data nowhere. Nothing in kiwi depends on tracing
// being active.
package telemetry

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// tracerName identifies kiwi's spans as one instrumentation scope.
//
// otel.Tracer(tracerName) is called fresh at each span start rather than
// cached in a package variable: OTel's global delegate is documented to
// support exactly one cutover from the no-op default to a real provider, not
// repeated swaps — a cached Tracer obtained before the first SetTracerProvider
// call stays bound to whichever provider was active at that first cutover.
// Asking again each time is cheap (a map lookup) and is what actually stays
// correct if the provider is ever reconfigured.
const tracerName = "kiwi"

func tracer() trace.Tracer { return otel.Tracer(tracerName) }

// Configure wires the global TracerProvider to an OTLP/HTTP exporter when the
// environment describes one, and returns a shutdown func that flushes
// pending spans — call it before the process exits.
//
// Two ways to configure it, checked in order:
//
//  1. Generic OTel variables (OTEL_EXPORTER_OTLP_ENDPOINT or
//     OTEL_EXPORTER_OTLP_TRACES_ENDPOINT): kiwi defers entirely to the
//     standard SDK's own env handling, so any OTel-compatible backend works.
//  2. LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY: translated into Langfuse's
//     OTLP ingestion endpoint and the Basic Auth header it expects.
//     LANGFUSE_HOST selects a self-hosted instance; unset, it defaults to
//     Langfuse Cloud, matching every other Langfuse SDK's own default.
//
// Neither present: Configure returns a no-op shutdown and leaves tracing off.
// It never fails startup over a telemetry backend being unreachable — that
// only shows up later, as export failures written to errorLog.
//
// errorLog receives one line per failed export (e.g. the backend being
// unreachable, wrong credentials). It must not be the terminal: the OTel SDK
// retries failed batches on its own schedule from a background goroutine, and
// without a handler installed here it logs straight to os.Stderr by default —
// which corrupts the TUI's display, since Bubble Tea assumes exclusive
// control of the terminal and has no way to know a write landed mid-frame. A
// nil errorLog drops these silently rather than risking that.
func Configure(ctx context.Context, errorLog io.Writer, extraResource ...attribute.KeyValue) (shutdown func(context.Context) error, err error) {
	noop := func(context.Context) error { return nil }

	// Installed unconditionally, before anything else: the default handler
	// writing to stderr is the actual bug being avoided, and that risk exists
	// the moment any OTel code path might call otel.Handle, not only once
	// exporting is confirmed enabled below.
	otel.SetErrorHandler(otel.ErrorHandlerFunc(func(err error) {
		if errorLog == nil {
			return
		}
		fmt.Fprintf(errorLog, "%s telemetry: %v\n", time.Now().Format(time.RFC3339), err)
	}))

	endpointURL, headers, enabled := resolveEndpoint()
	if !enabled {
		return noop, nil
	}

	opts := []otlptracehttp.Option{otlptracehttp.WithHeaders(headers)}
	if endpointURL != "" {
		opts = append(opts, otlptracehttp.WithEndpointURL(endpointURL))
	}
	exporter, err := otlptracehttp.New(ctx, opts...)
	if err != nil {
		return noop, fmt.Errorf("telemetry: creating exporter: %w", err)
	}

	res, err := resource.Merge(resource.Default(), resource.NewSchemaless(
		append([]attribute.KeyValue{attribute.String("service.name", "kiwi")}, extraResource...)...,
	))
	if err != nil {
		return noop, fmt.Errorf("telemetry: building resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter), sdktrace.WithResource(res))
	otel.SetTracerProvider(tp)
	return tp.Shutdown, nil
}

// resolveEndpoint decides where spans go, if anywhere. ok is false when
// nothing in the environment asked for tracing.
func resolveEndpoint() (endpointURL string, headers map[string]string, ok bool) {
	if os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT") != "" || os.Getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") != "" {
		// Let the SDK read the standard variables itself; passing no
		// endpoint option here is what makes that happen.
		return "", nil, true
	}

	public := os.Getenv("LANGFUSE_PUBLIC_KEY")
	secret := os.Getenv("LANGFUSE_SECRET_KEY")
	if public == "" || secret == "" {
		return "", nil, false
	}

	host := os.Getenv("LANGFUSE_HOST")
	if host == "" {
		host = "https://cloud.langfuse.com"
	}
	auth := base64.StdEncoding.EncodeToString([]byte(public + ":" + secret))
	return strings.TrimRight(host, "/") + "/api/public/otel/v1/traces", map[string]string{
		"Authorization": "Basic " + auth,
		// Without this header Langfuse queues directly-ingested OTel data
		// for its legacy pipeline, which can delay it up to ten minutes.
		"x-langfuse-ingestion-version": "4",
	}, true
}

// --- request-scoped identity ---
//
// Session and user id ride on ctx rather than threading through every
// function signature down to where a span actually starts — the same reason
// a request id or deadline does. Session id draws in the internal/session
// package's persisted sessions; user id defaults to the OS username, which is
// enough to tell one Kiwi user's traces apart from another's on a shared
// self-hosted Langfuse instance.

type ctxKey int

const (
	sessionKey ctxKey = iota
	userKey
)

func WithSessionID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, sessionKey, id)
}

func sessionID(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(sessionKey).(string)
	return id, ok && id != ""
}

func WithUserID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, userKey, id)
}

func userID(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(userKey).(string)
	return id, ok && id != ""
}

// --- spans ---
//
// Attribute names follow OpenTelemetry's GenAI semantic conventions
// (gen_ai.*) where one exists, so a generation renders with model and token
// data in Langfuse — and in any other OTel backend that understands the same
// convention. langfuse.* attributes are Langfuse-specific extensions for
// session/user grouping that Langfuse's own docs say take precedence over
// the generic ones.

// StartTurn starts the root span for one agent turn.
func StartTurn(ctx context.Context, providerName, model string) (context.Context, trace.Span) {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.system", providerName),
		attribute.String("gen_ai.request.model", model),
		attribute.StringSlice("langfuse.tags", []string{"kiwi"}),
	}
	if id, ok := sessionID(ctx); ok {
		attrs = append(attrs, attribute.String("langfuse.session.id", id))
	}
	if id, ok := userID(ctx); ok {
		attrs = append(attrs, attribute.String("langfuse.user.id", id))
	}
	return tracer().Start(ctx, "kiwi.turn", trace.WithAttributes(attrs...))
}

func EndTurn(span trace.Span, err error) { end(span, err) }

// StartModelCall starts a child span for one provider round trip within a
// turn — a turn with three tool calls makes three of these.
func StartModelCall(ctx context.Context, model string) (context.Context, trace.Span) {
	return tracer().Start(ctx, "chat "+model, trace.WithAttributes(
		attribute.String("gen_ai.request.model", model),
	))
}

// EndModelCall records token usage before ending a model-call span.
func EndModelCall(span trace.Span, inputTokens, outputTokens int, err error) {
	span.SetAttributes(
		attribute.Int("gen_ai.usage.input_tokens", inputTokens),
		attribute.Int("gen_ai.usage.output_tokens", outputTokens),
	)
	end(span, err)
}

// StartTool starts a child span for one tool invocation.
func StartTool(ctx context.Context, name string) (context.Context, trace.Span) {
	return tracer().Start(ctx, "execute_tool "+name, trace.WithAttributes(
		attribute.String("gen_ai.tool.name", name),
	))
}

// EndTool ends a tool span, flagging isErr as a span-level error without
// treating the tool's own recoverable failure as a reason to alarm on it the
// same way an unexpected exception would be.
func EndTool(span trace.Span, isErr bool) {
	if isErr {
		span.SetStatus(codes.Error, "tool returned an error")
	}
	span.End()
}

func end(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	span.End()
}
