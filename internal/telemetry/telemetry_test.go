package telemetry

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"os"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// withRecorder points the package's global tracer at an in-memory exporter
// for the duration of a test, so span creation can be asserted on directly
// without a network call — no fake server, no OTLP wire format to fake.
//
// It deliberately does not restore whatever provider was active before: the
// OTel API's global delegate is documented to support exactly one cutover
// from the no-op default to a real provider, made once at process startup —
// not repeated swaps back to an earlier placeholder mid-process. Tests below
// run sequentially in this package's own process and nothing else reads the
// global provider, so each just installs its own real one and moves on.
func withRecorder(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { tp.Shutdown(context.Background()) })
	return exp
}

func TestStartTurnRecordsProviderAndModel(t *testing.T) {
	exp := withRecorder(t)

	_, span := StartTurn(context.Background(), "anthropic", "claude-opus-5")
	span.End()

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	attrs := attrMap(spans[0].Attributes)
	if attrs["gen_ai.system"] != "anthropic" {
		t.Errorf("gen_ai.system = %v", attrs["gen_ai.system"])
	}
	if attrs["gen_ai.request.model"] != "claude-opus-5" {
		t.Errorf("gen_ai.request.model = %v", attrs["gen_ai.request.model"])
	}
}

// Session and user id must reach the span only when the caller actually put
// them on ctx — a turn with neither must not emit empty langfuse.* attributes
// that would show up as blank fields in Langfuse's UI.
func TestStartTurnPropagatesIdentityFromContext(t *testing.T) {
	exp := withRecorder(t)

	ctx := WithSessionID(context.Background(), "sess-123")
	ctx = WithUserID(ctx, "racso")
	_, span := StartTurn(ctx, "openai", "gpt-5.5")
	span.End()

	attrs := attrMap(exp.GetSpans()[0].Attributes)
	if attrs["langfuse.session.id"] != "sess-123" {
		t.Errorf("langfuse.session.id = %v", attrs["langfuse.session.id"])
	}
	if attrs["langfuse.user.id"] != "racso" {
		t.Errorf("langfuse.user.id = %v", attrs["langfuse.user.id"])
	}
}

func TestStartTurnOmitsIdentityWhenAbsent(t *testing.T) {
	exp := withRecorder(t)

	_, span := StartTurn(context.Background(), "openai", "gpt-5.5")
	span.End()

	attrs := attrMap(exp.GetSpans()[0].Attributes)
	if _, ok := attrs["langfuse.session.id"]; ok {
		t.Error("langfuse.session.id present despite no session on ctx")
	}
	if _, ok := attrs["langfuse.user.id"]; ok {
		t.Error("langfuse.user.id present despite no user on ctx")
	}
}

// A tool span started from the ctx StartTurn returned must nest under the
// turn span — this is what makes one turn render as one trace in Langfuse
// instead of a pile of disconnected spans.
func TestToolSpanNestsUnderTurnSpan(t *testing.T) {
	exp := withRecorder(t)

	ctx, turnSpan := StartTurn(context.Background(), "openai", "gpt-5.5")
	_, toolSpan := StartTool(ctx, "bash")
	toolSpan.End()
	turnSpan.End()

	spans := exp.GetSpans()
	if len(spans) != 2 {
		t.Fatalf("got %d spans, want 2", len(spans))
	}
	var tool, turn tracetest.SpanStub
	for _, s := range spans {
		if s.Name == "execute_tool bash" {
			tool = s
		} else if s.Name == "kiwi.turn" {
			turn = s
		}
	}
	if tool.Name == "" || turn.Name == "" {
		t.Fatalf("did not find both spans by name: %+v", spans)
	}
	if tool.Parent.SpanID() != turn.SpanContext.SpanID() {
		t.Error("tool span is not a child of the turn span")
	}
	if tool.SpanContext.TraceID() != turn.SpanContext.TraceID() {
		t.Error("tool span is on a different trace than its turn")
	}
}

func TestEndToolMarksErrorStatus(t *testing.T) {
	exp := withRecorder(t)

	_, span := StartTool(context.Background(), "bash")
	EndTool(span, true)

	got := exp.GetSpans()[0]
	if got.Status.Code != codes.Error {
		t.Errorf("status = %v, want Error", got.Status.Code)
	}
}

func TestEndToolSuccessLeavesStatusUnset(t *testing.T) {
	exp := withRecorder(t)

	_, span := StartTool(context.Background(), "bash")
	EndTool(span, false)

	got := exp.GetSpans()[0]
	if got.Status.Code == codes.Error {
		t.Error("a successful tool call was marked as an error")
	}
}

func TestEndModelCallRecordsUsage(t *testing.T) {
	exp := withRecorder(t)

	_, span := StartModelCall(context.Background(), "gpt-5.5")
	EndModelCall(span, 120, 45, nil)

	attrs := attrMap(exp.GetSpans()[0].Attributes)
	if v, ok := attrs["gen_ai.usage.input_tokens"].(int64); !ok || v != 120 {
		t.Errorf("gen_ai.usage.input_tokens = %v", attrs["gen_ai.usage.input_tokens"])
	}
	if v, ok := attrs["gen_ai.usage.output_tokens"].(int64); !ok || v != 45 {
		t.Errorf("gen_ai.usage.output_tokens = %v", attrs["gen_ai.usage.output_tokens"])
	}
}

func TestEndTurnRecordsError(t *testing.T) {
	exp := withRecorder(t)

	_, span := StartTurn(context.Background(), "openai", "gpt-5.5")
	EndTurn(span, errors.New("boom"))

	got := exp.GetSpans()[0]
	if got.Status.Code != codes.Error {
		t.Errorf("status = %v, want Error", got.Status.Code)
	}
	if got.Status.Description != "boom" {
		t.Errorf("status description = %q", got.Status.Description)
	}
	if len(got.Events) == 0 {
		t.Error("RecordError should have added an exception event")
	}
}

func TestEndTurnSuccessHasNoErrorStatus(t *testing.T) {
	exp := withRecorder(t)

	_, span := StartTurn(context.Background(), "openai", "gpt-5.5")
	EndTurn(span, nil)

	got := exp.GetSpans()[0]
	if got.Status.Code == codes.Error {
		t.Error("a successful turn was marked as an error")
	}
}

func TestSessionAndUserIDRoundTrip(t *testing.T) {
	ctx := context.Background()
	if _, ok := sessionID(ctx); ok {
		t.Error("sessionID should be absent from a bare context")
	}

	ctx = WithSessionID(ctx, "s1")
	got, ok := sessionID(ctx)
	if !ok || got != "s1" {
		t.Errorf("sessionID = (%q, %v)", got, ok)
	}

	ctx = WithUserID(ctx, "u1")
	got, ok = userID(ctx)
	if !ok || got != "u1" {
		t.Errorf("userID = (%q, %v)", got, ok)
	}
}

func TestWithSessionIDEmptyStringDoesNotCount(t *testing.T) {
	ctx := WithSessionID(context.Background(), "")
	if _, ok := sessionID(ctx); ok {
		t.Error("an empty session id should not read back as present")
	}
}

func attrMap(kvs []attribute.KeyValue) map[string]any {
	m := make(map[string]any, len(kvs))
	for _, kv := range kvs {
		switch kv.Value.Type() {
		case attribute.STRING:
			m[string(kv.Key)] = kv.Value.AsString()
		case attribute.INT64:
			m[string(kv.Key)] = kv.Value.AsInt64()
		case attribute.STRINGSLICE:
			m[string(kv.Key)] = kv.Value.AsStringSlice()
		default:
			m[string(kv.Key)] = kv.Value.AsInterface()
		}
	}
	return m
}

// Regression test: an earlier version cached the Tracer in a package
// variable at init time. OTel's global delegate only supports one cutover
// from the no-op default to a real provider, so that cached Tracer stayed
// bound to whichever provider was active at the *first* SetTracerProvider
// call, and every span from a later reconfiguration silently went nowhere.
// This exercises exactly that sequence: two providers, back to back.
func TestSpansFollowASecondProviderSwap(t *testing.T) {
	first := tracetest.NewInMemoryExporter()
	tp1 := sdktrace.NewTracerProvider(sdktrace.WithSyncer(first))
	otel.SetTracerProvider(tp1)
	_, span := StartTurn(context.Background(), "openai", "gpt-5.5")
	span.End()
	// Not shutting tp1 down here: InMemoryExporter.Shutdown calls Reset,
	// which would wipe the very spans this test wants to inspect below.

	second := tracetest.NewInMemoryExporter()
	tp2 := sdktrace.NewTracerProvider(sdktrace.WithSyncer(second))
	otel.SetTracerProvider(tp2)
	t.Cleanup(func() { tp2.Shutdown(context.Background()) })

	_, span = StartTurn(context.Background(), "anthropic", "claude-opus-5")
	span.End()

	if len(first.GetSpans()) != 1 {
		t.Errorf("first provider recorded %d spans, want 1", len(first.GetSpans()))
	}
	if len(second.GetSpans()) != 1 {
		t.Fatalf("second provider recorded %d spans, want 1 — spans stopped following provider reconfiguration", len(second.GetSpans()))
	}
	if second.GetSpans()[0].Name != "kiwi.turn" {
		t.Errorf("second provider's span = %+v", second.GetSpans()[0])
	}
}

func clearTelemetryEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
		"LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
	} {
		t.Setenv(k, "")
		os.Unsetenv(k)
	}
}

func TestResolveEndpointNothingConfigured(t *testing.T) {
	clearTelemetryEnv(t)

	_, _, ok := resolveEndpoint()
	if ok {
		t.Error("resolveEndpoint should report disabled when nothing is set")
	}
}

// A generic OTel setup must defer entirely to the SDK's own env handling —
// kiwi should not build an endpoint URL itself in this case, or it would
// override whatever the user already configured for their own backend.
func TestResolveEndpointGenericOTelDefersToSDK(t *testing.T) {
	clearTelemetryEnv(t)
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

	endpoint, headers, ok := resolveEndpoint()
	if !ok {
		t.Fatal("expected tracing to be enabled")
	}
	if endpoint != "" {
		t.Errorf("endpoint = %q, want empty so the SDK reads its own env vars", endpoint)
	}
	if headers != nil {
		t.Errorf("headers = %v, want nil", headers)
	}
}

func TestResolveEndpointLangfuseCredentials(t *testing.T) {
	clearTelemetryEnv(t)
	t.Setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
	t.Setenv("LANGFUSE_SECRET_KEY", "sk-test")
	t.Setenv("LANGFUSE_HOST", "http://localhost:3000")

	endpoint, headers, ok := resolveEndpoint()
	if !ok {
		t.Fatal("expected tracing to be enabled")
	}
	if endpoint != "http://localhost:3000/api/public/otel/v1/traces" {
		t.Errorf("endpoint = %q", endpoint)
	}
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("pk-test:sk-test"))
	if headers["Authorization"] != wantAuth {
		t.Errorf("Authorization = %q, want %q", headers["Authorization"], wantAuth)
	}
	if headers["x-langfuse-ingestion-version"] != "4" {
		t.Errorf("x-langfuse-ingestion-version = %q", headers["x-langfuse-ingestion-version"])
	}
}

func TestResolveEndpointLangfuseHostDefaultsToCloud(t *testing.T) {
	clearTelemetryEnv(t)
	t.Setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
	t.Setenv("LANGFUSE_SECRET_KEY", "sk-test")

	endpoint, _, ok := resolveEndpoint()
	if !ok {
		t.Fatal("expected tracing to be enabled")
	}
	if endpoint != "https://cloud.langfuse.com/api/public/otel/v1/traces" {
		t.Errorf("endpoint = %q", endpoint)
	}
}

func TestResolveEndpointLangfuseHostTrailingSlashStripped(t *testing.T) {
	clearTelemetryEnv(t)
	t.Setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
	t.Setenv("LANGFUSE_SECRET_KEY", "sk-test")
	t.Setenv("LANGFUSE_HOST", "http://localhost:3000/")

	endpoint, _, _ := resolveEndpoint()
	if endpoint != "http://localhost:3000/api/public/otel/v1/traces" {
		t.Errorf("endpoint = %q, want no double slash", endpoint)
	}
}

// A partial Langfuse configuration (one key without the other) must not be
// treated as "enabled" — that would send unauthenticated or malformed
// requests instead of just staying off.
func TestResolveEndpointPartialLangfuseCredentialsIsDisabled(t *testing.T) {
	clearTelemetryEnv(t)
	t.Setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

	if _, _, ok := resolveEndpoint(); ok {
		t.Error("a public key with no secret key should not enable tracing")
	}
}

// Generic OTel vars take priority over Langfuse-specific ones: if the user
// has already pointed OTEL_EXPORTER_OTLP_ENDPOINT somewhere on purpose, kiwi
// must not silently reroute to Langfuse instead.
func TestResolveEndpointGenericOTelTakesPriority(t *testing.T) {
	clearTelemetryEnv(t)
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
	t.Setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
	t.Setenv("LANGFUSE_SECRET_KEY", "sk-test")

	endpoint, headers, ok := resolveEndpoint()
	if !ok {
		t.Fatal("expected tracing to be enabled")
	}
	if endpoint != "" || headers != nil {
		t.Errorf("endpoint=%q headers=%v, want the generic path (both empty, SDK reads env)", endpoint, headers)
	}
}

func TestConfigureNoopWhenNothingConfigured(t *testing.T) {
	clearTelemetryEnv(t)

	shutdown, err := Configure(context.Background(), nil)
	if err != nil {
		t.Fatalf("Configure: %v", err)
	}
	if shutdown == nil {
		t.Fatal("Configure must always return a usable shutdown func")
	}
	if err := shutdown(context.Background()); err != nil {
		t.Errorf("no-op shutdown returned %v, want nil", err)
	}
}

// Regression test: the OTel SDK's default error handler writes straight to
// os.Stderr. With the TUI running, a background export failure landing on
// stderr corrupts the display, since Bubble Tea assumes exclusive control of
// the terminal. Configure must install a handler that routes errors to the
// given writer instead — never to the terminal, and never dropped outright
// when a writer is given.
func TestConfigureRoutesExportErrorsAwayFromStderr(t *testing.T) {
	clearTelemetryEnv(t)
	var buf bytes.Buffer

	shutdown, err := Configure(context.Background(), &buf)
	if err != nil {
		t.Fatalf("Configure: %v", err)
	}
	t.Cleanup(func() { shutdown(context.Background()) })

	otel.Handle(errors.New("simulated export failure"))

	if !strings.Contains(buf.String(), "simulated export failure") {
		t.Errorf("error did not reach the provided writer: %q", buf.String())
	}
}

// A nil errorLog must drop errors, not panic and not fall back to stderr.
func TestConfigureNilErrorLogDropsErrorsSafely(t *testing.T) {
	clearTelemetryEnv(t)

	shutdown, err := Configure(context.Background(), nil)
	if err != nil {
		t.Fatalf("Configure: %v", err)
	}
	t.Cleanup(func() { shutdown(context.Background()) })

	otel.Handle(errors.New("should not panic"))
}
