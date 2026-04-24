"""Module-level metrics singleton.

Usage
-----
    from memvox.observability import metrics

    async with metrics.span("asr.transcribe", turn_id=t):
        ...

    metrics.event(metrics.LLM_TTFT, latency_ms=180.0, turn_id=t)

Sink is selected at import time via MEMVOX_METRICS_SINK env var (default: memory).
In tests, swap the sink for the duration of a block:

    with metrics.override() as sink:
        ...
        assert sink.spans[0].name == "asr.transcribe"
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

# ── Predefined event names ────────────────────────────────────────────────────

SESSION_START   = "session.start"
SESSION_END     = "session.end"
BARGE_IN        = "barge_in"
ASR_DROP        = "asr.drop"
TTS_FIRST_CHUNK = "tts.first_chunk"
LLM_TTFT        = "llm.ttft"
WIKI_QUERY      = "wiki.query"

# ── Records (MemorySink inspection surface) ───────────────────────────────────

@dataclass
class SpanRecord:
    name: str
    duration_ms: float
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    name: str
    latency_ms: float | None
    attrs: dict[str, Any] = field(default_factory=dict)


# ── Sinks ─────────────────────────────────────────────────────────────────────

class _Sink:
    def record_span(self, name: str, duration_ms: float, attrs: dict[str, Any]) -> None:
        pass

    def record_event(self, name: str, latency_ms: float | None, attrs: dict[str, Any]) -> None:
        pass


class MemorySink(_Sink):
    """In-process sink. Inspect .spans and .events in tests."""

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []
        self.events: list[EventRecord] = []

    def record_span(self, name: str, duration_ms: float, attrs: dict[str, Any]) -> None:
        self.spans.append(SpanRecord(name=name, duration_ms=duration_ms, attrs=attrs))

    def record_event(self, name: str, latency_ms: float | None, attrs: dict[str, Any]) -> None:
        self.events.append(EventRecord(name=name, latency_ms=latency_ms, attrs=attrs))


class _OTLPSink(_Sink):
    """Thin OTel wrapper: spans → histogram, events with latency → histogram, counts → counter."""

    def __init__(self) -> None:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
        provider = MeterProvider(metric_readers=[reader])
        otel_metrics.set_meter_provider(provider)
        self._meter = provider.get_meter("memvox")
        self._histograms: dict[str, Any] = {}
        self._counters: dict[str, Any] = {}

    def _histogram(self, name: str) -> Any:
        if name not in self._histograms:
            safe = name.replace(".", "_")
            self._histograms[name] = self._meter.create_histogram(
                f"memvox.{safe}.ms", unit="ms", description=f"memvox {name} latency"
            )
        return self._histograms[name]

    def _counter(self, name: str) -> Any:
        if name not in self._counters:
            safe = name.replace(".", "_")
            self._counters[name] = self._meter.create_counter(f"memvox.{safe}.total")
        return self._counters[name]

    def record_span(self, name: str, duration_ms: float, attrs: dict[str, Any]) -> None:
        self._histogram(name).record(duration_ms, attributes=attrs)

    def record_event(self, name: str, latency_ms: float | None, attrs: dict[str, Any]) -> None:
        if latency_ms is not None:
            self._histogram(name).record(latency_ms, attributes=attrs)
        else:
            self._counter(name).add(1, attributes=attrs)


class _PrometheusSink(_Sink):
    """OTel → Prometheus HTTP scrape endpoint."""

    def __init__(self) -> None:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        import prometheus_client

        port = int(os.environ.get("MEMVOX_PROMETHEUS_PORT", "9090"))
        prometheus_client.start_http_server(port)
        reader = PrometheusMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        otel_metrics.set_meter_provider(provider)
        self._meter = provider.get_meter("memvox")
        self._histograms: dict[str, Any] = {}
        self._counters: dict[str, Any] = {}

    def _histogram(self, name: str) -> Any:
        if name not in self._histograms:
            safe = name.replace(".", "_")
            self._histograms[name] = self._meter.create_histogram(
                f"memvox_{safe}_ms", unit="ms"
            )
        return self._histograms[name]

    def _counter(self, name: str) -> Any:
        if name not in self._counters:
            safe = name.replace(".", "_")
            self._counters[name] = self._meter.create_counter(f"memvox_{safe}_total")
        return self._counters[name]

    def record_span(self, name: str, duration_ms: float, attrs: dict[str, Any]) -> None:
        self._histogram(name).record(duration_ms, attributes=attrs)

    def record_event(self, name: str, latency_ms: float | None, attrs: dict[str, Any]) -> None:
        if latency_ms is not None:
            self._histogram(name).record(latency_ms, attributes=attrs)
        else:
            self._counter(name).add(1, attributes=attrs)


# ── Collector ─────────────────────────────────────────────────────────────────

class MetricsCollector:
    def __init__(self, sink: _Sink) -> None:
        self._sink = sink

    @asynccontextmanager
    async def span(self, name: str, **attrs: Any):
        """Record the wall-clock duration of an async block."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._sink.record_span(name, (time.monotonic() - t0) * 1000, attrs)

    def event(self, name: str, **attrs: Any) -> None:
        """Record a named event. Pass latency_ms=x for a latency measurement."""
        latency_ms: float | None = attrs.pop("latency_ms", None)
        self._sink.record_event(name, latency_ms, attrs)


# ── Singleton + module-level API ──────────────────────────────────────────────

def _build_sink() -> _Sink:
    kind = os.environ.get("MEMVOX_METRICS_SINK", "memory").lower()
    if kind == "otlp":
        return _OTLPSink()
    if kind == "prometheus":
        return _PrometheusSink()
    return MemorySink()


_active = MetricsCollector(_build_sink())


@asynccontextmanager
async def span(name: str, **attrs: Any):
    async with _active.span(name, **attrs):
        yield


def event(name: str, **attrs: Any) -> None:
    _active.event(name, **attrs)


@contextmanager
def override() -> Generator[MemorySink, None, None]:
    """Swap in a fresh MemorySink for the duration of a with-block.

        with metrics.override() as sink:
            await component.run(...)
            assert sink.spans[0].name == "asr.transcribe"
            assert sink.spans[0].duration_ms < 600
    """
    global _active
    previous = _active
    sink = MemorySink()
    _active = MetricsCollector(sink)
    try:
        yield sink
    finally:
        _active = previous
