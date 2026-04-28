from memvox.observability import metrics


class TestSummary:
    def test_returns_none_when_no_events(self):
        with metrics.override():
            assert metrics.summary() is None

    def test_includes_event_with_latency(self):
        with metrics.override():
            metrics.event(metrics.MOUTH_TO_EAR, latency_ms=420.0, turn_id="t1")
            report = metrics.summary()
        assert report is not None
        assert "mouth_to_ear" in report
        assert "n=  1" in report
        assert "420" in report

    def test_includes_span_with_label(self):
        async def _emit():
            async with metrics.span("asr.transcribe", turn_id="t1"):
                pass

        import asyncio
        with metrics.override():
            asyncio.run(_emit())
            report = metrics.summary()
        assert report is not None
        assert "asr.transcribe (span)" in report

    def test_aggregates_multiple_samples(self):
        with metrics.override():
            metrics.event(metrics.MOUTH_TO_EAR, latency_ms=100.0)
            metrics.event(metrics.MOUTH_TO_EAR, latency_ms=200.0)
            metrics.event(metrics.MOUTH_TO_EAR, latency_ms=300.0)
            report = metrics.summary()
        assert report is not None
        assert "n=  3" in report
        # avg = 200, p95 should be 300 (last sample after sort)
        assert "200.0" in report
        assert "300.0" in report

    def test_skips_events_without_latency(self):
        with metrics.override():
            metrics.event(metrics.SESSION_START, session_id="s1")  # no latency_ms
            report = metrics.summary()
        # SESSION_START has no latency, summary should ignore it → no events,
        # no spans → returns None
        assert report is None
