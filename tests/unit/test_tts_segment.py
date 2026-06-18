"""Tests for mixed-script segmentation in tts_base (segment_for_tts).

Covers the language + speed plan that the Cartesia backend renders: embedded
Korean help phrases slow, immersion Korean normal.
"""

from memvox.voice.tts_base import segment_for_tts, split_script_runs


class TestSplitScriptRuns:
    def test_pure_english_is_one_run(self):
        assert split_script_runs("Hello there") == ["Hello there"]

    def test_pure_korean_is_one_run(self):
        assert split_script_runs("안녕하세요") == ["안녕하세요"]

    def test_boundary_between_scripts(self):
        assert split_script_runs("say 물") == ["say ", "물"]

    def test_punctuation_stays_with_current_run(self):
        # The trailing period attaches to the Korean run, not its own run.
        assert split_script_runs("It is 물.") == ["It is ", "물."]

    def test_leading_neutral_merges_into_next_run(self):
        assert split_script_runs("  hi") == ["  hi"]


class TestSegmentForTTS:
    def test_pure_english_normal(self):
        assert segment_for_tts("How are you?", "ko") == [
            ("How are you?", "en", "normal")
        ]

    def test_pure_korean_normal(self):
        assert segment_for_tts("안녕하세요!", "ko") == [
            ("안녕하세요!", "ko", "normal")
        ]

    def test_embedded_korean_help_phrase_is_slow(self):
        units = segment_for_tts("You say 안녕하세요 here.", "ko")
        langs_speeds = [(l, s) for _, l, s in units]
        assert ("ko", "slow") in langs_speeds
        assert all(s == "normal" for l, s in langs_speeds if l == "en")

    def test_korean_primary_with_english_loanword_stays_normal(self):
        # Korean is the majority script → not a "help phrase" → normal speed.
        units = segment_for_tts("네, 'subway'는 지하철이에요.", "ko")
        assert all(speed == "normal" for _, _, speed in units)

    def test_help_speed_is_configurable(self):
        units = segment_for_tts("Say 물 now.", "ko", korean_help_speed="fast")
        assert any(l == "ko" and s == "fast" for _, l, s in units)

    def test_whitespace_only_yields_nothing(self):
        assert segment_for_tts("   ", "ko") == []

    def test_punctuation_only_yields_nothing(self):
        # Cartesia rejects transcripts with no speakable characters.
        assert segment_for_tts("...!?", "ko") == []
        assert segment_for_tts("—", "ko") == []

    def test_punctuation_only_run_is_dropped(self):
        # The standalone "?!" between scripts must not become its own unit.
        for text, lang, _ in segment_for_tts("Say 물 ?!", "ko"):
            assert any(ch.isalnum() for ch in text)

    def test_non_korean_session_passes_lang_through(self):
        # Non-Korean session: language is the configured code, speed normal.
        assert segment_for_tts("Bonjour", "fr") == [("Bonjour", "fr", "normal")]
