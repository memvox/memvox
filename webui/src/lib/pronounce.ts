/** Pronunciation service.
 *
 * The UI only ever talks to `pronounce()` / `stopPronouncing()`, so the
 * implementation can be swapped without touching any mode component.
 *
 * Current implementation: browser SpeechSynthesis with a ko-KR voice — a
 * reference-quality stub. Planned replacements, in order:
 *   1. pre-rendered / streamed Cartesia audio served by the memvox bridge
 *   2. recording the learner and scoring against the reference (assessment)
 */

let cachedKoVoice: SpeechSynthesisVoice | null | undefined;

function koreanVoice(): SpeechSynthesisVoice | null {
  if (cachedKoVoice !== undefined) return cachedKoVoice;
  const voices = window.speechSynthesis.getVoices();
  cachedKoVoice =
    voices.find((v) => v.lang.replace("_", "-").toLowerCase() === "ko-kr") ??
    voices.find((v) => v.lang.toLowerCase().startsWith("ko")) ??
    null;
  return cachedKoVoice;
}

// Voice lists load asynchronously in most browsers; refresh the cache.
if (typeof window !== "undefined" && "speechSynthesis" in window) {
  window.speechSynthesis.addEventListener?.("voiceschanged", () => {
    cachedKoVoice = undefined;
  });
}

export function pronunciationAvailable(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/** Speak `text` in Korean. Cancels anything currently speaking. */
export function pronounce(text: string, opts?: { slow?: boolean }): void {
  if (!pronunciationAvailable()) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ko-KR";
  const voice = koreanVoice();
  if (voice) utterance.voice = voice;
  utterance.rate = opts?.slow ? 0.6 : 0.85;
  window.speechSynthesis.speak(utterance);
}

export function stopPronouncing(): void {
  if (pronunciationAvailable()) window.speechSynthesis.cancel();
}
