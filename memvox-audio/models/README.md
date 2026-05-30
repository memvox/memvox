# Vendored models

Binary model files committed into the repo so the Rust binary works out of the
box without a network fetch at startup. Keep this directory small — vendor
only models that are ≤ a few MB and have stable, immutable releases upstream.

## silero_vad.onnx

Voice Activity Detection model from [snakers4/silero-vad][repo].

| | |
|---|---|
| **Upstream version** | v4.0 |
| **Source URL**       | https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx |
| **License**          | MIT (see upstream `LICENSE`) |
| **Size**             | 1.8 MB |
| **SHA-256**          | `a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28` |
| **Downloaded**       | 2026-05-28 |

### Why v4 and not v5

v4 uses a 1536-sample window at 16 kHz, which matches the existing
`FRAME_SAMPLES` constant in `ingress.rs` (one VAD call per ~96 ms frame).
v5 reduces the window to 256/512 samples and changes the state-tensor
signature; adopting it would require reworking the framing loop.

### Graph signature (so future-you can sanity-check)

```
inputs:  input (f32 [1, 1536]), sr (i64 [1]), h (f32 [2, 1, 64]), c (f32 [2, 1, 64])
outputs: output (f32 [1, 1]),   hn (f32 [2, 1, 64]),              cn (f32 [2, 1, 64])
```

`h` and `c` are LSTM state — persist them across frames within a session,
zero them when starting a new session.

### Re-downloading / verifying

```bash
curl -L -o silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx
sha256sum silero_vad.onnx
# expect: a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28
```

If the hash drifts, upstream rewrote the v4 tag — pin to a commit SHA instead.

[repo]: https://github.com/snakers4/silero-vad
