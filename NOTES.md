# Notes

Loose collection of non-obvious things learned while building memvox. Anything
that took longer than 10 minutes to figure out and isn't self-evident from the
code goes here.

## `ort` 2.0.0-rc.x (ONNX Runtime crate)

Three pitfalls that aren't called out in the README. All discovered while
wiring up Silero VAD in `memvox-audio/src/vad.rs`.

1. **`ort::Error<_>` is `!Send + !Sync`.** It wraps raw FFI handles
   (`NonNull<OrtSessionOptions>`, etc.), so it cannot be auto-converted into
   `anyhow::Error` via `?`. Convert manually at every call site:

   ```rust
   fn oe<E: std::fmt::Display>(e: E) -> anyhow::Error {
       anyhow::anyhow!("ort: {e}")
   }
   let session = Session::builder().map_err(oe)?
       .with_optimization_level(GraphOptimizationLevel::Level3).map_err(oe)?
       .commit_from_file(path).map_err(oe)?;
   ```

2. **`ort::inputs![...]` returns `Vec<(Cow<str>, SessionInputValue)>`, not
   `Result<_>`.** Don't put a `?` after the macro:

   ```rust
   // wrong
   session.run(ort::inputs![ "x" => t? ]?)?;
   // right
   session.run(ort::inputs![ "x" => t? ])?;
   ```

3. **`try_extract_tensor::<T>()` returns `(&Shape, &[T])`** — a tuple, not an
   `ArrayView`. Destructure or use `.1` for the data:

   ```rust
   let (_, data) = outputs["output"].try_extract_tensor::<f32>().map_err(oe)?;
   let prob = data[0];
   ```

4. **Skip `ndarray` on the input side.** `Tensor::from_array` accepts a
   `(shape, Vec<T>)` tuple, which avoids pulling in `ndarray` and having to
   version-match it against ort's internal copy. Multiple `ndarray` versions
   in the dep graph causes a confusing `OwnedTensorArrayData` trait-bound
   failure.

   ```rust
   Tensor::from_array(([1_i64, 1536], samples_vec))
   ```

5. **`features = ["download-binaries"]` needs `libssl-dev`.** The fetcher
   crate links `openssl-sys`. On Debian/Ubuntu: `sudo apt install libssl-dev`.
   For a packaged binary, switch to `["load-dynamic"]` and ship/point at
   `libonnxruntime.so` explicitly via `ORT_DYLIB_PATH`.

## `cpal` 0.15 — `Stream` is `!Send`

`cpal::Stream` holds raw ALSA pointers and is `!Send`. You can't build it
inside an async function and then `.await` afterwards: rustc's drop-tracking
is conservative across await points and will poison the future's auto-`Send`
even if you `std::mem::forget(stream)` to suppress the destructor.

**Fix**: extract stream creation into a *sync* helper (see
`open_input_stream` / `open_output_stream` in `ingress.rs` / `egress.rs`).
The `!Send` value never enters the async state machine.

## Qwen3 thinking mode — silently eats `max_tokens`

Qwen3 defaults to thinking-on. Sending `enable_thinking: false` at the *top
level* of the chat completions request is ignored; it must go through
`extra_body.chat_template_kwargs`:

```python
extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
```

Symptom if you get this wrong: ASR + wiki run, no LLM TTFT event fires, no
audio output. The thinking tokens consume the entire `max_tokens` budget
before any visible content is produced, and our parser drops `<think>`
content. Always pass the flag explicitly — don't conditionally send only
when true.

## `faster-whisper` — `no_speech_prob` moved

In newer versions, `no_speech_prob` is per-segment, not on
`TranscriptionInfo`. Iterate segments and aggregate:

```python
seg_list = list(segs)
no_speech_prob = max(
    (getattr(s, "no_speech_prob", 0.0) for s in seg_list),
    default=1.0,
)
```

## `lancedb` async API

`search()` is async and returns the query builder. Chain after `await`:

```python
vec_query = await self._table.search(query_emb[0].tolist())
vec_rows = await vec_query.limit(k).to_list()
```

FTS index creation has also moved off the standalone `create_fts_index`
method:

```python
await self._table.create_index("chunk_text", config=FTS(), replace=True)
```

## Linux audio device routing — cpal vs PipeWire

`cpal` uses ALSA. PipeWire devices are not directly visible by their
PipeWire names. To pick a specific headset (e.g. Jabra):

- Use the `pipewire` PCM (routes through PipeWire respecting `wpctl`
  default), or
- Pin directly with `plughw:CARD=<NAME>` (e.g. `plughw:CARD=J50`).

Use `memvox-audio --list-devices` to see what cpal can see by name.

## CUDA library loading

When PyTorch is built against CUDA 13, you need `libnvrtc-builtins.so.13.0`
on `LD_LIBRARY_PATH`. The pip-installed copy lives at
`.venv/lib/python*/site-packages/nvidia/cu13/lib/`. Adding it (and
`nvidia-cublas-cu12 / nvidia-cudnn-cu12`'s lib dirs) to `LD_LIBRARY_PATH` in
the venv's `activate` script is the cleanest fix.
