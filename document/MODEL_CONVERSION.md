# Model Conversion (GGUF)

Granite loads GGUF models. The easiest workflow is to use llama.cpp tooling.

## Convert from Hugging Face

From a local Hugging Face model directory:

```bash
# In llama.cpp
python3 convert_hf_to_gguf.py /path/to/hf-model --outtype f16 --outfile model-f16.gguf
```

## Quantize

Quantize the GGUF weights for faster inference:

```bash
# In llama.cpp
./quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

Common quantization presets:

- `Q4_K_M` (balanced)
- `Q5_K_M` (higher quality)
- `Q8_0` (higher fidelity, more memory)

## Validate

Run Granite on the resulting GGUF:

```bash
./build/granite_main /path/to/model-q4_k_m.gguf "Hello"
```

## Notes

- Granite expects LLaMA-style GGUF metadata (tokenizer, rope config, etc.).
- See `document/gguf-implementation.md` for loader details.
