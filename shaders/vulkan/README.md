# Vulkan Shaders

These GLSL compute shaders are adapted from llama.cpp's Vulkan backend.

## Source

- Repository: https://github.com/ggml-org/llama.cpp
- Path: ggml/src/ggml-vulkan/vulkan-shaders/
- License: MIT

## License

```
MIT License

Copyright (c) 2023-2025 The llama.cpp Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Key Shaders for LLM Inference

### Matrix-Vector (Decode)
- `mul_mat_vec_q4_k.comp` - Q4_K quantized matvec
- `mul_mat_vec_q*.comp` - Other quantization formats
- `mul_mat_vec_base.glsl` - Shared matvec infrastructure

### Normalization
- `rms_norm.comp` - RMS normalization

### Attention
- `flash_attn.comp` - Flash attention
- `soft_max.comp` - Softmax

### Activations
- `silu.comp` - SiLU activation
- `gelu.comp` - GELU activation

### Position Embeddings
- `rope_multi.comp` - Rotary position embedding (multihead)

### Common Includes
- `types.glsl` - Quantization block type definitions
- `dequant_funcs.glsl` - Dequantization helpers
