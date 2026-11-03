# Mind A0 feasibility (language tiers)

Run 2026-11-03T22:12:18Z on the laptop (12 logical CPUs), llama.cpp b12021 build, measurement only.

| model | threads | load s | prompt tok/s | gen tok/s | RSS MB | JSON schema |
|---|---|---|---|---|---|---|
| LFM2.5-350M-Q8_0.gguf (379 MB) | 2 | 1.5 | 1649.5 | 55.0 | 722 | ok |
| LFM2.5-350M-Q8_0.gguf (379 MB) | 4 | 0.5 | 2206.8 | 65.6 | 722 | ok |
| LFM2.5-1.2B-Thinking-Q4_K_M.gguf (731 MB) | 2 | 3.5 | 833.0 | 24.5 | 1073 | ok |
| LFM2.5-1.2B-Thinking-Q4_K_M.gguf (731 MB) | 4 | 0.5 | 1075.1 | 31.8 | 1073 | ok |

LFM2.5-1.2B-Thinking-Q4_K_M.gguf reasoning on a short question: {'completion_tokens': 1480, 'reasoning_chars': 5079, 'answer_chars': 1791}

Qwen3.5-4B (GPU, lab winning settings): {"pp_tok_s": 510.6, "tg_tok_s": 29.0, "prompt_tokens": 578, "rss_mb": 1172, "json_schema_ok": true, "load_s": 2.1, "vram_mib": 3035, "unload_s": 0.28}
