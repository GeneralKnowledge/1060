# Mind A2: language tiers vs algorithms only

Run 2026-11-03T22:35:11Z, 6000 ticks per mind, seeds 11. Same world per seed in both arms.

Dream-only run at 9499324: `config/mind.toml` was temporarily set to `max_tier_calls_per_window = 1`, `llm_dream_every = 2`, `reflect_every = 1000000` for this run only (not committed). The tiers arm took 1079.8 s. One seed: differences between arms may be run-to-run variation, and no language-tier proposal passed the memory test.

## seed 11, algo

| measure | value |
|---|---|
| delayed causes in the world / believed correctly | 4 / 0 |
| cycles in the world / found | 3 / 3 |
| cause beliefs by status (n, matching a true rule) | belief: 4 (0), hypothesis: 2 (1), rejected: 4 (0) |
| dream pairs passing the memory test vs random-pair control | 0.029 (171) vs 0.045 (67) |
| dream hypotheses that are true rules | 1 |
| language-tier dream proposals passing (n) / reflection questions passing (n) | None (0) / None (0) |
| language-tier hypotheses that are true rules | 0 |
| tier calls | none |
| mean surprise, first window / last window (bits) | 2.335 / 0.7873 |
| self-tuning changes kept / tried | 5 / 19 |
| thoughts started / attention entries | 44 / 214 |
| state entries | awake 2, background 104, deep_sleep 21, dreaming 10, resting 95 |

Current self-tuned settings: think_threshold=0.5, wake_threshold=0.63, sleep_fatigue=0.62, ws_decay=0.85, interest_rate=0.020000000000000004, dream_walk=5, lift_min=3.5, forget_surprise=1.0

written /home/red/1060-mind-exp/mind/reports/2026-11-03-6000.md

## seed 11, tiers

| measure | value |
|---|---|
| delayed causes in the world / believed correctly | 4 / 2 |
| cycles in the world / found | 3 / 3 |
| cause beliefs by status (n, matching a true rule) | belief: 3 (2), hypothesis: 2 (1), rejected: 4 (1) |
| dream pairs passing the memory test vs random-pair control | 0.027 (183) vs 0.024 (85) |
| dream hypotheses that are true rules | 4 |
| language-tier dream proposals passing (n) / reflection questions passing (n) | 0.0 (24) / None (0) |
| language-tier hypotheses that are true rules | 0 |
| tier calls | reflective/dream: 11 (0 errors, 1 unparsed, 1076.1 s) |
| mean surprise, first window / last window (bits) | 2.335 / 0.7962 |
| self-tuning changes kept / tried | 5 / 19 |
| thoughts started / attention entries | 42 / 200 |
| state entries | awake 2, background 102, deep_sleep 21, dreaming 11, resting 92 |

Current self-tuned settings: think_threshold=0.5, wake_threshold=0.55, sleep_fatigue=0.62, ws_decay=0.8099999999999999, interest_rate=0.05, dream_walk=3, lift_min=4.5, forget_surprise=1.3

written /home/red/1060-mind-exp/mind/reports/2026-11-03-6000.md

## Qwen3.5-4B as dreamer, seed 11, 6000 ticks

Tiers arm only at a2c7a41, GPU tier, 737.5 s. `config/mind.toml` was temporarily set to `max_tier_calls_per_window = 1`, `llm_dream_every = 2`, `reflect_every = 1000000`, `dream_tier = "deep"` for this run only (not committed). Every call used its whole 2048-token budget on reasoning and returned no answer, so no Qwen proposal was tested: this says nothing yet about Qwen's dreams.

| measure | value |
|---|---|
| delayed causes in the world / believed correctly | 4 / 0 |
| cycles in the world / found | 3 / 3 |
| cause beliefs by status (n, matching a true rule) | belief: 4 (0), hypothesis: 2 (1), rejected: 4 (0) |
| dream pairs passing the memory test vs random-pair control | 0.029 (171) vs 0.045 (67) |
| dream hypotheses that are true rules | 1 |
| language-tier dream proposals passing (n) / reflection questions passing (n) | None (0) / None (0) |
| language-tier hypotheses that are true rules | 0 |
| tier calls | deep/dream: 10 (0 errors, 10 unparsed, 732.3 s) |
| mean surprise, first window / last window (bits) | 2.335 / 0.7873 |
| self-tuning changes kept / tried | 5 / 19 |
| thoughts started / attention entries | 44 / 214 |
| state entries | awake 2, background 104, deep_sleep 21, dreaming 10, resting 95 |

Current self-tuned settings: think_threshold=0.5, wake_threshold=0.63, sleep_fatigue=0.62, ws_decay=0.85, interest_rate=0.020000000000000004, dream_walk=5, lift_min=3.5, forget_surprise=1.0


## Qwen3.5-4B as dreamer, no thinking, seed 11, 6000 ticks

Tiers arm only at 503bdaa, GPU tier, about 2 min. `config/mind.toml` was temporarily set to `max_tier_calls_per_window = 3`, `llm_dream_every = 2`, `reflect_every = 1000000`, `dream_tier = "deep"` for this run only (not committed). Of 90 proposals, 75 were valid and none passed the memory test. One seed. The process raised an AttributeError in `Tier.stop` during shutdown, after all 6000 ticks and the shutdown record were written.

| measure | value |
|---|---|
| delayed causes in the world / believed correctly | 4 / 0 |
| cycles in the world / found | 3 / 3 |
| cause beliefs by status (n, matching a true rule) | belief: 4 (0), hypothesis: 2 (1), rejected: 4 (0) |
| dream pairs passing the memory test vs random-pair control | 0.029 (171) vs 0.045 (67) |
| dream hypotheses that are true rules | 1 |
| language-tier dream proposals passing (n) / reflection questions passing (n) | 0.0 (90) / None (0) |
| language-tier hypotheses that are true rules | 0 |
| tier calls | deep/dream: 30 (0 errors, 0 unparsed, 103.9 s) |
| mean surprise, first window / last window (bits) | 2.335 / 0.7873 |
| self-tuning changes kept / tried | 5 / 19 |
| thoughts started / attention entries | 44 / 214 |
| state entries | awake 2, background 104, deep_sleep 21, dreaming 10, resting 95 |

Current self-tuned settings: think_threshold=0.5, wake_threshold=0.63, sleep_fatigue=0.62, ws_decay=0.85, interest_rate=0.020000000000000004, dream_walk=5, lift_min=3.5, forget_surprise=1.0


## Qwen dreamer, 5 seeds

Seeds 21 to 25, 6000 ticks each, tiers arm only at 1859d2d, GPU tier without thinking, about 10 min in total. `config/mind.toml` was temporarily set to `max_tier_calls_per_window = 3`, `llm_dream_every = 2`, `reflect_every = 1000000`, `dream_tier = "deep"` for this run only (not committed). All five runs shut down cleanly. 135 calls, median 3.33 s, none unparsed, no errors, no reasoning.

| source | proposed | tested (valid) | passed | pass rate of tested | median lift | true rules |
|---|---|---|---|---|---|---|
| dream_llm (Qwen) | 405 | 357 | 8 | 0.022 | 1.68 | 1 |
| dream (graph walk) | 802 | 802 | 35 | 0.044 | 4.67 | 5 |
| dream_control (random pairs) | 355 | 355 | 11 | 0.031 | 4.16 | not recorded |

| seed | dream_llm tested / passed | dream n / passed | control n / passed | true rules llm / dream |
|---|---|---|---|---|
| 21 | 75 / 4 | 158 / 14 | 68 / 3 | 0 / 2 |
| 22 | 80 / 0 | 184 / 4 | 88 / 1 | 0 / 0 |
| 23 | 69 / 1 | 155 / 7 | 78 / 3 | 1 / 1 |
| 24 | 61 / 0 | 143 / 2 | 55 / 0 | 0 / 0 |
| 25 | 72 / 3 | 162 / 8 | 66 / 4 | 0 / 2 |

Qwen's pass rate is not better than random pairs (one-sided Fisher p = 0.32 for lower), and neither is the graph walk clearly better than random pairs (one-sided p = 0.20 for higher). Passes are pooled across seeds and few, so these are rough.
