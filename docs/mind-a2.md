# Mind A2: language tiers vs algorithms only

Run 2026-11-03T22:18:18Z, 4000 ticks per mind, seeds 11. Same world per seed in both arms.

Short run: `config/mind.toml` was temporarily set to `max_tier_calls_per_window = 1`, `llm_dream_every = 3`, `reflect_every = 10` for this run only (not committed). The tiers arm took 868.6 s. One seed and 4000 ticks is too short to compare dreaming: neither arm learned a delayed cause.

## seed 11, algo

| measure | value |
|---|---|
| delayed causes in the world / believed correctly | 4 / 0 |
| cycles in the world / found | 3 / 2 |
| cause beliefs by status (n, matching a true rule) | belief: 2 (0), hypothesis: 3 (0), rejected: 4 (0) |
| dream pairs passing the memory test vs random-pair control | 0.017 (117) vs 0.064 (47) |
| dream hypotheses that are true rules | 0 |
| language-tier dream proposals passing (n) / reflection questions passing (n) | None (0) / None (0) |
| language-tier hypotheses that are true rules | 0 |
| tier calls | none |
| mean surprise, first window / last window (bits) | 2.335 / 0.7323 |
| self-tuning changes kept / tried | 3 / 12 |
| thoughts started / attention entries | 41 / 167 |
| state entries | awake 2, background 69, deep_sleep 14, dreaming 7, resting 61 |

Current self-tuned settings: think_threshold=0.5, wake_threshold=0.55, sleep_fatigue=0.7, ws_decay=0.85, interest_rate=0.05, dream_walk=4, lift_min=3.5, forget_surprise=1.0

written /home/red/1060-mind-exp/mind/reports/2026-11-03-4000.md

## seed 11, tiers

| measure | value |
|---|---|
| delayed causes in the world / believed correctly | 4 / 0 |
| cycles in the world / found | 3 / 2 |
| cause beliefs by status (n, matching a true rule) | belief: 2 (0), hypothesis: 3 (0), rejected: 4 (0) |
| dream pairs passing the memory test vs random-pair control | 0.025 (119) vs 0.078 (51) |
| dream hypotheses that are true rules | 0 |
| language-tier dream proposals passing (n) / reflection questions passing (n) | None (0) / 0.0 (23) |
| language-tier hypotheses that are true rules | 0 |
| tier calls | reflective/reflect: 12 (0 errors, 2 unparsed, 757.1 s), reflective/dream: 2 (0 errors, 0 unparsed, 108.8 s) |
| mean surprise, first window / last window (bits) | 2.335 / 0.7323 |
| self-tuning changes kept / tried | 3 / 12 |
| thoughts started / attention entries | 41 / 168 |
| state entries | awake 2, background 69, deep_sleep 14, dreaming 7, resting 61 |

Current self-tuned settings: think_threshold=0.5, wake_threshold=0.47000000000000003, sleep_fatigue=0.62, ws_decay=0.85, interest_rate=0.05, dream_walk=4, lift_min=3.5, forget_surprise=1.0

Latest reflections (its own notes):
- The recent changes suggest unresolved lore mysteries requiring deeper analysis.
- The recent changes highlight unresolved mysteries requiring deeper analysis.
- The recent changes suggest unresolved lore mysteries requiring deeper analysis.
- The recent changes highlight unresolved lore mysteries requiring deeper analysis.
- The recent changes underscore unresolved mysteries requiring deeper analysis.

written /home/red/1060-mind-exp/mind/reports/2026-11-03-4000.md
