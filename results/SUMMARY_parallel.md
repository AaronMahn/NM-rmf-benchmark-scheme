# Parallel run scorecard (vs recalibrated GT)

All 3 samples per condition + both conditions run simultaneously (paired on the same samples). `conc` = concurrency used. Errors are medians over the model's **valid** samples.

| model | conc | cond | valid/3 | matter med% | M med% | R med% | mean s |
|---|---|---|---|---|---|---|---|
| gpt55 | 6-way | with_skill | 3/3 | 0.00 | 0.00 | 0.00 | 109 |
| gpt55 | 6-way | without_skill | 3/3 | 0.00 | 2.37 | 16.46 | 204 |
| claude | 6-way | with_skill | 3/3 | 0.00 | 0.06 | 0.54 | 103 |
| claude | 6-way | without_skill | 3/3 | 187.68 | 15.88 | 23.77 | 352 |
| gemini | 6-way | with_skill | 3/3 | 0.00 | 0.03 | 0.15 | 386 |
| gemini | 6-way | without_skill | 2/3 | 52.36 | 0.95 | 1.01 | 364 |
| qwen | 6-way | with_skill | 2/3 | 0.00 | 0.00 | 0.00 | 161 |
| qwen | 6-way | without_skill | 0/3 | — | — | — | 154 |
| deepseek | 2-way | with_skill | 3/3 | 0.00 | 0.03 | 0.00 | 394 |
| deepseek | 2-way | without_skill | 2/3 | 2214.32 | 100.00 | 9294676467.76 | 606 |
| kimi | 2-way | with_skill | 2/3 | 101.05 | 79.93 | 74.99 | 605 |
| kimi | 2-way | without_skill | 2/3 | 1334.44 | 100.00 | 85.65 | 609 |

Columns: `matter med%` = median (over valid samples) of the worst of the four matter quantities; `M med%`/`R med%` = median M and R relative error. Per-sample detail in `par_all.json`; per-model bundles in `par_bundle_<model>/`.
