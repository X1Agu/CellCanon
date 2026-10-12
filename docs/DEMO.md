# Synthetic demo

Run a small example without datasets, model weights or API access:

```sh
python -m cellcanon.demo --output '<EMPTY_OUTPUT_DIRECTORY>'
```

The output directory must be new or empty. The default seed is 29; use
`--seed` to change it. A CPU is sufficient, and classifier fitting can use an
available GPU.

The demo fits a small predictor, generates synthetic cells, evaluates their
expression, and demonstrates observation updates and evidence withdrawal.
Its Explain example uses a scripted answer to exercise source checking.
All inputs are synthetic, and the metrics describe this example run.

| Output | Contents |
| --- | --- |
| `summary.json` | Stage results, demo settings and output filenames |
| `predictions.npz` | Continuous and classification predictions |
| `generated_cells.npz` | Generated cells and controls |
| `population_readouts.npz`, `evaluation.json` | Expression statistics and metrics |
| `update_readouts.npz`, `update.json`, `world/` | Updated predictions and state history |
| `explanation.json` | Sources, scripted answer and proof checks |

For the full workflow, follow [Preprocessing](../PREPROCESSING.md) and the
[main README](../README.md).
