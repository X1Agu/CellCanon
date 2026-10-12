# CellCanon

CellCanon combines biological knowledge and cellular observations to predict
perturbation responses, explain source-supported hypotheses, and update
predictions after new measurements.

**Paper:** [CellCanon: A Knowledge-Grounded Agent for Cellular Prediction,
Explanation, and Updating](https://openreview.net/forum?id=fRua0VKGhz) —
NeurIPS 2026 Workshop AgenticLS, Poster.

[![CellCanon architecture: prediction, evidence-grounded explanation and observation-driven updating](docs/assets/overview.png)](docs/assets/overview.png)

- **Predict** single-cell perturbation responses using biological evidence
  shared across gene-specific numerical readouts.
- **Explain** biological hypotheses with source-linked evidence and revise
  answers when evidence checks identify inconsistencies.
- **Update** predictions for unmeasured cell contexts as new observations
  become available, while keeping predictor weights fixed.

A shared world state connects predictions, evidence and acquired measurements.

## Installation

Use Python 3.11 on Linux with an NVIDIA GPU. See the
[environment guide](environments/README.md) for dependency versions,
installation commands and CPU/GPU requirements.

After installation:

```sh
python -m cellcanon --help
```

An [optional synthetic demo](docs/DEMO.md) runs without datasets or API access:

```sh
python -m cellcanon.demo --output ../cellcanon-demo
```

## Data and configuration

Download the three prepared input bundles from
[Hugging Face](https://huggingface.co/datasets/X1Agu/CellCanon):

```sh
python tools/download_data.py --output ./data
```

The command checks the archive hashes and extracts the inputs for the existing
loaders. Use these paths in your local configuration:

| Setting | Path | Contents |
| --- | --- | --- |
| `data_root` | `data/expression` | Core2000 + Extra931 expression, DMSO, cell/dose/replicate metadata, continuous responses, DE/DIR labels and sampled queries |
| `metadata_root` | `data/metadata` | Drug structures and annotations, gene order, splits and manifests |
| `knowledge_root` | `data/knowledge` | Knowledge and route databases with source-linked evidence and DMSO context |
| `compound_registry` | `data/metadata/compound_registry.parquet` | Compound structures and annotations for rebuilding metadata |

Download [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
separately and set `embedding_model_root` to its local directory.
Source attribution and manual downloads are in [Data sources](DATA_SOURCES.md).
To rebuild inputs from raw sources, see [Preprocessing](PREPROCESSING.md).

Copy `config.example.json` to your configuration file and replace its path
placeholders. Paths can be absolute or relative to that file. Resource
preparation also uses `resource_paths.example.json`; see
[Input schemas](docs/INPUT_SCHEMAS.md).

The `compile` and `explain` stages use a JSON chat completion API. API settings
are blank; fill in your own endpoint, model and credential environment variable using
[API configuration](docs/MODEL_SERVICE.md). Keep credentials and generated
outputs outside the repository.

## Predict

After preparing the inputs, run the following commands. Replace `<CONFIG_JSON>`
with your configuration path. If preprocessing already completed `knowledge`,
start with `compile`.

```sh
python -m cellcanon knowledge --config '<CONFIG_JSON>'
python -m cellcanon compile --config '<CONFIG_JSON>'
python -m cellcanon embed --config '<CONFIG_JSON>'
python -m cellcanon prepare --config '<CONFIG_JSON>'
python -m cellcanon train --config '<CONFIG_JSON>'
python -m cellcanon predict --config '<CONFIG_JSON>'
python -m cellcanon fit-cells --config '<CONFIG_JSON>'
python -m cellcanon generate --config '<CONFIG_JSON>'
python -m cellcanon evaluate --config '<CONFIG_JSON>'
```

`predict` produces condition-level readouts. `generate` produces cells and
population statistics relative to matched DMSO controls. `evaluate` verifies
the saved output hashes before reading test responses.

The continuous endpoint uses 2,000 genes. Expression-based DE uses a rank-sum
normal approximation without tie correction, BH correction over 2,931 genes,
adjusted p-value below 0.05, and absolute log2FC above 0.25. DIR ranking uses
signed generated logFC; its hard decision is `logFC > 0`. The additional
931-gene panel is evaluated separately.

## Explain

```sh
python -m cellcanon prepare-explain --config '<CONFIG_JSON>'
python -m cellcanon explain --config '<CONFIG_JSON>'
python -m cellcanon score-explain --config '<CONFIG_JSON>'
```

Explain builds questions from the supplied sources, checks answers against
their cited evidence, and revises answers that fail those checks. Each claim
includes a `supported`, `contradicted` or `insufficient` decision, source IDs
and a reason. `insufficient` means the available evidence establishes neither
the claim nor its opposite. See [Evidence scope](docs/INPUT_SCHEMAS.md#evidence-scope)
for interpretation of background routes and experimental evidence.

## Update

```sh
python -m cellcanon fit-update-oof --config '<CONFIG_JSON>'
python -m cellcanon fit-update --config '<CONFIG_JSON>'
python -m cellcanon update --config '<CONFIG_JSON>'
python -m cellcanon evaluate-update --config '<CONFIG_JSON>'
```

Update fits five whole-drug out-of-fold parents using support data, then
applies the fitted update models to scheduled observations. Evaluation uses
condition-level continuous, DE and DIR readouts. DIR ranking uses the up
probability, with a hard decision at `p_up >= 0.5`.

Experiment definitions in `src/cellcanon/protocols/` specify gene order,
drug splits, model settings and observation schedules.

## Outputs

Results are written under the configured `run_root`:

| Path | Contents |
| --- | --- |
| `model/freeze/compiled_semantic.npz` | Condition-level continuous responses, DE/DIR probabilities and DE decisions |
| `cells_test/<CELL>.h5` | Generated expression grouped by drug and dose, with matched control row IDs |
| `cells_test/population_readouts.npz`, `predict_metrics.json` | Population statistics and prediction metrics |
| `explanations/<CASE_ID>/result.json` | Claims, cited evidence, route rankings and unresolved source checks |
| `explanation_metrics.json` | Claim and route scores from `score-explain` |
| `update/predictions/`, `update_metrics.json` | Predictions after each observation budget and their evaluation metrics |

In an Explain result, `value` contains the answers and route rankings;
`stages` records which inference and correction steps succeeded. Unresolved
checks identify answers or routes that still need review.

## Project structure

| Directory | Contents |
| --- | --- |
| `preprocessing/` | Expression preprocessing and drug splits |
| `src/cellcanon/resources/` | Public knowledge preparation |
| `src/cellcanon/model/` | Predict, Explain and Update implementations |
| `src/cellcanon/protocols/` | Experiment definitions |
| `tests/` | Automated tests |
| `environments/` | Dependency versions and installation instructions |

## License

The source code is licensed under [MIT](LICENSE). External datasets, database
records and model weights retain their own licenses; see [NOTICE.md](NOTICE.md).
Contribution instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).
