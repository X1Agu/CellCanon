# Input preparation

Prepared inputs are available in the [CellCanon dataset](https://huggingface.co/datasets/X1Agu/CellCanon).
For this route, use the README's download command and input paths, then check:

```sh
python -m cellcanon.resource_preflight --config config.json --stage prepared --hashes
python -m cellcanon knowledge --config config.json
```

Continue with Predict, Explain, and Update in the README. The sections below
describe the optional route for rebuilding these inputs from their raw sources.

Prepare inputs in external directories using the documented public releases.
The repository contains code and protocol metadata, not expression datasets,
knowledge snapshots, model weights or credentials. Replace every quoted path
placeholder before running a command. Input conversion, preflight, database
construction and raw preprocessing run on CPU; follow the main README's CUDA
configuration for model fitting and text embedding. Cell generation and
expression testing run on CPU.

## 1. Expression and metadata

The expression builder uses human features, official hash QC, complete metadata
and the 24 h time point. It recomputes human library totals, applies log1p(CP10K),
selects 2,000 Seurat-v3 HVGs, adds measured non-overlapping L1000 landmark genes,
and computes reference statistics and sampled DE/DIR queries.

```sh
python preprocessing/build_sciplex3_readout.py \
  --raw-root '<RAW_SCIPLEX_ROOT>' \
  --output-root '<PREPROCESSED_DATA_ROOT>' \
  --lincs-gene-info '<LINCS_GENE_INFO_GZ>' \
  --ncbi-gene-info '<NCBI_HUMAN_GENE_INFO_GZ>' \
  --official-valid-cells '<OFFICIAL_VALID_CELLS_TSV>' --stage all
python preprocessing/create_sciplex3_drug_zero_shot_split.py \
  --dataset-root '<PREPROCESSED_DATA_ROOT>'
python -m cellcanon.resource_preflight --config config.json --stage metadata
python -m cellcanon metadata --config config.json
```

Configure `data_root`, `metadata_root`, `knowledge_root`, `embedding_model_root`,
`run_root` and the external `compound_registry` using `config.example.json`.
Paths are relative to the configuration file unless absolute. The registry
requires `drug`, `smiles`, `target`, and `pathway` columns; the metadata command
maps catalogue names, computes fingerprints and checks the exact ordered gene
panel. The training, validation and test drug lists are stored in
[`src/cellcanon/protocols/split.json`](src/cellcanon/protocols/split.json).

## 2. Public-source normalization

Use `resource_paths.example.json` to configure five external source roots.
The complete file and column contracts, source-release restrictions, and each
converter command appear in [INPUT_SCHEMAS.md](docs/INPUT_SCHEMAS.md).

| Input | Executable preparation |
|---|---|
| HGNC approved records and public compound identities | `python -m cellcanon.input_convert public-entities ...` |
| Already sourced entity/alias tables | `python -m cellcanon.input_convert entities ...` |
| Human Reactome v97 GMT, OmniPath and DoRothEA A/B/C TSVs | `python -m cellcanon.input_convert network ... --source-release v97` |
| ChEMBL 37 activities, assays, targets, molecules and explicit identity mapping | `python -m cellcanon.input_convert bindings ...` |
| Guide to Pharmacology 2026.2 ligand and interaction CSVs | `python -m cellcanon.input_convert functional ... --source-release 2026.2` |
| DrugCentral 2021-09-01 structures and interaction TSVs | `python -m cellcanon.input_convert drugcentral ... --source-release 2021-09-01` |

These converters perform no network or language-model requests. Raw source
exports and the compound-to-molecule mapping must be supplied. The binding
converter checks full standard-parent chemical identity. Supplier statements
require public source references and matching compound identities. Required
tables and columns are validated before preparation; missing action signs
remain missing.

Supported source versions and downloads are listed in
[Data sources](DATA_SOURCES.md) and [Input schemas](docs/INPUT_SCHEMAS.md).
Guide to Pharmacology version headers are checked in both CSVs. Reactome GMT
and DrugCentral TSV lack release headers, so preparation records the supplied
version declaration and input hashes. DoRothEA rows require at least one A/B/C
confidence assignment: mixed lists such as `A;D` are accepted; D/E-only and
missing-confidence rows are rejected.

Keep NCBI human gene information at
`<DATA_PARENT>/reference/ncbi_gene/Homo_sapiens.gene_info.gz` for gene enrichment.
`<DATA_PARENT>` is the parent of the configured `data_root`. The control-context
stage also needs the raw matrix recorded in the preprocessing build manifest.

## 3. Knowledge resources in dependency order

Before a stage, `--check-only --hashes` inspects its local prerequisites without
writing files or making network requests. It reports logical resource names
and missing fields, rather than local machine paths:

```sh
python -m cellcanon.resource_prepare local-audit --config config.json \
  --paths resource_paths.json --check-only --hashes
```

After normalizing the inputs, execute these stages in order:

```sh
for stage in local-audit public-proteins drug-background gene-background \
  network-background vehicle-context catalogue-identity catalogue-annotations \
  ensembl-background compile-store functional-inputs functional-review build-routes
do
  python -m cellcanon.resource_prepare "$stage" \
    --config config.json --paths resource_paths.json || exit $?
done
```

| Stage | Immediate input products |
|---|---|
| `local-audit` | Prepared metadata, entity/alias tables, network sources, binding bundles, dataset manifest |
| `public-proteins` | Public reviewed-human UniProt export; no earlier resource output |
| `drug-background` | Drug coverage table from `local-audit` and supplied binding/identity evidence |
| `gene-background` | Gene coverage table from `local-audit`, entity table, UniProt, NCBI human gene file |
| `network-background` | Drug coverage table from `local-audit`, UniProt, normalized network and DrugCentral exports |
| `vehicle-context` | Complete raw preprocessing products and original count matrix |
| `catalogue-identity` | Drug-background identity resolution and source receipts |
| `catalogue-annotations` | Accepted catalogue identity records |
| `ensembl-background` | Gene cards |
| `compile-store` | All preceding resource products and explicit supplier annotations |
| `functional-inputs` | Compiled store, network/gene/control manifests, HGNC cache and functional exports |
| `functional-review` | Seed evidence, unique ligand mapping, functional exports and HGNC cache |
| `build-routes` | Store, network/control manifests, original and reviewed seeds, unsigned witnesses |

All configured roots must be filled before executing resource stages, even
when a particular stage does not read every source. The functional-inputs stage
constructs its seed and route files from the compiled biological store.

Enrichment stages may contact public source services and retain cache receipts.
`public-proteins`, drug/gene/network enrichment, catalogue enrichment, Ensembl
metadata and functional reference review are the public-network stages. They
do not call the language-model service. Cache availability and the requested
public release determine whether a later reproduction can replay them offline.

## 4. Check final knowledge inputs and continue

```sh
python -m cellcanon.resource_preflight --config config.json \
  --paths resource_paths.json --stage prepared --hashes
python -m cellcanon knowledge --config config.json
```

The main knowledge command consumes
`<KNOWLEDGE_ROOT>/store/knowledge.sqlite` and
`<KNOWLEDGE_ROOT>/functional_targets/routes/{routes.sqlite,manifest.json}`.
Continue with the main README's compilation, embedding, training, generation,
Explain and Update commands. Only the later compilation and Explain requests
use the explicitly configured language-model API.

Use matching source snapshots and record their hashes when reproducing an
experiment. Keep generated data, source exports and caches outside the repository.
