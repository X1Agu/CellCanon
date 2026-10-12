# Input contracts and source preparation

The input checker is offline. It reports logical names such as
`network_source_root/signed_network.parquet`, missing columns, missing JSON
keys, and optional SHA-256 mismatches.

```sh
python -m cellcanon.resource_preflight --config config.json \
  --paths resource_paths.json --stage all --hashes
python -m cellcanon.resource_prepare local-audit --config config.json \
  --paths resource_paths.json --check-only --hashes
```

Exit status is 0 for present prerequisites and 2 for missing/invalid inputs.
`all` inventories both source inputs and derived resources; missing downstream
products are expected before their stages have run. For a single stage, only
that stage's immediate prerequisites are inspected.

## Paths and formats

Paths in either configuration file are resolved relative to that configuration
file. Set all five keys in `resource_paths.example.json` before executing the
resource preparation stages:

| Setting | Contents |
|---|---|
| `normalized_sources_root` | Two curated binding bundles and explicit supplier annotations |
| `entity_source_root` | Entity and alias Parquet registries with `prepare_manifest.json` |
| `network_source_root` | Reactome memberships and normalized OmniPath/DoRothEA edges |
| `functional_source_root` | Guide to Pharmacology ligand/interaction CSV exports and manifest |
| `drugcentral_source_root` | DrugCentral structure/interaction TSV exports and manifest |

The main configuration also supplies `data_root`, `metadata_root`, and
`knowledge_root`. The latter is a new output directory. Resource functions
write under the configured output roots; use separate directories for each run.

CSV inputs have a header, UTF-8 encoding, and no index column. TSV inputs use
literal tabs. Parquet preserves list, Boolean, nullable integer and floating
columns. JSON lists contain objects, except where a nested container is
explicitly documented. IDs are strings unless a source format specifies an
integer. Missing annotations remain missing.

## Evidence scope

Knowledge inputs describe compound and gene identities, binding, functional
actions, regulatory relations, pathway membership and DMSO expression. Binding
does not determine an action sign, and pathway membership is an association.
Explain uses these records to assess source support and construct background
route hypotheses; establishing a causal effect in a treated condition requires
condition-specific experimental evidence.

## Expression data and compound metadata

The preprocessing script requires these original SciPlex files under its
`--raw-root`:

- `GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz`
- `GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz`
- `GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz`
- `GSM4150378_sciPlex3_pData.txt.gz`
- `GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz`

Provide the official valid-cell list, LINCS gene information, and NCBI human
gene information using the script's explicit flags. No count matrix is
included in this repository. The exact official valid-cell list hash and gene
panel checks are enforced by the code.

`compound_registry` is CSV/Parquet with unique `drug` names and columns
`drug`, `smiles`, `target`, `pathway`. Here `drug` is the public catalogue name
used by the deterministic metadata-name mapper. `target` may contain
comma/semicolon/pipe-separated target labels. `python -m cellcanon metadata --config config.json`
produces:

| File under `metadata_root` | Required use |
|---|---|
| `drug_metadata.parquet` | Catalogue ID `drug`, `split`, `canonical_name`, `legacy_drug`, `canonical_smiles`, fingerprints, `target_roles`, `drug_moa` |
| `gene_order.parquet` | Ordered 2,000 genes with `gene`, `gene_symbol`, gene index |
| `extra_gene_order.parquet` | Ordered 931 additional genes with the same schema |
| `prepare_manifest.json` | `outputs` maps the three table names to filenames; `output_sha256` maps the same names to hashes |

The training, validation and test drug lists are stored in
[`src/cellcanon/protocols/split.json`](../src/cellcanon/protocols/split.json).
The `legacy_drug` column retains the matched name from the supplied compound
registry.

`vehicle-context` additionally needs preprocessing's `build_manifest.json`,
`metadata/{valid_cells_pdata,human_features,all_features}.parquet`,
`metadata/feature_to_human.i32`, `intermediate/library_size.u64`, and the raw
matrix recorded in the build manifest. This stage reads DMSO control cells
and builds the all-human expression context. It requires a C++17 compiler and
zlib. It does not use treated-cell response labels.

`gene-background` expects the NCBI file at
`<DATA_PARENT>/reference/ncbi_gene/Homo_sapiens.gene_info.gz`, where
`<DATA_PARENT>` is the parent of `data_root`. Preserve the source columns
`#tax_id`, `GeneID`, `Symbol`, `Synonyms`, `dbXrefs`, and `description`.

## Entity and alias context

To build entity and alias tables from public sources, use the HGNC approved
records JSON (`response.docs`, as returned by
`https://rest.genenames.org/fetch/status/Approved`) and a public compound table:

```sh
python -m cellcanon.input_convert public-entities \
  --hgnc '<HGNC_APPROVED_JSON>' \
  --compounds '<COMPOUND_TABLE_CSV_OR_PARQUET>' \
  --output '<ENTITY_ALIAS_TABLE_ROOT>'
```

The compound table requires catalogue-ID `drug` and `name`; `canonical_name`
is also accepted in place of `name`. Optional `description` is copied
verbatim. The main `metadata/drug_metadata.parquet` output can supply these
identities. For HGNC, `status`, `symbol`, `name`, `ensembl_gene_id`,
`uniprot_ids`, `alias_symbol`, and `prev_symbol` are used when present. Only
approved records are accepted. Descriptions provide identity context.

Alternatively, import explicitly sourced normalized tables:

```sh
python -m cellcanon.input_convert entities \
  --entities '<ENTITY_TABLE_CSV_OR_PARQUET>' \
  --aliases '<ALIAS_TABLE_CSV_OR_PARQUET>' \
  --output '<ENTITY_ALIAS_TABLE_ROOT>'
```

| Output | Required columns and meaning |
|---|---|
| `entity_registry.parquet` | `entity_type` (`drug`/`gene`), `canonical_id`, `entity_text`, `text_source`, `ensembl_id`, `uniprot_accession` |
| `alias_registry.parquet` | `entity_type`, `alias`, `canonical_id` |

Entity `(entity_type,canonical_id)` keys are unique. Aliases must reference a
known entity. Identifier fields may be scalars or JSON-encoded lists; list
strings must parse as lists. `prepare_manifest.json` is generated with output
filenames and hashes.

## Reactome and signed regulatory network

Provide the human Reactome v97 GMT ZIP and the public OmniPath interaction and
DoRothEA A/B/C TSV exports. Source downloads remain external. Use the ZIP's
actual member filename:

```sh
python -m cellcanon.input_convert network \
  --reactome-zip '<REACTOME_V97_GMT_ZIP>' --member '<GMT_MEMBER_NAME>' \
  --omnipath '<OMNIPATH_TSV>' --dorothea '<DOROTHEA_ABC_TSV>' \
  --source-release v97 \
  --output '<PATHWAY_SIGNED_NETWORK_TABLE_ROOT>'
```

Each GMT line is `pathway_name<TAB>pathway_id<TAB>gene_symbol...`.
The ZIP must contain the supplied member. Since GMT has no release header,
the manifest records the required `--source-release v97` declaration and input
hashes. OmniPath TSV fields are
`source`, `target`, `source_genesymbol`, `target_genesymbol`, `is_directed`,
`is_stimulation`, `is_inhibition`, `consensus_direction`,
`consensus_stimulation`, `consensus_inhibition`, `sources`, `references`,
`curation_effort`; DoRothEA additionally requires `dorothea_level`.
Every DoRothEA row must have at least one A/B/C confidence assignment; mixed lists such as `A;D` remain valid, but D/E-only and missing assignments fail validation. Boolean fields are source Boolean/`true` values. Sources and references use
semicolon-separated strings. Preserve source row ordering, which contributes
to stable edge IDs. The two TSVs must be restricted to human records
and the stated source collections before conversion.

Outputs are `reactome_pathway_memberships.parquet`, `signed_network.parquet`,
and `network_manifest.json`. The latter has an `outputs` list whose
records contain `name`, relative filename, and `sha256`.

Membership columns are `pathway_id`, `pathway_name`, `gene_symbol`, `species`,
`source_release`. Signed edges include `edge_id`, source/target UniProt IDs and
gene symbols, source and consensus Boolean fields, `sign`, `sign_status`,
`reference_count`, `references_json`, `sources_json`, and `interaction_type`.
A simultaneous positive and negative consensus remains a missing sign with
`sign_status="conflict"`; absent signs remain unknown.
OmniPath interactions are `post_translational`, while DoRothEA
edges are `transcriptional`. Only the latter can be a transcription endpoint.

The later `network-background` stage verifies human identities against HGNC
and reviewed UniProt, resolves typed complex nodes, and retrieves the public
Ensembl-to-Reactome membership export. GO terms use UniProt's format
`term name [GO:0000000]`; terms with invalid identifiers are excluded.

## ChEMBL binding exports

The offline converter accepts complete exported records from ChEMBL 37;
responses with pagination must be assembled before conversion. JSON may be a
list or contain its matching `activities`, `assays`, `targets`, or `molecules`
list. No API key or model inference is involved.

```sh
python -m cellcanon.input_convert bindings \
  --mapping '<DRUG_MOLECULE_MAPPING_CSV>' \
  --activities '<CHEMBL_ACTIVITIES_JSON>' --assays '<CHEMBL_ASSAYS_JSON>' \
  --targets '<CHEMBL_TARGETS_JSON>' --molecules '<CHEMBL_MOLECULES_JSON>' \
  --output '<NORMALIZED_PUBLIC_SOURCE_ROOT>'
```

Mapping columns are catalogue-ID `drug`, `split` (`support`/`validation`/`test`),
`molecule_chembl_id`, and `canonical_smiles`. Each `(drug,molecule_chembl_id)`
occurs once. Include every retained main-cohort drug. A drug with no mapped
molecule can have a single explicitly empty molecule-ID record; it receives
an empty evidence list and is subsequently eligible for public identity
retrieval. A nonempty mapping is accepted only if the supplied molecule's
full standard-parent structure agrees with the native compound structure.

`model/chembl_binding_annotations.py::curate` defines the source contract:
standardized, nonduplicate binding Ki/Kd/IC50 measurements; explicit human
activity, assay and single-protein target; direct assay assignment with
confidence 9; one resolvable gene component; no annotated sequence variant.
Relations `=`, `<`, `<=`, `>`, `>=` are retained. M, mM, uM/µM/μM, nM and pM
are converted to nM without collapsing censoring intervals. The manifest
records exclusion counts by reason.

The converter writes two bundles:

| Directory under `normalized_sources_root` | Files |
|---|---|
| `sciplex_chembl_binding_annotations_v1` | `binding_evidence.parquet`, `drug_bindings.json`, `binding_manifest.json` for support/validation |
| `sciplex_binding_query_annotations_v1` | `binding_evidence.parquet`, `query_drug_bindings.json`, `query_binding_manifest.json` for test |

Binding rows retain drug/split, activity/molecule/assay/target IDs,
`target_gene_symbol`, `target_uniprot`, `measurement_type`, `relation`,
`value_nM`, reported units/value, potency bounds/inclusivity, assignment
confidence, assay context and source release. Descriptor records contain
`drug`, `annotation_reference_drugs`, `measured_genes`, `active_genes_1uM`,
`binding_records`. Each bundle manifest hashes its table and descriptors.

## Guide to Pharmacology, DrugCentral, and supplier evidence

```sh
python -m cellcanon.input_convert functional \
  --ligands '<GTOPDB_2026_2_LIGANDS_CSV>' \
  --interactions '<GTOPDB_2026_2_INTERACTIONS_CSV>' \
  --source-release 2026.2 \
  --output '<GUIDE_TO_PHARMACOLOGY_EXPORT_ROOT>'
python -m cellcanon.input_convert drugcentral \
  --structures '<DRUGCENTRAL_STRUCTURES_TSV>' \
  --interactions '<DRUGCENTRAL_INTERACTIONS_TSV>' \
  --source-release 2021-09-01 \
  --output '<DRUGCENTRAL_EXPORT_ROOT>'
```

Guide to Pharmacology CSVs retain the initial version/comment line, then the
header. Both headers must explicitly state `GtoPdb Version: 2026.2`, matching
the required declaration. Ligands require `Ligand ID`, `ChEMBL ID`. Interactions require
`Ligand ID`, `Target Species`, `Target Gene Symbol`, `Target Subunit IDs`,
`Type`, `Action`, `Primary Target`, `PubMed ID`, `Webpage URLs`. Unique exact
ChEMBL-ID mapping supplies candidate ligands. Functional review admits only
unambiguous human single-gene primary targets with an accepted action pair
and a resolved bibliographic reference. Multisubunit ambiguity and action
conflicts remain explicit exclusions.

DrugCentral structures require `ID`, `SMILES`, `INN`. Interaction TSV fields
include `STRUCT_ID`, `ORGANISM`, `MOA`, `GENE`, `DRUG_NAME`, `ACTION_TYPE`,
`TARGET_NAME`, `MOA_SOURCE_URL`, `ACT_SOURCE_URL`. Use the 2021-09-01 export;
downstream preparation selects human curated-MOA records. The converter copies
the supplied files and records their hashes. Since these TSVs have no release
header, their version is supplied through `--source-release 2021-09-01`.
Other version declarations are rejected.

Supplier overrides are an explicitly sourced JSON file at
`<NORMALIZED_PUBLIC_SOURCE_ROOT>/configs/metadata/sciplex_background_source_overrides.json`:
`{"records": [...]}`. Each record requires `drug`, `catalog_number`,
`source_smiles`, `statement`, `source_url`, `reported_target_symbol`.
The catalogue ID and standard-parent structure must agree. The source snapshot
used in the paper includes catalogue S8481's SRT3025 annotation from
`https://www.selleckchem.com/products/srt3025.html`, with SIRT1 as a
supplier-reported target. Retain its statement and source URL. An empty
`records` list supplies no supplier annotations.

## Derived functional inputs and route store

`functional-inputs` builds `knowledge_root/functional_audit` from the compiled
store and supplied functional exports. Its outputs are:

| File | Contents |
|---|---|
| `seed_evidence.json` | Drug, gene symbol/node, evidence tier/ID/URL, action/sign, identity tier |
| `gtopdb_candidate_inventory.json` | `mapping` records with drug/split, candidate ligand IDs, exact-unique flag, identity tier |
| `route_witnesses.jsonl` | Cited unsigned direct-target, directed-network and pathway-association witnesses |
| `manifest.json` | Source and output hashes, preparation status and actual counts |

Every route retains source evidence, node sequence, edge IDs, pathway
references and endpoint identity. Searches are cycle-free and limited to
three network edges. `functional-review` and `build-routes` use the same
route tiers, distinguishing signed regulatory paths from pathway associations.

Final `knowledge` inputs are `knowledge_root/store/knowledge.sqlite` and
`knowledge_root/functional_targets/routes/{routes.sqlite,manifest.json}`.
Preflight checks the main store tables and required fields. It does not
replace record-level identity, sign, or source checks performed during build.

## Online versus offline work

`input_convert`, `resource_preflight`, `local-audit`, `vehicle-context`,
`compile-store`, `functional-inputs`, and `build-routes` use supplied local
files. `public-proteins`, drug/gene/network enrichment, catalogue enrichment,
Ensembl metadata, and functional reference review may call public source
services and cache receipts. None of these resource stages invokes the
language-model API. The later main `compile` and `explain` commands do.
Public services and licenses can change; retain the cited release exports
when reproducing a fixed resource snapshot. Data, source snapshots, caches,
databases, and credentials remain outside the repository.
