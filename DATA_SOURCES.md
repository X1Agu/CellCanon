# Data sources

## Prepared CellCanon inputs

The [CellCanon dataset](https://huggingface.co/datasets/X1Agu/CellCanon) provides
the three input bundles consumed by the code:

| Bundle | Download | Contents |
| --- | --- | --- |
| Expression and labels | [cellcanon-expression.zip](https://huggingface.co/datasets/X1Agu/CellCanon/resolve/main/cellcanon-expression.zip) | Single-cell expression and controls, continuous responses, DE/DIR labels, sampled queries and cell metadata |
| Metadata | [cellcanon-metadata.zip](https://huggingface.co/datasets/X1Agu/CellCanon/resolve/main/cellcanon-metadata.zip) | Drug structures and annotations, ordered gene panels, splits, schedules and manifests |
| Knowledge | [cellcanon-knowledge.zip](https://huggingface.co/datasets/X1Agu/CellCanon/resolve/main/cellcanon-knowledge.zip) | Knowledge and route databases, evidence records, functional actions and DMSO background context |

Run `python tools/download_data.py --output ./data` to download, verify, and
extract all three bundles, or add `--only metadata` to select a single bundle.
Checksums are recorded in [data_manifest.json](data_manifest.json).
The prepared bundles can be used directly; raw preprocessing is optional.

## Original resources

| Resource | Source | Use |
| --- | --- | --- |
| SciPlex3 | [GEO GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378), series GSE139944 | Raw count coordinates, gene/cell annotations, pData and hash metadata |
| Official SciPlex QC and compound catalogue | [Original dataset repository](https://github.com/cole-trapnell-lab/sci-plex) | Official valid-cell list, compound identities and annotations |
| LINCS L1000 landmark membership | [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742), `GSE92742_Broad_LINCS_gene_info.txt.gz` | Select 978 landmarks; the measured non-HVG intersection yields Extra931 |
| NCBI human gene metadata | [NCBI Gene FTP](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/) | Entrez/Ensembl/symbol mapping; `Homo_sapiens.gene_info.gz` |
| ChEMBL | [Official API](https://www.ebi.ac.uk/chembl/api/data/docs) | Compound identity, assay scope, target binding and functional annotations |
| Guide to Pharmacology | [Official downloads](https://www.guidetopharmacology.org/download.jsp) | Referenced human functional target actions |
| PubChem | [PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) | Structure-matched compound descriptions |
| UniProt | [REST service](https://rest.uniprot.org/) | Reviewed human protein function and GO annotations |
| HGNC | [REST service](https://rest.genenames.org/) | Approved symbol and identity resolution |
| Ensembl | [REST service](https://rest.ensembl.org/) | Human gene identity and descriptions |
| Reactome | [Downloads](https://reactome.org/download-data) | Pathway memberships |
| OmniPath | [Interactions](https://omnipathdb.org/interactions?organisms=9606) | Referenced directed and signed regulatory relations |
| DrugCentral | [Downloads](https://drugcentral.org/download) | Human mechanism-of-action source records |
| Qwen semantic encoder | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Text embeddings from a fixed pretrained encoder; download weights separately |

SciPlex raw filenames are enumerated by `RAW_NAMES` in the preprocessing
script. The official QC list used in the paper comes from dataset-repository revision
`079639c50811dd43a206a779ab2f0199a147c98f`; its SHA-256 is
`fd3735ab48d976d09f35bb510023e1ca9cd391e6f5dfa2bd04e078e66b3c1e9c`.
The Qwen revision is `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

Source versions and preparation requirements are described in
[Input schemas](docs/INPUT_SCHEMAS.md), including the
[evidence scope](docs/INPUT_SCHEMAS.md#evidence-scope).
Dataset resource attribution and reuse terms are listed in the
[dataset source notice](https://huggingface.co/datasets/X1Agu/CellCanon/blob/main/SOURCES.md).
