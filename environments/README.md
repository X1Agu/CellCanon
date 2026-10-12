# Environment and dependencies

CellCanon uses Python 3.11 on Linux x86_64. Model fitting and text embedding
require an NVIDIA GPU with a driver compatible with CUDA 13.0. Raw data
preprocessing requires a C++17 compiler and zlib development headers.

## Versions

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.11.0, CUDA 13.0 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.7.2 |
| joblib | 1.5.3 |
| h5py | 3.16.0 |
| AnnData | 0.12.10 |
| PyArrow | 23.0.1 |
| scikit-misc | 0.5.2 |
| RDKit | 2025.9.6 |
| Transformers | 4.57.6 |
| requests | 2.32.5 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| setuptools / wheel | 81.0.0 / 0.46.3 |

Runtime dependencies are pinned in [requirements.txt](../requirements.txt).
[cuda-py311.constraints.txt](cuda-py311.constraints.txt) specifies transitive
versions. [cpu-py311.lock.txt](cpu-py311.lock.txt) is used by CI for offline tests.

## Install

Run these commands from the repository root. The environment name `cellcanon`
can be replaced with a name of your choice.

```sh
conda create -n cellcanon python=3.11.15 pip
conda activate cellcanon
python -m pip install 'setuptools==81.0.0' 'wheel==0.46.3'
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -c environments/cuda-py311.constraints.txt -r requirements.txt
python -m pip install --no-deps .
python -m pip check
python tools/check_environment.py --include-transitive
python tools/check_install.py
```

`check_environment.py` reports dependency differences and CUDA availability.
It does not install packages. See the
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)
for the CUDA wheel distribution.

## Compute requirements

| Stage | Device or service |
| --- | --- |
| Preprocessing, metadata, knowledge preparation and feature projection | CPU |
| Evidence compilation and Explain inference | User-configured API |
| Text embedding with the pretrained encoder | GPU, CUDA FP16 |
| Atlas/MLP and DE/DIR head fitting | GPU |
| Saved classifier coefficient readout | CPU |
| Dose, detection, state and Extra931 auxiliary fitting | CPU |
| Cell generation and expression testing | CPU |
| Five-fold Update parent fitting | GPU |
| Update residual/tree/count fitting and observation application | CPU |
| Evaluation and evidence/state checks | CPU |

Run these stages in the same environment. Some training functions support a
CPU fallback; use GPU fitting for the paper's configuration. Changing the
device or library versions can change numerical results.

See [Data sources](../DATA_SOURCES.md) and [Preprocessing](../PREPROCESSING.md)
for inputs, and [API configuration](../docs/MODEL_SERVICE.md) for service setup.
