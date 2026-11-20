# Dual-Scale Temporal Learning with Hyperchaotic and Chaootic TCN-Informer

This is a **reference implementation** of a hybrid TCN–Informer model and
sequence-derived initialization from 3D Chen and 4D hyperchaotic systems.
The source comes from consolidated notebook components, with example settings
separated into JSON configuration files. **It is not an exact reproduction
package for the paper's reported experiments.** The paper-specific numerical
configurations, per-run manifests, datasets, and trained checkpoints are not included.

## Structure

- `models/initialization.py` — trajectory generation, transient removal, z-score normalization,
  global scaling, layer-dependent scaling, and selected weight assignment
- `models/` — Informer and hybrid TCN–Informer architectures
- `data/`, `utils/` — ETT data loading and utilities
- `exp/` — training and evaluation pipeline
- `main_informer.py` — training command-line entry point
- `configs/example_training.json` — **small illustrative example**
- `configs/initialization_template.json` — required fields for dynamic-source initialization;
  `null` placeholders must be replaced with suitable values
- `scripts/smoke_test.py` — forward-pass test without datasets

## Installation

For Windows PowerShell, from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.smoke_test
```

For Linux/macOS substitute `python` and the local virtual environment path.
See the official PyTorch installation instructions when GPU/CUDA support is required.

## Dataset

Download ETT CSV data from the [ETDataset repository](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)
and put `ETTm1.csv` (or other ETT datasets) under `data/ETT/`.
CSV files and trained checkpoints are excluded from the repository.

## Example training 

```powershell
.\.venv\Scripts\python.exe main_informer.py --config configs/example_training.json
```

The example runs **standard initialization** with deliberately small model/training settings.
It is intended for demonstrating how to run the software, not matching any specific result.
Available overrides include `--model`, `--init_mode`, `--data`, `--data_path`, `--pred_len`,
`--seed`, and `--use_gpu`.

## Chaotic and hyperchaotic initialization

Copy `configs/initialization_template.json` to a file such as `init_config.json`
and enter the complete required coefficients, initial conditions, integration step,
burn-in, and global scaling parameter. The template has **no secret defaults**;
with missing values, the program raises a clear exception.

```powershell
.\.venv\Scripts\python.exe main_informer.py --config configs/example_training.json --init_mode chaotic --init_config init_config.json
```

Likewise, use `--init_mode hyperchaotic` for the four-dimensional source.
Do not use the example training settings to describe a replication of published metrics.

## Reproducibility scope and licensing

All methodological steps implemented by this release are directly available for review.
Exact numerical results from the manuscript require the **actual experimental
configuration files**, corresponding dataset versions, complete seed runs, and
hardware/software environment. These are **not included** here; no reproducibility
claim is made for the manuscript's reported MSE/MAE values. If a journal requires
those materials, supply them through an approved supplementary or reviewer-access process.

No software license is included at this stage. Users should seek permission for
reuse beyond rights automatically granted by law.

## Citation

If you use this repository, its implementation, or the proposed chaotic and hyperchaotic initialization approach in your research, please cite our paper:

**Ahmed, W., Akhtar, M. W., Bashir, Z., & Malik, M. G. A. (2026).** Chaotic and hyperchaotic weight initialization with dual-branch temporal learning in informer for long-term time-series forecasting. *Evolutionary Intelligence*.

**DOI:** https://doi.org/10.1007/s12065-026-01291-3

**Read the full-text article:** [Springer Nature SharedIt – Read Online](https://rdcu.be/rdZ8T6TxR3za)

### BibTeX

```bibtex
@article{Ahmed2026,
  author  = {Ahmed, Waqar and Akhtar, Muhammad Waseem and Bashir, Zia and Malik, M. G. Abbas},
  title   = {Chaotic and hyperchaotic weight initialization with dual-branch temporal learning in informer for long-term time-series forecasting},
  journal = {Evolutionary Intelligence},
  year    = {2026},
  volume  = {20},
  number  = {1},
  doi     = {10.1007/s12065-026-01291-3}
}
```
