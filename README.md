# Tumor Microenvironment Imaging Signatures for Predicting Immunotherapy Response in Melanoma

A single-cohort feasibility analysis testing whether multiplexed ion beam imaging (MIBI)
features of the tumor microenvironment (TME) predict immunotherapy response in melanoma.

**Headline finding:** in this 54-patient cohort, a pre-specified imaging-feature classifier
performed *worse than chance* (LOOCV AUC = 0.33), while a single already-available clinical
variable — stage at diagnosis — significantly outperformed it (LOOCV AUC = 0.60, p = 0.002).
The full writeup, including why that comparison matters, is in [`docs/Manuscript.docx`](docs/Manuscript.docx).

This is reported as a transparent negative/hypothesis-generating result rather than adjusted
until something "worked" — see [`docs/Workflow_Decisions.docx`](docs/Workflow_Decisions.docx)
for a full, ordered log of every analytical decision and why it was made.

## Data

Source: [Vesely & Johnson, *Processed MIBI of tumor microenvironments of 54 melanoma
samples following immunotherapy*, Mendeley Data (2025)](https://doi.org/10.17632/79y7bht7tf.1),
originally presented at SITC 2021. 54 patients, 54,989 segmented cells, 29-marker protein panel,
per-cell spatial coordinates, and RECIST-based response labels.

Raw per-cell CSVs are **not redistributed here** — see [`data/raw/README.md`](data/raw/README.md)
to download them. The derived, patient-level analytical dataset (no per-cell data) is included
at [`results/tables/analytical_dataset.csv`](results/tables/analytical_dataset.csv).

## Pipeline

```
data/raw (cell_protein_data, cell_spatial_data, patient_info)
        │
        ▼
scripts/01_cell_typing.py          per-marker positivity gating → cell lineage per cell
        │
        ▼
scripts/02_feature_engineering.py  per-patient composition, density, PD-L1, proximity (NND),
        │                          spatial clustering (Ripley's K) → analytical_dataset.csv
        ▼
scripts/03_modeling.py             LOOCV random forest vs. clinical-stage baseline,
        │                          univariate group comparisons
        ▼
scripts/04_make_figures.py         publication figures (300 DPI, PNG + PDF)
```

Run the full pipeline:
```bash
pip install -r requirements.txt
# then place the 3 raw CSVs in data/raw/ — see data/raw/README.md
./run_all.sh
```

## Key methodological decisions (see docs/Workflow_Decisions.docx for full detail)

- **Cell typing:** k-means and 2-component GMM gating were tried first and rejected — marker
  distributions are heavily zero-inflated, so both methods just split "zero" from "nonzero"
  rather than finding real biology. Used per-marker 80th-percentile positivity gating with a
  fixed hierarchical lineage rule instead.
- **Missing proximity metrics** (patients with 0 detected CD8⁺ T cells or tumor cells) were
  coded as an explicit "immune desert" category, not imputed as missing — the absence is
  itself biologically meaningful.
- **Validation:** leave-one-out cross-validation (LOOCV), chosen to make honest use of n=54
  without a held-out test set that would leave too few patients to train on.
- **Feature set:** an 11-feature, hypothesis-driven model (matching the four pre-specified
  analysis domains) is reported as the confirmatory result; a 35-feature "kitchen sink" model
  was run only as an exploratory comparison, to avoid overfitting a small cohort by search.

## Results

| Model | LOOCV AUC | Accuracy |
|---|---|---|
| Imaging features (11, pre-specified) | 0.33 | 0.44 |
| Imaging features (35, exploratory) | 0.27–0.36 | — |
| Clinical stage alone | **0.60** | — |

See [`results/tables/table2_feature_comparison.csv`](results/tables/table2_feature_comparison.csv)
for full per-feature statistics and [`results/figures/`](results/figures/) for all plots.

## Repository structure

```
├── data/
│   ├── raw/          # source CSVs (not redistributed — see README there)
│   └── interim/      # cell-level lineage assignments
├── scripts/          # numbered, reproducible pipeline steps
├── results/
│   ├── tables/       # analytical_dataset.csv, feature comparison, model summary
│   └── figures/      # PNG + PDF, 300 DPI
├── docs/
│   ├── Manuscript.docx
│   ├── Workflow_Decisions.docx
│   └── Cover_Letter.docx
└── run_all.sh
```

## Citation

If you use this pipeline or dataset, please cite the original data source:

> Vesely MD, Johnson D. Processed Multiplexed Ion Beam Imaging (MIBI) of tumor
> microenvironments of 54 melanoma samples following immunotherapy. Mendeley Data,
> V1, 2025. https://doi.org/10.17632/79y7bht7tf.1

## License

Code in this repository is released under the MIT License (see `LICENSE`). The underlying
dataset has its own license — check the Mendeley page before redistributing raw data.
