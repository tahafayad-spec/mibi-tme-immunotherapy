# Raw data

This folder is where the source MIBI CSV files go. They are **not redistributed
in this repository** — download them directly from the original source:

> Vesely MD, Johnson D. *Processed Multiplexed Ion Beam Imaging (MIBI) of tumor
> microenvironments of 54 melanoma samples following immunotherapy.*
> Mendeley Data, V1, 2025. https://doi.org/10.17632/79y7bht7tf.1

After downloading, place these three files here:
- `cell_protein_data.csv`
- `cell_spatial_data.csv`
- `patient_info.csv`

**Important:** do not open these CSVs in Excel, Numbers, or Google Sheets before
placing them here (not even just to preview them). Those apps can silently
corrupt CSV quoting on save/autosave — specifically, one field in
`cell_protein_data.csv` contains a comma inside its own name
(`"HLA class 1 A, B, and C, Na-K-ATPase"`), and spreadsheet apps sometimes
re-quote the entire row instead of just that field, merging all 30 columns
into one. `scripts/01_cell_typing.py` checks for this and will fail with a
clear message if it happens — if you see that error, just re-download a fresh
copy and use it without opening it first.

Check the dataset's license on the Mendeley page before redistributing it
yourself; this repo only redistributes the derived, aggregated per-patient
analytical dataset (`results/tables/analytical_dataset.csv`), not the raw
per-cell data.
