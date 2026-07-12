"""
Step 1 — Cell-lineage assignment.

Input:  data/raw/cell_protein_data.csv, data/raw/cell_spatial_data.csv
Output: data/interim/cell_types.csv  (id, cell_type per cell)

Method: per-marker 80th-percentile positivity gating + hierarchical lineage
rules. See docs/Workflow_Decisions.docx (Decision 2) for why k-means and
GMM gating were tried first and rejected.
"""
import pandas as pd
import numpy as np

RAW = "data/raw"
OUT = "data/interim"

LINEAGE_MARKERS = ['CD45', 'CD3', 'CD4', 'CD8', 'FOXP3', 'CD20', 'CD68', 'CD163',
                    'CD11b', 'CD11c', 'CD14', 'CD56', 'SOX10', 'CD31', 'SMA', 'Podoplanin']

POSITIVITY_PERCENTILE = 0.80  # documented, fixed a priori — not tuned against outcome


def classify(row):
    if row['SOX10'] and not row['CD45']:
        return 'Tumor'
    if row['CD45']:
        if row['CD3']:
            if row['FOXP3']:
                return 'Treg'
            if row['CD8']:
                return 'CD8_T'
            if row['CD4']:
                return 'CD4_T'
            return 'Other_T'
        if row['CD20']:
            return 'B_cell'
        if row['CD56']:
            return 'NK'
        if row['CD68'] or row['CD163'] or row['CD14'] or row['CD11b'] or row['CD11c']:
            return 'Myeloid'
        return 'Other_immune'
    if row['CD31']:
        return 'Endothelial'
    if row['Podoplanin']:
        return 'Lymphatic'
    if row['SMA']:
        return 'Stromal'
    return 'Unclassified'


def main():
    prot = pd.read_csv(f"{RAW}/cell_protein_data.csv")

    thresholds = prot[LINEAGE_MARKERS].quantile(POSITIVITY_PERCENTILE)
    pos = prot[LINEAGE_MARKERS] > thresholds

    prot['cell_type'] = pos.apply(classify, axis=1).values

    print("Cell-type composition (%):")
    print((prot['cell_type'].value_counts(normalize=True) * 100).round(2))

    prot[['id', 'cell_type']].to_csv(f"{OUT}/cell_types.csv", index=False)
    thresholds.to_csv(f"{OUT}/gating_thresholds.csv")


if __name__ == "__main__":
    main()
