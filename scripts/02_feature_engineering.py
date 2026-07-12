"""
Step 2 — Per-patient feature engineering.

Input:  data/raw/{cell_protein_data,cell_spatial_data,patient_info}.csv
        data/interim/cell_types.csv
Output: results/tables/analytical_dataset.csv  (locked, 54 patients x 44 features)

Builds: cell-type composition & density, PD-L1 expression summaries,
CD8T-to-tumor / CD8T-to-myeloid nearest-neighbor proximity, and an
immune-cell Ripley's K clustering index. See docs/Workflow_Decisions.docx
(Decisions 3-6) for the reasoning behind each choice.
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree, ConvexHull

RAW = "data/raw"
INTERIM = "data/interim"
OUT = "results/tables"

IMMUNE_TYPES = ['CD8_T', 'CD4_T', 'Treg', 'Other_T', 'B_cell', 'NK', 'Myeloid', 'Other_immune']
CELL_TYPES = ['Tumor', 'CD8_T', 'CD4_T', 'Treg', 'B_cell', 'NK', 'Myeloid',
              'Endothelial', 'Stromal', 'Lymphatic', 'Unclassified']

RIPLEY_RADIUS_PX = 50  # fixed a priori, ~2x median inter-cell spacing


def per_patient_features(df):
    feats = []
    for pid, g in df.groupby('id'):
        n_cells = len(g)
        pts = g[['x_centroid', 'y_centroid']].values
        try:
            hull_area = ConvexHull(pts).volume
        except Exception:
            hull_area = np.nan

        comp = g['cell_type'].value_counts(normalize=True)
        row = {'id': pid, 'n_cells': n_cells, 'tissue_area_px2': hull_area}
        for ct in CELL_TYPES:
            row[f'pct_{ct}'] = comp.get(ct, 0.0) * 100
            row[f'density_{ct}_per_1000px2'] = (
                (comp.get(ct, 0.0) * n_cells) / hull_area * 1000 if hull_area else np.nan
            )
        row['pct_total_immune'] = sum(comp.get(ct, 0.0) for ct in IMMUNE_TYPES) * 100

        row['PDL1_mean_all'] = g['PD-L1'].mean()
        row['PDL1_pct_pos_all'] = g['PD-L1_pos'].mean() * 100
        tumor_mask = g['cell_type'] == 'Tumor'
        immune_mask = g['cell_type'].isin(IMMUNE_TYPES)
        row['PDL1_mean_tumor'] = g.loc[tumor_mask, 'PD-L1'].mean() if tumor_mask.sum() > 0 else np.nan
        row['PDL1_mean_immune'] = g.loc[immune_mask, 'PD-L1'].mean() if immune_mask.sum() > 0 else np.nan

        cd8 = g[g['cell_type'] == 'CD8_T'][['x_centroid', 'y_centroid']].values
        tum = g[g['cell_type'] == 'Tumor'][['x_centroid', 'y_centroid']].values
        mye = g[g['cell_type'] == 'Myeloid'][['x_centroid', 'y_centroid']].values

        row['CD8T_to_Tumor_meanNND'] = (
            cKDTree(tum).query(cd8)[0].mean() if len(cd8) > 0 and len(tum) > 0 else np.nan
        )
        row['CD8T_to_Myeloid_meanNND'] = (
            cKDTree(mye).query(cd8)[0].mean() if len(cd8) > 0 and len(mye) > 0 else np.nan
        )

        imm_pts = g[g['cell_type'].isin(IMMUNE_TYPES)][['x_centroid', 'y_centroid']].values
        if len(imm_pts) > 5:
            d3, _ = cKDTree(imm_pts).query(imm_pts, k=2)
            row['immune_selfclustering_meanNND'] = d3[:, 1].mean()
        else:
            row['immune_selfclustering_meanNND'] = np.nan

        n = len(imm_pts)
        if n > 10 and hull_area and hull_area > 0:
            pairs = cKDTree(imm_pts).query_pairs(r=RIPLEY_RADIUS_PX)
            K_obs = (hull_area / (n ** 2)) * (len(pairs) * 2)
            K_csr = np.pi * RIPLEY_RADIUS_PX ** 2
            row['immune_RipleyK_clustering_index'] = (K_obs / K_csr) - 1
        else:
            row['immune_RipleyK_clustering_index'] = np.nan

        feats.append(row)
    return pd.DataFrame(feats)


def apply_immune_desert_coding(df):
    """0 CD8T / 0 tumor cells detected is biologically meaningful, not missing (Decision 5)."""
    for col, flag in [('CD8T_to_Tumor_meanNND', 'no_CD8T_or_Tumor_detected'),
                       ('CD8T_to_Myeloid_meanNND', 'no_CD8T_or_Myeloid_detected'),
                       ('immune_selfclustering_meanNND', 'sparse_immune_lt5cells')]:
        df[flag] = df[col].isna().astype(int)
        sentinel = df[col].max(skipna=True) * 1.2
        df[col] = df[col].fillna(sentinel)

    df['no_tumor_detected'] = (df['pct_Tumor'] == 0).astype(int)
    df['PDL1_mean_tumor'] = df['PDL1_mean_tumor'].fillna(df['PDL1_mean_tumor'].mean())

    df['ripley_missing_flag'] = df['immune_RipleyK_clustering_index'].isna().astype(int)
    df['immune_RipleyK_clustering_index'] = df['immune_RipleyK_clustering_index'].fillna(
        df['immune_RipleyK_clustering_index'].mean()
    )
    return df


def main():
    prot = pd.read_csv(f"{RAW}/cell_protein_data.csv")
    spat = pd.read_csv(f"{RAW}/cell_spatial_data.csv")
    pinfo = pd.read_csv(f"{RAW}/patient_info.csv")
    cell_types = pd.read_csv(f"{INTERIM}/cell_types.csv")

    df = pd.concat([spat, prot.drop(columns='id'), cell_types['cell_type']], axis=1)
    df['PD-L1_pos'] = df['PD-L1'] > df['PD-L1'].quantile(0.80)

    feat_df = per_patient_features(df)
    feat_df = apply_immune_desert_coding(feat_df)

    merged = feat_df.merge(pinfo, on='id', how='inner')
    assert merged.isna().sum().sum() == 0, "unexpected missing values remain"

    merged.to_csv(f"{OUT}/analytical_dataset.csv", index=False)
    print(f"Wrote {OUT}/analytical_dataset.csv  shape={merged.shape}")


if __name__ == "__main__":
    main()
