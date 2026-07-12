"""
Step 4 — Generate manuscript figures (300 DPI, PNG + PDF, no embedded titles).

Input:  data/raw/*, data/interim/cell_types.csv, results/tables/analytical_dataset.csv,
        results/tables/roc_fpr.npy, roc_tpr.npy
Output: results/figures/fig1_workflow.{png,pdf}
        results/figures/fig2_spatial_map.{png,pdf}
        results/figures/fig3_feature_boxplots.{png,pdf}
        results/figures/fig4_roc_curve.{png,pdf}
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RAW = "data/raw"
INTERIM = "data/interim"
TABLES = "results/tables"
FIGDIR = "results/figures"

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    ax.set_xlim(0, 10)
    ax.axis('off')
    steps = [
        (0.3, 4.6, "MIBI single-cell data\n(54 patients, 54,989 cells,\n29 protein markers + xy coordinates)"),
        (0.3, 3.3, "Lineage gating\n(per-marker 80th-percentile\npositivity + hierarchical rules)"),
        (0.3, 2.0, "Per-patient feature engineering\n(composition, density, PD-L1,\nNND proximity, Ripley's K)"),
        (0.3, 0.7, "Locked analytical dataset\n(54 patients x 44 features)"),
    ]
    w, h = 9.4, 0.95
    for x, y, txt in steps:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                                     linewidth=1.1, edgecolor='#333333', facecolor='#eef3f8'))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=8)
    for (x1, y1, _), (x2, y2, _) in zip(steps[:-1], steps[1:]):
        ax.annotate('', xy=(x2 + w / 2, y2 + h), xytext=(x1 + w / 2, y1),
                     arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.2))
    ax.add_patch(FancyBboxPatch((0.3, -1.0), w, 0.85, boxstyle="round,pad=0.05,rounding_size=0.08",
                                 linewidth=1.1, edgecolor='#333333', facecolor='#f7ece1'))
    ax.text(0.3 + w / 2, -1.0 + 0.425,
            "Leave-one-out cross-validated\nrandom-forest classification + univariate group comparisons",
            ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(0.3 + w / 2, -0.15), xytext=(0.3 + w / 2, 0.7),
                arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.2))
    ax.set_ylim(-1.3, 5.8)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig1_workflow.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGDIR}/fig1_workflow.pdf", bbox_inches='tight')
    plt.close()


def fig2_spatial_map(patient_id='R10C2'):
    prot = pd.read_csv(f"{RAW}/cell_protein_data.csv")
    spat = pd.read_csv(f"{RAW}/cell_spatial_data.csv")
    cell_types = pd.read_csv(f"{INTERIM}/cell_types.csv")
    df = pd.concat([spat, prot.drop(columns='id'), cell_types['cell_type']], axis=1)
    g = df[df.id == patient_id]

    color_map = {
        'Tumor': '#d62728', 'CD8_T': '#1f77b4', 'CD4_T': '#17becf', 'Treg': '#9467bd',
        'B_cell': '#8c564b', 'NK': '#e377c2', 'Myeloid': '#ff7f0e', 'Other_T': '#7f7f7f',
        'Other_immune': '#bcbd22', 'Endothelial': '#2ca02c', 'Stromal': '#a1a1a1',
        'Lymphatic': '#98df8a', 'Unclassified': '#e0e0e0',
    }
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    for ct, sub in g.groupby('cell_type'):
        ax.scatter(sub.x_centroid, sub.y_centroid, s=6, c=color_map.get(ct, '#000000'), label=ct, linewidths=0)
    ax.set_xlabel('X centroid (px)')
    ax.set_ylabel('Y centroid (px)')
    ax.set_aspect('equal')
    ax.legend(markerscale=2, fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig2_spatial_map.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGDIR}/fig2_spatial_map.pdf", bbox_inches='tight')
    plt.close()


def fig3_boxplots():
    df = pd.read_csv(f"{TABLES}/analytical_dataset.csv")
    feats = [
        ('pct_CD8_T', 'CD8+ T-cell fraction (%)'),
        ('CD8T_to_Tumor_meanNND', 'CD8+T-tumor mean NND (px)'),
        ('immune_RipleyK_clustering_index', "Immune Ripley's K clustering index"),
        ('PDL1_mean_tumor', 'Mean tumor-cell PD-L1 expression'),
    ]
    labels = {0: 'Non-responder\n(PD/SD)', 1: 'Responder\n(CR/PR)'}
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), dpi=300)
    for ax, (col, ylab) in zip(axes, feats):
        data = [df.loc[df.response_binary == g, col].values for g in [0, 1]]
        bp = ax.boxplot(data, tick_labels=[labels[0], labels[1]], widths=0.55, showfliers=False, patch_artist=True)
        for patch, c in zip(bp['boxes'], ['#bdbdbd', '#4292c6']):
            patch.set_facecolor(c)
        for gpos, vals in zip([1, 2], data):
            jitter = np.random.default_rng(0).normal(gpos, 0.05, size=len(vals))
            ax.scatter(jitter, vals, s=8, color='black', alpha=0.5, zorder=3)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(axis='x', labelsize=7.5)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig3_feature_boxplots.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGDIR}/fig3_feature_boxplots.pdf", bbox_inches='tight')
    plt.close()


def fig4_roc():
    fpr = np.load(f"{TABLES}/roc_fpr.npy")
    tpr = np.load(f"{TABLES}/roc_tpr.npy")
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    ax.plot(fpr, tpr, color='#08519c', lw=1.8, label='LOOCV RF (AUC = 0.33)')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance (AUC = 0.50)')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig4_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGDIR}/fig4_roc_curve.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fig1_workflow()
    fig2_spatial_map()
    fig3_boxplots()
    fig4_roc()
    print("All figures written to", FIGDIR)
