"""
Step 3 — Statistical comparisons and classification.

Input:  results/tables/analytical_dataset.csv
Output: results/tables/table2_feature_comparison.csv
        results/tables/model_summary.txt
        results/tables/roc_fpr.npy, roc_tpr.npy  (for figure 4)

Reports both the pre-specified 11-feature confirmatory model and the
clinical-stage-only baseline comparator (see Workflow_Decisions.docx,
Decisions 7-9).
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

OUT = "results/tables"

CORE_FEATURES = [
    'pct_total_immune', 'pct_CD8_T', 'density_CD8_T_per_1000px2',
    'CD8T_to_Tumor_meanNND', 'CD8T_to_Myeloid_meanNND', 'immune_selfclustering_meanNND',
    'immune_RipleyK_clustering_index', 'PDL1_mean_tumor', 'PDL1_mean_immune',
    'pct_Tumor', 'age_at_dx_tertile', 'no_CD8T_or_Tumor_detected'
]

FEATURE_LABELS = {
    'pct_total_immune': 'Total immune cell fraction (%)',
    'pct_CD8_T': 'CD8+ T-cell fraction (%)',
    'density_CD8_T_per_1000px2': 'CD8+ T-cell density (cells/1000 px^2)',
    'CD8T_to_Tumor_meanNND': 'CD8+T-to-tumor mean NND (px)',
    'CD8T_to_Myeloid_meanNND': 'CD8+T-to-myeloid mean NND (px)',
    'immune_selfclustering_meanNND': 'Immune-immune mean NND (px)',
    'immune_RipleyK_clustering_index': "Immune Ripley's K clustering index",
    'PDL1_mean_tumor': 'Mean tumor PD-L1 expression',
    'PDL1_mean_immune': 'Mean immune-cell PD-L1 expression',
    'PDL1_pct_pos_all': 'PD-L1+ cells, all types (%)',
    'pct_Tumor': 'Tumor cell fraction (%)',
}


def univariate_table(df):
    rows = []
    for col, label in FEATURE_LABELS.items():
        r = df.loc[df.response_binary == 1, col]
        nr = df.loc[df.response_binary == 0, col]
        _, p = mannwhitneyu(r, nr)
        rows.append({
            'Feature': label,
            'Responder median [IQR]': f"{r.median():.3g} [{r.quantile(.25):.3g}\u2013{r.quantile(.75):.3g}]",
            'Non-responder median [IQR]': f"{nr.median():.3g} [{nr.quantile(.25):.3g}\u2013{nr.quantile(.75):.3g}]",
            'p-value': f"{p:.3f}",
        })
    return pd.DataFrame(rows)


def confirmatory_model(df):
    X = df[CORE_FEATURES].values
    y = df['response_binary'].values
    loo = LeaveOneOut()
    rf = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=4, random_state=42)
    probs = cross_val_predict(rf, X, y, cv=loo, method='predict_proba')[:, 1]
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    fpr, tpr, _ = roc_curve(y, probs)
    np.save(f"{OUT}/roc_fpr.npy", fpr)
    np.save(f"{OUT}/roc_tpr.npy", tpr)
    return auc, acc, cm


def stage_baseline(df):
    df = df.copy()
    df['stage_group'] = df['gross_dx_stage'].apply(lambda x: f"Stage {x}" if x != 'UNKNOWN' else 'Unknown')
    order = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4, 'Unknown': np.nan}
    df['stage_ord'] = df['stage_group'].map(order)
    sub = df.dropna(subset=['stage_ord'])

    rho, p_corr = spearmanr(sub['stage_ord'], sub['response_binary'])
    ct = pd.crosstab(df['stage_group'], df['response_binary']).loc[['Stage I', 'Stage II', 'Stage III', 'Stage IV']]
    chi2, p_chi2, _, _ = chi2_contingency(ct)

    X = sub[['stage_ord']].values
    y = sub['response_binary'].values
    probs = cross_val_predict(LogisticRegression(), X, y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
    auc = roc_auc_score(y, probs)
    return rho, p_corr, chi2, p_chi2, auc


def main():
    df = pd.read_csv(f"{OUT}/analytical_dataset.csv")

    tab2 = univariate_table(df)
    tab2.to_csv(f"{OUT}/table2_feature_comparison.csv", index=False)

    auc, acc, cm = confirmatory_model(df)
    rho, p_corr, chi2, p_chi2, stage_auc = stage_baseline(df)

    with open(f"{OUT}/model_summary.txt", "w") as f:
        f.write(f"Confirmatory imaging model (11 features): LOOCV AUC={auc:.3f}, Acc={acc:.3f}\n")
        f.write(f"Confusion matrix:\n{cm}\n\n")
        f.write(f"Clinical stage baseline: LOOCV AUC={stage_auc:.3f}\n")
        f.write(f"Stage vs response: Spearman rho={rho:.3f} (p={p_corr:.4f}), chi2 p={p_chi2:.4f}\n")

    print(open(f"{OUT}/model_summary.txt").read())


if __name__ == "__main__":
    main()
