"""
Task A1 — TCGA-HNSC: miR-125a and miR-200a expression in tumour vs normal tissue
FYP Group 18, E21, University of Peradeniya
"""

import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os

# ── 0. Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "../data"
FIG_DIR    = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

MIRNA_FILE  = os.path.join(DATA_DIR, "TCGA.HNSC.sampleMap_miRNA_HiSeq_gene.gz")
CLIN_FILE   = os.path.join(DATA_DIR, "TCGA.HNSC.sampleMap_HNSC_clinicalMatrix")

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading clinical data...")
clin = pd.read_csv(CLIN_FILE, sep="\t", index_col=0, low_memory=False)
clin = clin[["sample_type"]].copy()
print(clin["sample_type"].value_counts())

print("\nLoading miRNA expression data...")
with gzip.open(MIRNA_FILE, "rt") as f:
    mirna = pd.read_csv(f, sep="\t", index_col=0)

# mirna: rows = miRNA identifiers (MIMAT…), columns = sample IDs
print(f"miRNA matrix: {mirna.shape[0]} miRNAs x {mirna.shape[1]} samples")

# ── 2. Identify our target miRNAs ─────────────────────────────────────────────
# Park et al. (2009): miR-125a and miR-200a significantly lower in OSCC saliva
# In TCGA Xena the row names are MIMAT accessions; search by name in index too
targets = {
    "hsa-miR-125a-5p": "MIMAT0000443",
    "hsa-miR-125a-3p": "MIMAT0004602",
    "hsa-miR-200a-3p": "MIMAT0000682",
    "hsa-miR-200a-5p": "MIMAT0004564",
}

# Check which are present
present = [acc for acc in targets.values() if acc in mirna.index]
print("\nTarget miRNAs found in dataset:", present)
# Map back to names
acc_to_name = {v: k for k, v in targets.items()}

# ── 3. Filter to Primary Tumour and Solid Tissue Normal ───────────────────────
keep_types = ["Primary Tumor", "Solid Tissue Normal"]
clin_filt  = clin[clin["sample_type"].isin(keep_types)]

# Match sample IDs between clinical and miRNA
common_samples = clin_filt.index.intersection(mirna.columns)
print(f"\nSamples with both miRNA and clinical data: {len(common_samples)}")

clin_filt  = clin_filt.loc[common_samples]
mirna_filt = mirna[common_samples]

print("Tumour:", (clin_filt["sample_type"] == "Primary Tumor").sum())
print("Normal:", (clin_filt["sample_type"] == "Solid Tissue Normal").sum())

# ── 4. Extract target miRNA expression ───────────────────────────────────────
results = {}
for acc in present:
    name = acc_to_name[acc]
    expr = mirna_filt.loc[acc]          # log2(RPM+1) values
    tumour = expr[clin_filt["sample_type"] == "Primary Tumor"].values
    normal = expr[clin_filt["sample_type"] == "Solid Tissue Normal"].values

    stat, pval = stats.mannwhitneyu(tumour, normal, alternative="two-sided")
    fc = np.median(tumour) - np.median(normal)   # log2 fold-change (median)

    results[name] = {
        "accession": acc,
        "tumour_median": np.median(tumour),
        "normal_median": np.median(normal),
        "log2FC_median": fc,
        "direction": "DOWN in tumour" if fc < 0 else "UP in tumour",
        "p_value": pval,
        "tumour_n": len(tumour),
        "normal_n": len(normal),
        "tumour_vals": tumour,
        "normal_vals": normal,
    }

# ── 5. Print summary table ───────────────────────────────────────────────────
print("\n── Results ──────────────────────────────────────────────────────")
for name, r in results.items():
    print(f"\n{name} ({r['accession']})")
    print(f"  Tumour median (log2 RPM+1): {r['tumour_median']:.3f}  (n={r['tumour_n']})")
    print(f"  Normal median (log2 RPM+1): {r['normal_median']:.3f}  (n={r['normal_n']})")
    print(f"  log2 FC (tumour vs normal): {r['log2FC_median']:+.3f}  → {r['direction']}")
    print(f"  Mann-Whitney U p-value:     {r['p_value']:.2e}")

# ── 6. Figure: boxplots for each target miRNA ─────────────────────────────────
PALETTE = {"Primary Tumor": "#E05C5C", "Solid Tissue Normal": "#5B8DB8"}
LABELS  = {"Primary Tumor": "Primary Tumour", "Solid Tissue Normal": "Solid Tissue Normal"}

n_plots = len(present)
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), constrained_layout=True)
if n_plots == 1:
    axes = [axes]

fig.suptitle(
    "TCGA-HNSC: Target miRNA Expression — Tumour vs Normal Tissue\n"
    "FYP Group 18 · Task A1 · Pilot Analysis",
    fontsize=13, fontweight="bold", y=1.01
)

for ax, acc in zip(axes, present):
    name = acc_to_name[acc]
    r    = results[name]

    data_plot = pd.DataFrame({
        "Expression (log2 RPM+1)": np.concatenate([r["tumour_vals"], r["normal_vals"]]),
        "Sample type": (["Primary Tumor"] * r["tumour_n"] +
                        ["Solid Tissue Normal"] * r["normal_n"])
    })

    sns.boxplot(
        data=data_plot, x="Sample type", y="Expression (log2 RPM+1)",
        palette=PALETTE, width=0.5, fliersize=2, ax=ax
    )
    sns.stripplot(
        data=data_plot, x="Sample type", y="Expression (log2 RPM+1)",
        color="black", alpha=0.15, size=2, jitter=True, ax=ax
    )

    # Significance annotation
    y_max = data_plot["Expression (log2 RPM+1)"].max() * 1.05
    pval  = r["p_value"]
    sig   = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    ax.plot([0, 1], [y_max, y_max], color="black", lw=1)
    ax.text(0.5, y_max * 1.01, f"{sig}\np = {pval:.2e}", ha="center", fontsize=9)

    direction_colour = "#cc0000" if r["log2FC_median"] < 0 else "#006600"
    ax.set_title(
        f"{name}\n"
        f"log2FC = {r['log2FC_median']:+.2f}  ({r['direction']})",
        fontsize=10, color=direction_colour, fontweight="bold"
    )
    ax.set_xticklabels([LABELS[t] for t in ["Primary Tumor", "Solid Tissue Normal"]], fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Expression log2(RPM+1)", fontsize=9)

# Legend note
fig.text(
    0.5, -0.04,
    "Data: TCGA-HNSC (UCSC Xena). Units: log2(RPM+1). "
    "Statistical test: Mann-Whitney U (two-sided).\n"
    "Park et al. (2009) reported miR-125a and miR-200a LOWER in OSCC saliva vs controls.",
    ha="center", fontsize=8, style="italic", color="#444444"
)

out_path = os.path.join(FIG_DIR, "tcga_mirna_tumour_vs_normal.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
plt.show()

# ── 7. Save results table ─────────────────────────────────────────────────────
rows = []
for name, r in results.items():
    rows.append({
        "miRNA": name,
        "Accession": r["accession"],
        "Tumour median log2(RPM+1)": round(r["tumour_median"], 3),
        "Normal median log2(RPM+1)": round(r["normal_median"], 3),
        "log2 FC (tumour vs normal)": round(r["log2FC_median"], 3),
        "Direction": r["direction"],
        "Mann-Whitney p-value": f"{r['p_value']:.3e}",
        "Tumour n": r["tumour_n"],
        "Normal n": r["normal_n"],
    })

df_results = pd.DataFrame(rows)
df_results.to_csv(os.path.join(FIG_DIR, "tcga_mirna_results_table.csv"), index=False)
print("\nResults table:")
print(df_results.to_string(index=False))