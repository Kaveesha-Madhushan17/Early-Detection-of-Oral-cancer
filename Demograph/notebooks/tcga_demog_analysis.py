"""
Task A2 — TCGA-HNSC: Demographic & Clinical Characterisation
FYP Group 18, E21, University of Peradeniya
n = 604 patients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

CLIN_FILE = "../data/TCGA.HNSC.sampleMap_HNSC_clinicalMatrix"
OUT_DIR   = "../figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
clin = pd.read_csv(CLIN_FILE, sep="\t", index_col=0, low_memory=False)
print(f"Loaded: {clin.shape[0]} samples × {clin.shape[1]} columns")

# ── Key columns ───────────────────────────────────────────────────────────────
df = pd.DataFrame()
df["Age"]           = pd.to_numeric(clin["age_at_initial_pathologic_diagnosis"], errors="coerce")
df["Gender"]        = clin["gender"].str.strip().str.title()
df["TumourSite"]    = clin["anatomic_neoplasm_subdivision"].str.strip().str.title()
df["Stage"]         = clin["pathologic_stage"].str.strip()
df["VitalStatus"]   = clin["vital_status"].str.strip().str.title()
df["Tobacco"]       = pd.to_numeric(clin["tobacco_smoking_history"], errors="coerce")
df["Alcohol"]       = clin["alcohol_history_documented"].str.strip().str.upper()
df["HPV_p16"]       = clin["hpv_status_by_p16_testing"].str.strip().str.title()
df["SampleType"]    = clin["sample_type"].str.strip()

# Tobacco: TCGA coding 1=never, 2=current reformed <15yr, 3=current reformed ≥15yr, 4=current smoker, 5=unknown
tobacco_map = {1: "Never", 2: "Former (<15 yr)", 3: "Former (≥15 yr)", 4: "Current", 5: "Unknown/NOS"}
df["TobaccoLabel"] = df["Tobacco"].map(tobacco_map).fillna("Unknown/NOS")

# Simplify stage
stage_map = {"Stage I": "I", "Stage II": "II", "Stage III": "III",
             "Stage IVA": "IVA", "Stage IVB": "IVB", "Stage IVC": "IVC"}
df["StageSim"] = df["Stage"].map(stage_map).fillna("Unknown")

# Simplify tumour site
site_remap = {
    "Oral Tongue": "Oral Tongue", "Base Of Tongue": "Base of Tongue",
    "Floor Of Mouth": "Floor of Mouth", "Larynx": "Larynx",
    "Hypopharynx": "Hypopharynx", "Oropharynx": "Oropharynx",
    "Lip": "Lip", "Hard Palate": "Hard Palate",
    "Buccal Mucosa": "Buccal Mucosa", "Alveolar Ridge": "Alveolar Ridge",
}
df["SiteSim"] = df["TumourSite"].map(site_remap).fillna("Other/Unknown")

# Filter to primary tumour only for most panels
primary = df[df["SampleType"] == "Primary Tumor"].copy()
print(f"Primary tumour samples: {len(primary)}")
print(f"Age: mean={primary['Age'].mean():.1f}, median={primary['Age'].median():.0f}, range={primary['Age'].min():.0f}–{primary['Age'].max():.0f}")
print(f"Gender:\n{primary['Gender'].value_counts()}")
print(f"Stage:\n{primary['StageSim'].value_counts()}")
print(f"Top sites:\n{primary['SiteSim'].value_counts().head(8)}")
print(f"Tobacco:\n{primary['TobaccoLabel'].value_counts()}")
print(f"Alcohol:\n{primary['Alcohol'].value_counts()}")
print(f"HPV p16:\n{primary['HPV_p16'].value_counts()}")
print(f"Vital status:\n{primary['VitalStatus'].value_counts()}")

# ── Colours ───────────────────────────────────────────────────────────────────
TEAL_DARK  = "#1a5f70"
TEAL_MID   = "#2d7a8e"
TEAL_LIGHT = "#a8d4dc"
TEAL_PALE  = "#d6eaee"
RED        = "#c94040"
PINK       = "#c4a0b8"
AMBER      = "#d4824a"
GRAY       = "#8a9eaa"
BG         = "#f5f7f9"
TAKEAWAY   = "#1a5f70"

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 11), facecolor=BG)
fig.suptitle(
    "TCGA-HNSC: Demographic & Clinical Characterisation  (n = 604)\n"
    "FYP Group 18 · Task A2 · Public Cohort Reference Analysis",
    fontsize=14, fontweight="bold", color=TEAL_DARK, y=0.98
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38,
                       left=0.06, right=0.97, top=0.90, bottom=0.10)

# ── P1: Age distribution ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ages = primary["Age"].dropna()
bins = np.arange(20, 100, 5)
n, edges, patches = ax1.hist(ages, bins=bins, color=TEAL_MID, edgecolor=BG, alpha=0.88)
ax1.axvline(ages.median(), color=RED, lw=1.8, ls="--", label=f"Median {ages.median():.0f} yr")
ax1.axvline(ages.mean(),   color=AMBER, lw=1.5, ls=":",  label=f"Mean {ages.mean():.1f} yr")
ax1.set_xlabel("Age at diagnosis (years)", fontsize=9, color=GRAY)
ax1.set_ylabel("Count", fontsize=9, color=GRAY)
ax1.set_title("Age Distribution", fontsize=11, fontweight="bold", color=TEAL_DARK)
ax1.legend(fontsize=8, framealpha=0)
ax1.set_facecolor(BG); ax1.spines[["top","right"]].set_visible(False)
ax1.tick_params(colors=GRAY)
for sp in ["left","bottom"]: ax1.spines[sp].set_color(TEAL_LIGHT)

# ── P2: Sex ───────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sex = primary["Gender"].value_counts()
wedge_colors = [TEAL_MID, PINK]
wedges, texts, autotexts = ax2.pie(
    sex.values, labels=sex.index, colors=wedge_colors[:len(sex)],
    autopct="%1.1f%%", startangle=90, pctdistance=0.72,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2)
)
for t in texts:    t.set_fontsize(10); t.set_color(TEAL_DARK); t.set_fontweight("bold")
for at in autotexts: at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
ax2.text(0, 0, f"n={len(primary)}", ha="center", va="center",
         fontsize=12, fontweight="bold", color=TEAL_DARK)
ax2.set_title("Sex", fontsize=11, fontweight="bold", color=TEAL_DARK)
ax2.set_facecolor(BG)

# ── P3: Tumour stage ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
stage_order = ["I", "II", "III", "IVA", "IVB", "IVC", "Unknown"]
stage_counts = primary["StageSim"].value_counts().reindex(stage_order, fill_value=0)
stage_counts = stage_counts[stage_counts > 0]
stage_colors = [TEAL_PALE, TEAL_LIGHT, TEAL_MID, TEAL_DARK, RED, PINK, GRAY][:len(stage_counts)]
bars3 = ax3.bar(stage_counts.index, stage_counts.values,
                color=stage_colors, edgecolor=BG, width=0.6)
for bar, val in zip(bars3, stage_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 2,
             str(val), ha="center", fontsize=9, fontweight="bold", color=TEAL_DARK)
ax3.set_ylabel("Count", fontsize=9, color=GRAY)
ax3.set_title("Pathologic Stage", fontsize=11, fontweight="bold", color=TEAL_DARK)
ax3.set_facecolor(BG); ax3.spines[["top","right"]].set_visible(False)
ax3.tick_params(colors=GRAY)
for sp in ["left","bottom"]: ax3.spines[sp].set_color(TEAL_LIGHT)

# ── P4: Vital status ──────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
vital = primary["VitalStatus"].value_counts()
v_colors = {"Living": TEAL_MID, "Deceased": RED}
vcolors = [v_colors.get(v, GRAY) for v in vital.index]
wedges4, texts4, auto4 = ax4.pie(
    vital.values, labels=vital.index, colors=vcolors,
    autopct="%1.1f%%", startangle=90, pctdistance=0.72,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2)
)
for t in texts4:  t.set_fontsize(10); t.set_color(TEAL_DARK); t.set_fontweight("bold")
for at in auto4:  at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
ax4.text(0, 0, "Survival", ha="center", va="center",
         fontsize=9, fontweight="bold", color=TEAL_DARK)
ax4.set_title("Vital Status", fontsize=11, fontweight="bold", color=TEAL_DARK)
ax4.set_facecolor(BG)

# ── P5: Tumour site ───────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 0:2])
site_counts = primary["SiteSim"].value_counts().head(9)
colors5 = [TEAL_DARK, TEAL_MID, TEAL_LIGHT, TEAL_PALE,
           RED, PINK, AMBER, GRAY, "#b0c8d0"][:len(site_counts)]
bars5 = ax5.barh(site_counts.index[::-1], site_counts.values[::-1],
                 color=colors5[::-1], edgecolor=BG, height=0.6)
for bar, val in zip(bars5, site_counts.values[::-1]):
    ax5.text(val + 1, bar.get_y() + bar.get_height()/2,
             str(val), va="center", fontsize=9, fontweight="bold", color=TEAL_DARK)
ax5.set_xlabel("Count", fontsize=9, color=GRAY)
ax5.set_title("Tumour Site Distribution", fontsize=11, fontweight="bold", color=TEAL_DARK)
ax5.set_facecolor(BG); ax5.spines[["top","right"]].set_visible(False)
ax5.tick_params(colors=GRAY, labelsize=9)
for sp in ["left","bottom"]: ax5.spines[sp].set_color(TEAL_LIGHT)

# ── P6: Risk factors (tobacco, alcohol, HPV) ─────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])

# Tobacco simplified
tob_simple = primary["TobaccoLabel"].replace(
    {"Former (<15 yr)": "Former", "Former (≥15 yr)": "Former", "Unknown/NOS": "Unknown"})
tob_counts = tob_simple.value_counts()
tob_order = ["Current", "Former", "Never", "Unknown"]
tob_counts = tob_counts.reindex([x for x in tob_order if x in tob_counts.index])
tob_colors = {"Current": RED, "Former": AMBER, "Never": TEAL_MID, "Unknown": GRAY}
t_colors = [tob_colors.get(t, GRAY) for t in tob_counts.index]
bars6 = ax6.bar(tob_counts.index, tob_counts.values, color=t_colors, edgecolor=BG, width=0.5)
for bar, val in zip(bars6, tob_counts.values):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 2,
             str(val), ha="center", fontsize=9, fontweight="bold", color=TEAL_DARK)
ax6.set_ylabel("Count", fontsize=9, color=GRAY)
ax6.set_title("Tobacco Smoking History", fontsize=11, fontweight="bold", color=TEAL_DARK)
ax6.set_facecolor(BG); ax6.spines[["top","right"]].set_visible(False)
ax6.tick_params(colors=GRAY)
for sp in ["left","bottom"]: ax6.spines[sp].set_color(TEAL_LIGHT)

# ── P7: HPV + Alcohol ────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[1, 3])
hpv = primary["HPV_p16"].value_counts()
alc = primary["Alcohol"].str.title().value_counts()

# Side-by-side mini bars for HPV and Alcohol
categories = ["HPV+\n(p16)", "HPV−\n(p16)", "Alcohol\nYes", "Alcohol\nNo"]
# HPV
hpv_pos = hpv.get("Positive", 0)
hpv_neg = hpv.get("Negative", 0)
alc_yes = alc.get("Yes", 0)
alc_no  = alc.get("No", 0)
vals = [hpv_pos, hpv_neg, alc_yes, alc_no]
bar_colors = [RED, TEAL_MID, AMBER, TEAL_LIGHT]
bars7 = ax7.bar(categories, vals, color=bar_colors, edgecolor=BG, width=0.5)
for bar, val in zip(bars7, vals):
    ax7.text(bar.get_x() + bar.get_width()/2, val + 2,
             str(val), ha="center", fontsize=9, fontweight="bold", color=TEAL_DARK)
ax7.set_ylabel("Count", fontsize=9, color=GRAY)
ax7.set_title("HPV Status & Alcohol", fontsize=11, fontweight="bold", color=TEAL_DARK)
ax7.set_facecolor(BG); ax7.spines[["top","right"]].set_visible(False)
ax7.tick_params(colors=GRAY, labelsize=9)
for sp in ["left","bottom"]: ax7.spines[sp].set_color(TEAL_LIGHT)

# ── Takeaway ──────────────────────────────────────────────────────────────────
takeaway = (
    f"TCGA-HNSC (n=604):  "
    f"Age {ages.median():.0f} yr median (range {ages.min():.0f}–{ages.max():.0f})  ·  "
    f"{sex.get('Male',0)} Male / {sex.get('Female',0)} Female  ·  "
    f"Stage IVA most common ({stage_counts.get('IVA',0)} patients)  ·  "
    f"Oral tongue dominant site ({site_counts.iloc[0]} cases)  ·  "
    f"Current/former smokers: {tob_counts.get('Current',0)+tob_counts.get('Former',0)}  ·  "
    f"HPV+ (p16): {hpv_pos}"
)
fig.text(0.5, 0.025, takeaway,
         ha="center", va="center", fontsize=9.5,
         color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor=TAKEAWAY, edgecolor="none", alpha=0.95))

fig.text(0.01, 0.005,
         "Data: TCGA-HNSC (UCSC Xena, 2015). Primary tumour samples only. FYP Group 18.",
         fontsize=7.5, color=GRAY, style="italic")

out = os.path.join(OUT_DIR, "tcga_demographics.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"\nFigure saved: {out}")