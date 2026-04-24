# NEW 1 — DINOv2 class as contamination predictor
# NEW 2 — No defect blind spot per vendor
# NEW 3 — Correlation comparison table
# All three in one block

# ─────────────────────────────────────────────────────────────
# NEW 1 — Class × contamination rate
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig.suptitle(
    "DINOv2 Predicted Class — Contamination Rate Analysis",
    fontsize=13, fontweight="semibold",
    color=TEXT, y=1.01
)

CLASSES_ALL = ["fod", "no defect", "polish defect",
               "wax", "scratch", "water spot"]
CLASS_COLORS_MAP = {
    "fod":           "#2196F3",
    "polish defect": "#FF9800",
    "wax":           "#9C27B0",
    "scratch":       "#F44336",
    "water spot":    "#00BCD4",
    "no defect":     "#9E9E9E",
}

df["contaminated"] = (
    df["total_coverage_excl_last"] > 0.05
).astype(int)

# ── Left: contamination rate per class overall ─────────────
ax = axes[0]
styled_ax(ax)

rates = []
ns    = []
for cls in CLASSES_ALL:
    subset = df[df.predicted_defects == cls]
    rate   = subset["contaminated"].mean() * 100
    rates.append(rate)
    ns.append(len(subset))

colors = [CLASS_COLORS_MAP[c] for c in CLASSES_ALL]
bars   = ax.bar(range(len(CLASSES_ALL)), rates,
                color=colors, alpha=0.85,
                width=0.6, zorder=3)

for bar, rate, n in zip(bars, rates, ns):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center", va="bottom",
            fontsize=8.5, color=TEXT,
            fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2,
            1,
            f"n={n:,}",
            ha="center", va="bottom",
            fontsize=7, color="white")

# Reference line — overall contamination rate
overall = df["contaminated"].mean() * 100
ax.axhline(overall, color=MUTED, lw=1.2,
           linestyle="--", alpha=0.6)
ax.text(len(CLASSES_ALL) - 0.1, overall + 1,
        f"overall {overall:.1f}%",
        color=MUTED, fontsize=7.5, ha="right")

ax.set_xticks(range(len(CLASSES_ALL)))
ax.set_xticklabels(CLASSES_ALL, rotation=20,
                    ha="right", fontsize=9)
ax.set_ylabel("% mirrors contaminated\n(coverage > 0.05%)")
ax.set_title("Contamination rate per class\n(both vendors)",
             fontsize=11)
ax.set_ylim(0, 110)

# ── Middle: per vendor × class heatmap ────────────────────
ax = axes[1]
ax.set_facecolor(BG_PANEL)
ax.grid(False)

vendors = ["REO", "GH"]
matrix  = np.zeros((len(CLASSES_ALL), len(vendors)))

for ci, cls in enumerate(CLASSES_ALL):
    for vi, vendor in enumerate(vendors):
        subset = df[(df.predicted_defects == cls) &
                    (df.SUB_VENDOR == vendor)]
        if len(subset) > 0:
            matrix[ci, vi] = subset["contaminated"].mean() * 100
        else:
            matrix[ci, vi] = np.nan

im = ax.imshow(matrix, cmap="RdYlGn_r",
               aspect="auto", vmin=0, vmax=100)

for ci in range(len(CLASSES_ALL)):
    for vi in range(len(vendors)):
        val = matrix[ci, vi]
        if not np.isnan(val):
            txt_color = "white" if val > 60 else TEXT
            ax.text(vi, ci, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=9, color=txt_color,
                    fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(["REO", "GH"], fontsize=11,
                    fontweight="bold")
ax.set_yticks(range(len(CLASSES_ALL)))
ax.set_yticklabels(CLASSES_ALL, fontsize=9)
ax.set_title("Contamination rate\nper vendor per class",
             fontsize=11)

plt.colorbar(im, ax=ax, label="% contaminated",
             fraction=0.046)

# ── Right: no defect blind spot ───────────────────────────
ax = axes[2]
ax.set_facecolor(BG_CARD)
ax.axis("off")

no_def = df[df.predicted_defects == "no defect"]
nd_stats = no_def.groupby("SUB_VENDOR").agg(
    total=("contaminated", "count"),
    n_contaminated=("contaminated", "sum"),
    pct=("contaminated", "mean")
).round(3)

ax.text(0.5, 0.97,
        "Model Blind Spot",
        transform=ax.transAxes,
        fontsize=12, color=TEXT,
        fontweight="bold",
        va="top", ha="center")

ax.text(0.5, 0.89,
        "Mirrors labeled 'No Defect'\nthat are actually contaminated",
        transform=ax.transAxes,
        fontsize=9, color=MUTED,
        va="top", ha="center",
        style="italic")

ax.plot([0.05, 0.95], [0.82, 0.82],
        transform=ax.transAxes,
        color=BORDER, linewidth=0.8)

y = 0.76
for vendor, color in [("REO", VENDOR_A), ("GH", VENDOR_B)]:
    if vendor not in nd_stats.index:
        continue
    row   = nd_stats.loc[vendor]
    pct   = row["pct"] * 100
    total = int(row["total"])
    n_con = int(row["n_contaminated"])

    ax.text(0.5, y, vendor,
            transform=ax.transAxes,
            fontsize=14, color=color,
            fontweight="bold",
            va="top", ha="center")
    y -= 0.08

    ax.text(0.5, y, f"{pct:.1f}%",
            transform=ax.transAxes,
            fontsize=28, color=color,
            fontweight="bold",
            va="top", ha="center")
    y -= 0.12

    ax.text(0.5, y,
            f"{n_con:,} of {total:,} 'no defect' mirrors\n"
            f"have coverage > 0.05%",
            transform=ax.transAxes,
            fontsize=8.5, color=TEXT,
            va="top", ha="center",
            linespacing=1.5)
    y -= 0.18

ax.plot([0.05, 0.95], [y + 0.04, y + 0.04],
        transform=ax.transAxes,
        color=BORDER, linewidth=0.8)
y -= 0.02

ax.text(0.5, y,
        "Implication: DINOv2 is passing\n"
        "contaminated mirrors as clean.\n"
        "These mirrors may reach assembly\n"
        "without further inspection.",
        transform=ax.transAxes,
        fontsize=8.5, color=TEXT,
        va="top", ha="center",
        linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=BG_PANEL,
                  edgecolor=BORDER,
                  alpha=0.8))

fig.text(0.5, -0.03,
         "Contamination threshold: total_coverage_excl_last > 0.05%. "
         "Outer ring (R9) excluded.",
         ha="center", fontsize=8,
         color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# NEW 3 — Correlation comparison table
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle(
    "What Predicts Mirror Contamination? REO vs GH",
    fontsize=13, fontweight="semibold",
    color=TEXT, y=1.01
)

SIZE_CATS  = ["small", "medium", "large", "xlarge", "xxlarge"]
predictors = {
    "count_small_excl_last":   "Small defect count",
    "count_medium_excl_last":  "Medium defect count",
    "count_large_excl_last":   "Large defect count",
    "count_xlarge_excl_last":  "XLarge defect count",
    "count_xxlarge_excl_last": "XXLarge defect count",
    "defect_count_excl_last":  "Total defect count",
    "total_coverage_excl_last":"Total coverage %",
    "nn_clark_evans_r":        "Clustering (CE-R)",
    "nn_mean_nnd":             "NN distance",
    "_weighted_sum":           "Size severity score",
}

# ── Left: correlation bar chart ───────────────────────────
ax = axes[0]
styled_ax(ax)

labels   = list(predictors.values())
reo_corrs = []
gh_corrs  = []

for col in predictors.keys():
    if col not in df.columns:
        reo_corrs.append(np.nan)
        gh_corrs.append(np.nan)
        continue
    reo_corrs.append(
        df[df.SUB_VENDOR=="REO"][col]
        .corr(df[df.SUB_VENDOR=="REO"]["contaminated"])
    )
    gh_corrs.append(
        df[df.SUB_VENDOR=="GH"][col]
        .corr(df[df.SUB_VENDOR=="GH"]["contaminated"])
    )

x     = np.arange(len(labels))
width = 0.35

bars_r = ax.barh(x + width/2, reo_corrs, width,
                 color=VENDOR_A, alpha=0.85,
                 label="REO", zorder=3)
bars_g = ax.barh(x - width/2, gh_corrs,  width,
                 color=VENDOR_B, alpha=0.85,
                 label="GH",  zorder=3)

for val, y_pos in zip(reo_corrs, x + width/2):
    if not np.isnan(val):
        ax.text(val + 0.005, y_pos,
                f"{val:.3f}",
                va="center", fontsize=7.5,
                color=VENDOR_A)

for val, y_pos in zip(gh_corrs, x - width/2):
    if not np.isnan(val):
        ax.text(val + 0.005, y_pos,
                f"{val:.3f}",
                va="center", fontsize=7.5,
                color=VENDOR_B)

# Weak/moderate reference lines
ax.axvline(0.3, color=MUTED, lw=0.8,
           linestyle="--", alpha=0.5)
ax.axvline(0.5, color=MUTED, lw=0.8,
           linestyle="--", alpha=0.5)
ax.text(0.3, len(labels) - 0.3, "moderate",
        color=MUTED, fontsize=7, ha="center")
ax.text(0.5, len(labels) - 0.3, "strong",
        color=MUTED, fontsize=7, ha="center")

ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Correlation with contamination (r)")
ax.set_title("Predictor strength per vendor",
             fontsize=11)
ax.set_xlim(0, 0.7)
ax.legend(fontsize=9)

# ── Right: interpretation summary ─────────────────────────
ax = axes[1]
ax.set_facecolor(BG_CARD)
ax.axis("off")

ax.text(0.5, 0.97, "Key Finding",
        transform=ax.transAxes,
        fontsize=12, color=TEXT,
        fontweight="bold",
        va="top", ha="center")

ax.plot([0.05, 0.95], [0.91, 0.91],
        transform=ax.transAxes,
        color=BORDER, linewidth=0.8)

findings = [
    (VENDOR_B, "GH — predictable contamination",
     "All size categories correlate\n"
     "moderately with contamination (r>0.4).\n"
     "Total count r=0.504.\n"
     "More defects = more likely contaminated.\n"
     "Contamination has detectable causes."),

    (VENDOR_A, "REO — unpredictable contamination",
     "No size category strongly predicts\n"
     "contamination (max r=0.334 XLarge).\n"
     "Total count r=0.227 — very weak.\n"
     "REO mirrors contaminate regardless\n"
     "of individual defect character."),

    (MUTED, "Process implication",
     "REO contamination is driven by\n"
     "baseline process state — not by\n"
     "specific defect events.\n"
     "Fixing individual defect types\n"
     "may not reduce REO's 71.8% rate.\n"
     "Process-level intervention needed."),
]

y = 0.87
for color, title, body in findings:
    ax.text(0.04, y, title,
            transform=ax.transAxes,
            fontsize=9, color=color,
            fontweight="bold", va="top")
    y -= 0.06
    ax.text(0.04, y, body,
            transform=ax.transAxes,
            fontsize=8, color=TEXT,
            va="top", linespacing=1.5)
    y -= 0.25

fig.text(0.5, -0.03,
         "Correlation = Pearson r between predictor "
         "and contamination flag (coverage > 0.05%). "
         "Outer ring excluded.",
         ha="center", fontsize=8,
         color=MUTED, style="italic")

plt.tight_layout()
plt.show()
