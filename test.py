# NEW 3 — Correlation comparison — standalone visual

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle(
    "What Predicts Mirror Contamination? REO vs GH",
    fontsize=13, fontweight="semibold",
    color=TEXT, y=1.01
)

predictors = {
    "count_small_excl_last":   "Small defect count",
    "count_medium_excl_last":  "Medium defect count",
    "count_large_excl_last":   "Large defect count",
    "count_xlarge_excl_last":  "XLarge defect count",
    "count_xxlarge_excl_last": "XXLarge defect count",
    "defect_count_excl_last":  "Total defect count",
    "nn_clark_evans_r":        "Clustering (CE-R)",
    "nn_mean_nnd":             "NN distance",
    "_weighted_sum":           "Size severity score",
}

labels    = list(predictors.values())
reo_corrs = []
gh_corrs  = []

df["contaminated"] = (
    df["total_coverage_excl_last"] > 0.05
).astype(int)

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

# ── Left: horizontal bar chart ────────────────────────────
ax = axes[0]
styled_ax(ax)

x     = np.arange(len(labels))
width = 0.35

ax.barh(x + width/2, reo_corrs, width,
        color=VENDOR_A, alpha=0.85,
        label="REO", zorder=3)
ax.barh(x - width/2, gh_corrs,  width,
        color=VENDOR_B, alpha=0.85,
        label="GH",  zorder=3)

for val, y_pos in zip(reo_corrs, x + width/2):
    if not np.isnan(val) and val > 0.01:
        ax.text(val + 0.005, y_pos,
                f"{val:.3f}",
                va="center", fontsize=8,
                color=VENDOR_A, fontweight="bold")

for val, y_pos in zip(gh_corrs, x - width/2):
    if not np.isnan(val) and val > 0.01:
        ax.text(val + 0.005, y_pos,
                f"{val:.3f}",
                va="center", fontsize=8,
                color=VENDOR_B, fontweight="bold")

# Reference lines
for x_val, lbl in [(0.3, "moderate"), (0.5, "strong")]:
    ax.axvline(x_val, color=MUTED, lw=0.8,
               linestyle="--", alpha=0.5)
    ax.text(x_val, len(labels) - 0.2, lbl,
            color=MUTED, fontsize=7.5,
            ha="center", va="bottom")

ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Correlation with contamination  (Pearson r)")
ax.set_title("Predictor strength per vendor",
             fontsize=11)
ax.set_xlim(0, 0.7)
ax.legend(fontsize=9)

# ── Right: interpretation ─────────────────────────────────
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
    (VENDOR_B, "GH — predictable",
     "All size categories r > 0.4.\n"
     "Total count r = 0.504.\n"
     "More defects = more likely contaminated.\n"
     "Contamination has detectable causes\n"
     "at the individual mirror level."),

    (VENDOR_A, "REO — unpredictable",
     "No predictor exceeds r = 0.334.\n"
     "Total count r = 0.227 — very weak.\n"
     "REO mirrors contaminate regardless\n"
     "of individual defect character.\n"
     "Size and count don't explain it."),

    (MUTED, "Process implication",
     "REO contamination is a baseline\n"
     "process state — not driven by\n"
     "specific defect events.\n"
     "Targeting individual defect types\n"
     "will not fix REO's 71.8% rate.\n"
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
            fontsize=8.5, color=TEXT,
            va="top", linespacing=1.6)
    y -= 0.27

fig.text(0.5, -0.03,
         "Pearson r between predictor and contamination flag "
         "(coverage > 0.05%). "
         "Outer ring (R9) excluded.",
         ha="center", fontsize=8,
         color=MUTED, style="italic")

plt.tight_layout()
plt.show()
