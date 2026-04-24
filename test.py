# Small defect spatial pattern breakdown
# Addition to Phase 4 — standalone cell

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
fig.suptitle(
    "Small Defect Spatial Pattern Breakdown  "
    "(Clark-Evans R — Small blobs only)",
    fontsize=13, fontweight="semibold",
    color=TEXT, y=1.01
)

col = "clarck_evans_r_small"

PATTERN_THRESHOLDS = {
    "Clustered\n(CE-R < 0.7)":    (None, 0.7),
    "Random\n(0.7 – 1.3)":        (0.7,  1.3),
    "Dispersed\n(CE-R > 1.3)":    (1.3,  None),
}
PATTERN_COLORS = {
    "Clustered\n(CE-R < 0.7)":  "#E74C3C",
    "Random\n(0.7 – 1.3)":      "#95A5A6",
    "Dispersed\n(CE-R > 1.3)":  "#2ECC71",
}

def get_pattern_pcts(vendor):
    vdf  = df[df.SUB_VENDOR == vendor][col].dropna()
    pcts = {}
    for label, (lo, hi) in PATTERN_THRESHOLDS.items():
        if lo is None:
            pcts[label] = (vdf < hi).mean() * 100
        elif hi is None:
            pcts[label] = (vdf > lo).mean() * 100
        else:
            pcts[label] = ((vdf >= lo) & (vdf <= hi)).mean() * 100
    return pcts

reo_pcts = get_pattern_pcts("REO")
gh_pcts  = get_pattern_pcts("GH")

# ── Left: grouped bar ─────────────────────────────────────
ax = axes[0]
styled_ax(ax)

patterns = list(PATTERN_THRESHOLDS.keys())
x        = np.arange(len(patterns))
width    = 0.35

bars_r = ax.bar(x - width/2,
                [reo_pcts[p] for p in patterns],
                width, color=VENDOR_A, alpha=0.85,
                label="REO", zorder=3)
bars_g = ax.bar(x + width/2,
                [gh_pcts[p] for p in patterns],
                width, color=VENDOR_B, alpha=0.85,
                label="GH", zorder=3)

for bars, pcts, color in [
    (bars_r, reo_pcts, VENDOR_A),
    (bars_g, gh_pcts,  VENDOR_B)
]:
    for bar, pattern in zip(bars, patterns):
        val = pcts[pattern]
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=9, color=color,
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(patterns, fontsize=9)
ax.set_ylabel("% of mirrors")
ax.set_title("Pattern breakdown — small defects",
             fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=9)

# ── Middle: CE-R distribution small only ──────────────────
ax = axes[1]
styled_ax(ax)

for vendor, color in [("REO", VENDOR_A), ("GH", VENDOR_B)]:
    vdf  = df[df.SUB_VENDOR == vendor][col].dropna()
    _, p = mannwhitneyu(
        df[df.SUB_VENDOR=="REO"][col].dropna(),
        df[df.SUB_VENDOR=="GH"][col].dropna(),
        alternative="two-sided"
    )
    bins = np.linspace(0, 2.5, 40)
    ax.hist(vdf.clip(upper=2.5), bins=bins,
            color=color, alpha=0.70, density=True,
            label=f"{vendor}  "
                  f"μ={vdf.mean():.3f}  "
                  f"med={vdf.median():.3f}")

# Threshold lines
for x_val, lbl, col_line in [
    (0.7, "clustered\nthreshold", "#E74C3C"),
    (1.3, "dispersed\nthreshold", "#2ECC71"),
]:
    ax.axvline(x_val, color=col_line, lw=1.2,
               linestyle="--", alpha=0.7)
    ax.text(x_val + 0.02, 0, lbl,
            transform=ax.get_xaxis_transform(),
            fontsize=7, color=col_line,
            va="bottom")

ax.set_xlabel("Clark-Evans R (small defects)")
ax.set_ylabel("Density")
ax.set_title("CE-R distribution — small defects",
             fontsize=11)
ax.legend(fontsize=8)
ax.text(0.97, 0.95, sig_label(p),
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=BG_CARD,
                  edgecolor=BORDER, alpha=0.8))

# ── Right: summary interpretation ─────────────────────────
ax = axes[2]
ax.set_facecolor(BG_CARD)
ax.axis("off")

ax.text(0.5, 0.97, "Interpretation",
        transform=ax.transAxes,
        fontsize=11, color=TEXT,
        fontweight="semibold",
        va="top", ha="center")

ax.plot([0.05, 0.95], [0.91, 0.91],
        transform=ax.transAxes,
        color=BORDER, linewidth=0.8)

findings = [
    (VENDOR_A, "REO small defects",
     "90.7% random spatial pattern.\n"
     "High density uniform coverage.\n"
     "Consistent with process generating\n"
     "contamination across full surface."),

    (VENDOR_B, "GH small defects",
     "15.1% clustered — 2x more than REO.\n"
     "11.7% dispersed — 9x more than REO.\n"
     "More bimodal — either grouped\n"
     "or spread, less uniform than REO."),

    (MUTED, "Key insight",
     "REO's high small defect count\n"
     "creates dense uniform coverage.\n"
     "GH's fewer small defects form\n"
     "distinct clusters when present."),
]

y = 0.87
for color, title, body in findings:
    ax.text(0.04, y, title,
            transform=ax.transAxes,
            fontsize=9, color=color,
            fontweight="bold", va="top")
    y -= 0.07
    ax.text(0.04, y, body,
            transform=ax.transAxes,
            fontsize=8, color=TEXT,
            va="top", linespacing=1.5)
    y -= 0.22

fig.text(0.5, -0.03,
         "Small blobs: 1–25px. "
         "CE-R computed per mirror on small blob centroids only. "
         "Mirrors with <3 small blobs excluded from CE-R calculation.",
         ha="center", fontsize=7.5,
         color=MUTED, style="italic")

plt.tight_layout()
plt.show()
