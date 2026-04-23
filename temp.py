# 5A — Predicted defect class distribution per vendor

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle("5A — Predicted Defect Class Distribution: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

# Class order and shades per vendor
CLASSES = ["fod", "no defect", "polish defect", "wax", "scratch", "water spot"]

# Shades light → dark per vendor matching class order
shades_reo = ["#0D3F5E", "#1D6FA4", "#3A8CBF", "#6BAED6", "#A8D4F0", "#D4EAF7"]
shades_gh  = ["#641E16", "#922B21", "#C0392B", "#E06B5A", "#F5A99A", "#FAD4CC"]

# ── Left: stacked bar ─────────────────────────────────────
ax = axes[0]
styled_ax(ax)

for xi, (vendor, shades) in enumerate(zip(["REO", "GH"],
                                           [shades_reo, shades_gh])):
    vdf    = df[df.SUB_VENDOR == vendor]
    total  = len(vdf)
    bottom = 0

    for cls, shade in zip(CLASSES, shades):
        pct = (vdf["predicted_defects"] == cls).sum() / total * 100

        ax.bar(xi, pct, bottom=bottom,
               color=shade, width=0.45,
               zorder=3)

        if pct > 1.5:
            ax.text(xi, bottom + pct / 2,
                    f"{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=8.5, color="white",
                    fontweight="bold")
        bottom += pct

ax.set_xticks([0, 1])
ax.set_xticklabels(["REO", "GH"], fontsize=12,
                    fontweight="bold")
ax.set_ylabel("% of all mirrors")
ax.set_ylim(0, 108)
ax.set_title("Class composition per vendor", fontsize=11)

# Legend
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(facecolor=shades_reo[i], label=cls)
    for i, cls in enumerate(CLASSES)
]
ax.legend(handles=legend_patches,
          fontsize=8, loc="upper right",
          facecolor=BG_PANEL, edgecolor=BORDER)

# ── Right: grouped bar — class by class comparison ────────
ax = axes[1]
styled_ax(ax)

x     = np.arange(len(CLASSES))
width = 0.35

reo_pcts = [(df[df.SUB_VENDOR=="REO"]["predicted_defects"] == cls).mean() * 100
            for cls in CLASSES]
gh_pcts  = [(df[df.SUB_VENDOR=="GH"]["predicted_defects"]  == cls).mean() * 100
            for cls in CLASSES]

bars_r = ax.bar(x - width/2, reo_pcts, width,
                color=VENDOR_A, alpha=0.85,
                label="REO", zorder=3)
bars_g = ax.bar(x + width/2, gh_pcts,  width,
                color=VENDOR_B, alpha=0.85,
                label="GH",  zorder=3)

for bars, vals, color in [(bars_r, reo_pcts, VENDOR_A),
                           (bars_g, gh_pcts,  VENDOR_B)]:
    for bar, val in zip(bars, vals):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    rotation=90)

# Highlight biggest differences
for i, (r, g, cls) in enumerate(zip(reo_pcts, gh_pcts, CLASSES)):
    diff = abs(r - g)
    if diff > 5:
        ax.text(i, max(r, g) + 2.5,
                f"Δ{diff:.1f}%",
                ha="center", fontsize=7.5,
                color=MUTED, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=20,
                    ha="right", fontsize=9)
ax.set_ylabel("% of all mirrors")
ax.set_title("Class by class comparison", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 65)

fig.text(0.5, -0.04,
         "Includes all mirrors — defective and clean. "
         "REO: FOD and Polish Defect dominate. "
         "GH: more balanced, Wax present (REO near zero).",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()
