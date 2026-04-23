from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────
# 4A — Clark-Evans overall distribution
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
styled_ax(ax)

defective = df[df.defect_count_excl_last > 0]
reo = defective[defective.SUB_VENDOR == "REO"]["nn_clark_evans_r"].dropna()
gh  = defective[defective.SUB_VENDOR == "GH"]["nn_clark_evans_r"].dropna()

_, p = mannwhitneyu(reo, gh, alternative="two-sided")

bins = np.linspace(0, 2.0, 40)
ax.hist(reo, bins=bins, color=VENDOR_A, alpha=0.70, density=True,
        label=f"REO  μ={reo.mean():.3f}  med={reo.median():.3f}")
ax.hist(gh,  bins=bins, color=VENDOR_B, alpha=0.70, density=True,
        label=f"GH   μ={gh.mean():.3f}  med={gh.median():.3f}")

# Reference lines
for x_val, lbl in [(0.7, "clustered"), (1.0, "random"), (1.3, "dispersed")]:
    ax.axvline(x_val, color=MUTED, lw=0.8,
               linestyle="--", alpha=0.5)
    ax.text(x_val, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            lbl, color=MUTED, fontsize=7.5,
            ha="center", va="bottom",
            transform=ax.get_xaxis_transform())

ax.set_xlabel("Clark-Evans R  (< 1 = clustered,  1 = random,  > 1 = dispersed)")
ax.set_ylabel("Density")
ax.set_title("4A — Defect Spatial Pattern: Clark-Evans R Distribution",
             fontsize=12, fontweight="semibold", color=TEXT)
ax.legend(fontsize=9)
ax.text(0.97, 0.95, sig_label(p),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=BG_CARD, edgecolor=BORDER, alpha=0.8))

fig.text(0.5, -0.03,
         "Defective mirrors only. "
         "Clark-Evans R computed on all blob centroids per mirror.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 4B — Clark-Evans by size category
# ─────────────────────────────────────────────────────────────

SIZE_CATS  = ["small", "medium", "large", "xlarge", "xxlarge"]
ce_cols    = [f"clarck_evans_r_{sz}" for sz in SIZE_CATS]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle("4B — Clark-Evans R by Size Category: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

shades_reo = ["#A8D4F0", "#6BAED6", "#3A8CBF", "#1D6FA4", "#0D3F5E"]
shades_gh  = ["#F5A99A", "#E06B5A", "#C0392B", "#922B21", "#641E16"]

# ── Left: median CE per size category ─────────────────────
ax = axes[0]
styled_ax(ax)

x     = np.arange(len(SIZE_CATS))
width = 0.35

reo_vals = defective[defective.SUB_VENDOR=="REO"][ce_cols].median().values
gh_vals  = defective[defective.SUB_VENDOR=="GH"][ce_cols].median().values

bars_r = ax.bar(x - width/2, reo_vals, width,
                color=VENDOR_A, alpha=0.85,
                label="REO", zorder=3)
bars_g = ax.bar(x + width/2, gh_vals,  width,
                color=VENDOR_B, alpha=0.85,
                label="GH",  zorder=3)

for bars, vals, color in [(bars_r, reo_vals, VENDOR_A),
                           (bars_g, gh_vals,  VENDOR_B)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color=color)

ax.axhline(1.0, color=MUTED, lw=1.2,
           linestyle="--", alpha=0.6,
           label="random (R=1)")
ax.axhline(0.7, color=MUTED, lw=0.8,
           linestyle=":", alpha=0.4)
ax.axhline(1.3, color=MUTED, lw=0.8,
           linestyle=":", alpha=0.4)

ax.text(4.6, 0.71, "clustered threshold",
        fontsize=7, color=MUTED, alpha=0.7)
ax.text(4.6, 1.31, "dispersed threshold",
        fontsize=7, color=MUTED, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(SIZE_CATS)
ax.set_ylabel("Median Clark-Evans R")
ax.set_title("Median CE-R by size", fontsize=11)
ax.set_ylim(0.5, 1.5)
ax.legend(fontsize=9)

# ── Right: distribution per size — small vs xxlarge ───────
ax = axes[1]
styled_ax(ax)

# Show distribution for small and xxlarge only
# most interesting — small clusters, xxlarge reverses
for sz, color_r, color_g, ls in [
    ("small",   shades_reo[0], shades_gh[0], "-"),
    ("xxlarge", shades_reo[4], shades_gh[4], "--"),
]:
    col = f"clarck_evans_r_{sz}"
    reo_v = defective[defective.SUB_VENDOR=="REO"][col].dropna()
    gh_v  = defective[defective.SUB_VENDOR=="GH"][col].dropna()

    bins = np.linspace(0, 2.5, 35)
    ax.hist(reo_v.clip(upper=2.5), bins=bins,
            color=color_r, alpha=0.55, density=True,
            linestyle=ls, label=f"REO {sz}")
    ax.hist(gh_v.clip(upper=2.5),  bins=bins,
            color=color_g, alpha=0.55, density=True,
            linestyle=ls, label=f"GH {sz}")

ax.axvline(1.0, color=MUTED, lw=1.0,
           linestyle="--", alpha=0.5)
ax.text(1.01, 0, "random", color=MUTED, fontsize=7.5,
        transform=ax.get_xaxis_transform(), va="bottom")

ax.set_xlabel("Clark-Evans R")
ax.set_ylabel("Density")
ax.set_title("Distribution: Small vs XXLarge", fontsize=11)
ax.legend(fontsize=8, ncol=2)

fig.text(0.5, -0.03,
         "Small defects cluster (R<1). "
         "Medium–XLarge disperse (R>1). "
         "GH XXLarge clusters more than REO XXLarge.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 4C + 4D — NN distance and spacing regularity
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle("4C/4D — Nearest Neighbour Distance: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

for ax, col, title, xlabel in [
    (axes[0], "nn_mean_nnd", "4C — Mean NN Distance",
     "Mean nearest-neighbour distance (px)"),
    (axes[1], "nn_std_nnd",  "4D — NN Spacing Regularity",
     "Std dev of NN distance (px)  ← low = regular spacing"),
]:
    styled_ax(ax)

    reo_v = defective[defective.SUB_VENDOR=="REO"][col].dropna()
    gh_v  = defective[defective.SUB_VENDOR=="GH"][col].dropna()
    _, p  = mannwhitneyu(reo_v, gh_v, alternative="two-sided")

    cap  = defective[col].quantile(0.95)
    bins = np.linspace(0, cap, 35)

    ax.hist(reo_v.clip(upper=cap), bins=bins,
            color=VENDOR_A, alpha=0.70, density=True,
            label=f"REO  μ={reo_v.mean():.0f}px"
                  f"  med={reo_v.median():.0f}px")
    ax.hist(gh_v.clip(upper=cap),  bins=bins,
            color=VENDOR_B, alpha=0.70, density=True,
            label=f"GH   μ={gh_v.mean():.0f}px"
                  f"  med={gh_v.median():.0f}px")

    ax.axvline(reo_v.median(), color=VENDOR_A,
               lw=1.2, linestyle="--", alpha=0.8)
    ax.axvline(gh_v.median(),  color=VENDOR_B,
               lw=1.2, linestyle="--", alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, sig_label(p),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=BG_CARD,
                      edgecolor=BORDER, alpha=0.8))

fig.text(0.5, -0.03,
         "REO defects are ~2x closer together than GH defects. "
         "Lower std = more regular spacing = repeating process contact.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 4E — NN distance by size category
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle("4E — Mean NN Distance by Size Category: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

nnd_cols   = [f"mean_nnd_{sz}" for sz in SIZE_CATS]
shades_reo = ["#A8D4F0", "#6BAED6", "#3A8CBF", "#1D6FA4", "#0D3F5E"]
shades_gh  = ["#F5A99A", "#E06B5A", "#C0392B", "#922B21", "#641E16"]

# ── Left: bar chart median NND per size ───────────────────
ax = axes[0]
styled_ax(ax)

x     = np.arange(len(SIZE_CATS))
width = 0.35

reo_nnd = defective[defective.SUB_VENDOR=="REO"][nnd_cols].median().values
gh_nnd  = defective[defective.SUB_VENDOR=="GH"][nnd_cols].median().values

bars_r = ax.bar(x - width/2, reo_nnd, width,
                color=VENDOR_A, alpha=0.85,
                label="REO", zorder=3)
bars_g = ax.bar(x + width/2, gh_nnd,  width,
                color=VENDOR_B, alpha=0.85,
                label="GH",  zorder=3)

for bars, vals, color in [(bars_r, reo_nnd, VENDOR_A),
                           (bars_g, gh_nnd,  VENDOR_B)]:
    for bar, val in zip(bars, vals):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 3,
                    f"{val:.0f}px",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(SIZE_CATS)
ax.set_ylabel("Median NN distance (px)")
ax.set_title("Median NN distance per size", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)

# ── Right: xxlarge reversal highlight ─────────────────────
ax = axes[1]
styled_ax(ax)

# Ratio REO/GH per size — < 1 means REO closer, > 1 means GH closer
ratios = reo_nnd / (gh_nnd + 1e-9)
colors_bar = [VENDOR_A if r < 1 else VENDOR_B for r in ratios]

bars = ax.bar(x, ratios, color=colors_bar, alpha=0.85,
              width=0.5, zorder=3)
ax.axhline(1.0, color=MUTED, lw=1.2,
           linestyle="--", alpha=0.6,
           label="REO = GH")

for bar, val, sz in zip(bars, ratios, SIZE_CATS):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.2f}×",
            ha="center", va="bottom",
            fontsize=8.5, color=TEXT,
            fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(SIZE_CATS)
ax.set_ylabel("REO / GH median NN distance ratio")
ax.set_title("REO/GH ratio  (< 1 = REO closer together)",
             fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.8)

# Annotation
ax.text(0.03, 0.95,
        "Blue bars: REO defects closer\n"
        "Red bar:   GH defects closer (XXLarge)",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8, color=MUTED, style="italic")

fig.text(0.5, -0.03,
         "XXLarge reversal: GH XXLarge defects cluster tighter than REO. "
         "All other sizes: REO defects consistently closer together.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()
