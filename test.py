import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

# ─────────────────────────────────────────────────────────────
# 3D — Small defect radial profile: mean + median bars
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor=BG)
fig.suptitle("3D — Small Defect Radial Profile: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

small_cols  = [f"radial_small_ring_{i}" for i in range(9)]
ring_labels = [f"R{i}" for i in range(9)]
x     = np.arange(9)
width = 0.35

for ax, agg, title in zip(axes,
                           ["mean", "median"],
                           ["Mean small blob count per mirror per ring",
                            "Median small blob count per mirror per ring"]):
    styled_ax(ax)

    for vendor, color, offset in [
        ("REO", VENDOR_A, -width/2),
        ("GH",  VENDOR_B,  width/2),
    ]:
        vdf  = df[df.SUB_VENDOR == vendor]
        vals = (vdf[small_cols].mean() if agg == "mean"
                else vdf[small_cols].median()).values

        bars = ax.bar(x + offset, vals, width,
                      color=color, alpha=0.85,
                      label=vendor, zorder=3)

        # Value labels
        for bar, val in zip(bars, vals):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f"{val:.1f}",
                        ha="center", va="bottom",
                        fontsize=6.5, color=color,
                        rotation=90)

    # Zone shading
    ax.axvspan(-0.5, 2.5, alpha=0.04, color="#3B82F6", zorder=1)
    ax.axvspan(2.5,  5.5, alpha=0.04, color="#10B981", zorder=1)
    ax.axvspan(5.5,  8.5, alpha=0.04, color="#EF4444", zorder=1)

    for pos, lbl in [(1.0, "inner"), (4.0, "mid"), (7.0, "outer")]:
        ax.text(pos, 0.02, lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8,
                color=MUTED, alpha=0.6)

    # REO/GH ratio at R8
    reo_v = (df[df.SUB_VENDOR=="REO"][small_cols].mean()
             if agg=="mean"
             else df[df.SUB_VENDOR=="REO"][small_cols].median()).values
    gh_v  = (df[df.SUB_VENDOR=="GH"][small_cols].mean()
             if agg=="mean"
             else df[df.SUB_VENDOR=="GH"][small_cols].median()).values

    ratio = reo_v[8] / gh_v[8] if gh_v[8] > 0 else np.nan
    if not np.isnan(ratio):
        ax.text(0.98, 0.95,
                f"REO/GH at R8: {ratio:.1f}×",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=BG_CARD,
                          edgecolor=BORDER, alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(ring_labels)
    ax.set_xlabel("Ring  (R0 = center → R8 = edge)")
    ax.set_ylabel("Small blob count")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

fig.text(0.5, -0.02,
         "Small blobs: 1–25px. Outer ring (R9) excluded.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 3E — Radial by size: mean count + % mirrors with ≥1 blob
# ─────────────────────────────────────────────────────────────

SIZE_CATS = ["small", "medium", "large", "xlarge", "xxlarge"]

# Shades per vendor — light to dark = small to xxlarge
shades_reo = ["#A8D4F0", "#6BAED6", "#3A8CBF", "#1D6FA4", "#0D3F5E"]
shades_gh  = ["#F5A99A", "#E06B5A", "#C0392B", "#922B21", "#641E16"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG)
fig.suptitle("3E — Radial Profile by Size Category: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

ring_labels = [f"R{i}" for i in range(9)]

for col_idx, (vendor, shades) in enumerate([
    ("REO", shades_reo),
    ("GH",  shades_gh),
]):
    vdf = df[df.SUB_VENDOR == vendor]

    # ── Row 0: Mean count ─────────────────────────────────
    ax = axes[0][col_idx]
    styled_ax(ax)

    for sz, shade in zip(SIZE_CATS, shades):
        cols = [f"radial_{sz}_ring_{i}" for i in range(9)]
        vals = vdf[cols].mean().values
        ax.plot(range(9), vals, color=shade, lw=2.0,
                marker="o", markersize=4,
                label=sz, zorder=4)
        ax.fill_between(range(9), vals,
                        alpha=0.06, color=shade, zorder=3)

    ax.axvspan(0, 2.5, alpha=0.04, color="#3B82F6", zorder=1)
    ax.axvspan(2.5, 5.5, alpha=0.04, color="#10B981", zorder=1)
    ax.axvspan(5.5, 8.5, alpha=0.04, color="#EF4444", zorder=1)

    ax.set_xticks(range(9))
    ax.set_xticklabels(ring_labels)
    ax.set_xlabel("Ring  (R0 = center → R8 = edge)")
    ax.set_ylabel("Mean blob count per mirror")
    ax.set_title(f"Mean count — {vendor}",
                 fontsize=11, color=(VENDOR_A if vendor=="REO"
                                     else VENDOR_B),
                 fontweight="semibold")
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, title="Size",
              title_fontsize=8)

    # Colored top spine
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color(VENDOR_A if vendor=="REO" else VENDOR_B)
    ax.spines["top"].set_linewidth(2.5)

    # ── Row 1: % mirrors with ≥1 blob ────────────────────
    ax = axes[1][col_idx]
    styled_ax(ax)

    for sz, shade in zip(SIZE_CATS, shades):
        cols = [f"radial_{sz}_ring_{i}" for i in range(9)]
        pct  = (vdf[cols] > 0).mean().values * 100
        ax.plot(range(9), pct, color=shade, lw=2.0,
                marker="o", markersize=4,
                label=sz, zorder=4)
        ax.fill_between(range(9), pct,
                        alpha=0.06, color=shade, zorder=3)

    ax.axvspan(0,   2.5, alpha=0.04, color="#3B82F6", zorder=1)
    ax.axvspan(2.5, 5.5, alpha=0.04, color="#10B981", zorder=1)
    ax.axvspan(5.5, 8.5, alpha=0.04, color="#EF4444", zorder=1)

    ax.set_xticks(range(9))
    ax.set_xticklabels(ring_labels)
    ax.set_xlabel("Ring  (R0 = center → R8 = edge)")
    ax.set_ylabel("% mirrors with ≥1 blob")
    ax.set_title(f"% mirrors with ≥1 blob — {vendor}",
                 fontsize=11, color=(VENDOR_A if vendor=="REO"
                                     else VENDOR_B),
                 fontweight="semibold")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, title="Size",
              title_fontsize=8)

    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color(VENDOR_A if vendor=="REO" else VENDOR_B)
    ax.spines["top"].set_linewidth(2.5)

fig.text(0.5, -0.02,
         "Outer ring (R9) excluded. "
         "Row 1: mean count per mirror. "
         "Row 2: % mirrors with at least one blob of that size.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 3F — Inner vs outer ring correlation
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle("3F — Inner vs Outer Ring Coverage Correlation",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

inner_cols = [f"coverage_ring_{i}" for i in range(3)]    # R0-R2
outer_cols = [f"coverage_ring_{i}" for i in range(6, 9)] # R6-R8

df["_inner"] = df[inner_cols].mean(axis=1)
df["_outer"] = df[outer_cols].mean(axis=1)

# Clip at 95th pct for visual clarity
inner_cap = df["_inner"].quantile(0.95)
outer_cap = df["_outer"].quantile(0.95)

for ax, vendor, color in [
    (axes[0], "REO", VENDOR_A),
    (axes[1], "GH",  VENDOR_B),
]:
    styled_ax(ax)
    vdf = df[df.SUB_VENDOR == vendor].copy()

    inner = vdf["_inner"].clip(upper=inner_cap)
    outer = vdf["_outer"].clip(upper=outer_cap)
    corr  = vdf["_inner"].corr(vdf["_outer"])

    # Scatter — sample to avoid overplotting
    sample = vdf.sample(min(2000, len(vdf)), random_state=42)
    ax.scatter(sample["_inner"].clip(upper=inner_cap),
               sample["_outer"].clip(upper=outer_cap),
               color=color, alpha=0.12, s=6,
               zorder=3, linewidths=0)

    # Trend line
    mask = (inner > 0) & (outer > 0)
    if mask.sum() > 10:
        coeffs = np.polyfit(inner[mask], outer[mask], 1)
        x_line = np.linspace(0, inner_cap, 100)
        ax.plot(x_line, np.polyval(coeffs, x_line),
                color=color, lw=2.2, zorder=4,
                label=f"r = {corr:.2f}")

    # Diagonal reference — perfect correlation
    lim = max(inner_cap, outer_cap)
    ax.plot([0, lim], [0, lim], color=MUTED,
            lw=1.0, linestyle="--", alpha=0.5,
            label="perfect correlation")

    ax.set_xlabel("Mean inner ring coverage R0–R2 (%)")
    ax.set_ylabel("Mean outer ring coverage R6–R8 (%)")
    ax.set_title(vendor, fontsize=12,
                 color=color, fontweight="semibold")
    ax.set_xlim(0, inner_cap * 1.05)
    ax.set_ylim(0, outer_cap * 1.05)
    ax.legend(fontsize=9)

    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color(color)
    ax.spines["top"].set_linewidth(2.5)

    # Interpretation note
    ax.text(0.03, 0.95,
            "Dirty mirrors are dirty everywhere\n"
            "Clean mirrors are clean everywhere",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=7.5, color=MUTED, style="italic")

fig.text(0.5, -0.03,
         "One point per mirror. Clipped at 95th percentile for visual clarity. "
         "Outer ring (R9) excluded.",
         ha="center", fontsize=8, color=MUTED, style="italic")

# Cleanup temp columns
df.drop(columns=["_inner", "_outer"], inplace=True)

plt.tight_layout()
plt.show()
