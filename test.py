import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=BG)
fig.suptitle("Radial Coverage Mirror Maps: REO vs GH",
             fontsize=14, fontweight="semibold", color=TEXT, y=1.01)

ring_cols  = [f"coverage_ring_{i}" for i in range(9)]
VMAX       = 1.0   # shared colorbar cap (%)
CMAP       = "RdYlGn_r"   # green=clean, red=dirty
cmap       = cm.get_cmap(CMAP)
norm       = mcolors.Normalize(vmin=0, vmax=VMAX)

vendors    = ["REO", "GH"]
agg_funcs  = ["mean", "median"]
titles_col = ["REO", "GH"]
titles_row = ["Mean", "Median"]
v_colors   = [VENDOR_A, VENDOR_B]

for row, agg in enumerate(agg_funcs):
    for col, (vendor, vcolor) in enumerate(zip(vendors, v_colors)):
        ax = axes[row][col]
        ax.set_facecolor(BG_PANEL)
        ax.set_aspect("equal")

        vdf  = df[df.SUB_VENDOR == vendor]
        vals = (vdf[ring_cols].mean() if agg == "mean"
                else vdf[ring_cols].median()).values   # shape (9,)

        worst_ring = int(np.argmax(vals))

        # Draw rings from outside in (R8 → R0)
        # Each ring is a filled circle, outer drawn first
        n_rings = len(vals)
        for i in range(n_rings - 1, -1, -1):
            r_outer = (i + 1) / n_rings   # normalized radius
            cval    = np.clip(vals[i] / VMAX, 0, 1)
            color   = cmap(cval)

            circle = plt.Circle((0, 0), r_outer,
                                 color=color, zorder=i + 1)
            ax.add_patch(circle)

            # Ring boundary
            ring_border = plt.Circle((0, 0), r_outer,
                                      fill=False,
                                      edgecolor="white",
                                      linewidth=0.6,
                                      linestyle="--",
                                      alpha=0.5,
                                      zorder=i + 2)
            ax.add_patch(ring_border)

        # Annotate ring values
        for i in range(n_rings):
            r_mid  = ((i + 0.5) / n_rings)
            label  = f"R{i}\n{vals[i]:.3f}%"
            cval   = np.clip(vals[i] / VMAX, 0, 1)
            # White text on dark rings, dark text on light rings
            txt_color = "white" if cval > 0.5 else "#333333"
            ax.text(0, r_mid, label,
                    ha="center", va="center",
                    fontsize=6.5, color=txt_color,
                    fontweight="bold", zorder=20)

        # Mirror boundary
        boundary = plt.Circle((0, 0), 1.0,
                               fill=False,
                               edgecolor=vcolor,
                               linewidth=2.0,
                               zorder=25)
        ax.add_patch(boundary)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.axis("off")

        # Title
        ax.set_title(
            f"{titles_row[row]} — {vendor}  "
            f"(worst: R{worst_ring}={vals[worst_ring]:.3f}%)",
            fontsize=11, color=vcolor,
            fontweight="semibold", pad=8
        )

        # Vendor colored top border
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_color(vcolor)
        ax.spines["top"].set_linewidth(3)

# Shared colorbar
sm  = cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Coverage %", color=TEXT, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)

fig.text(0.5, -0.02,
         "Coverage % of defect pixels per ring. "
         "Shared colorbar capped at 1.0% (95th pct REO). "
         "Outer ring (R9) excluded.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.show()





fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor=BG)
fig.suptitle("Radial Coverage Profile: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

ring_cols = [f"coverage_ring_{i}" for i in range(9)]
ring_labels = [f"R{i}" for i in range(9)]
x = np.arange(9)
width = 0.35

for ax, agg, title in zip(axes,
                           ["mean",   "median"],
                           ["Mean coverage per ring",
                            "Median coverage per ring"]):
    styled_ax(ax)

    for vendor, color, offset in [
        ("REO", VENDOR_A, -width/2),
        ("GH",  VENDOR_B,  width/2),
    ]:
        vdf  = df[df.SUB_VENDOR == vendor]
        vals = (vdf[ring_cols].mean() if agg == "mean"
                else vdf[ring_cols].median()).values

        bars = ax.bar(x + offset, vals, width,
                      color=color, alpha=0.85,
                      label=vendor, zorder=3)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=6.5, color=color,
                        rotation=90)

    # Zone shading
    ax.axvspan(-0.5, 2.5, alpha=0.04, color="#3B82F6", zorder=1)
    ax.axvspan(2.5,  5.5, alpha=0.04, color="#10B981", zorder=1)
    ax.axvspan(5.5,  8.5, alpha=0.04, color="#EF4444", zorder=1)

    for pos, lbl in [(1.0, "inner"), (4.0, "mid"), (7.0, "outer")]:
        ax.text(pos, 0, lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=8, color=MUTED, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(ring_labels)
    ax.set_xlabel("Ring  (R0 = center → R8 = edge,  750px radius)")
    ax.set_ylabel("Coverage %")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    # Annotate gap ratio at worst ring
    reo_vals = (df[df.SUB_VENDOR=="REO"][ring_cols].mean()
                if agg == "mean"
                else df[df.SUB_VENDOR=="REO"][ring_cols].median()).values
    gh_vals  = (df[df.SUB_VENDOR=="GH"][ring_cols].mean()
                if agg == "mean"
                else df[df.SUB_VENDOR=="GH"][ring_cols].median()).values

    worst = int(np.argmax(reo_vals))
    ratio = reo_vals[worst] / gh_vals[worst] if gh_vals[worst] > 0 else np.nan
    if not np.isnan(ratio):
        ax.text(0.98, 0.95,
                f"REO/GH at R{worst}: {ratio:.1f}×",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=BG_CARD,
                          edgecolor=BORDER, alpha=0.8))

fig.text(0.5, -0.02,
         "Coverage % of defect pixels per ring. "
         "Outer ring (R9) excluded throughout.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()
