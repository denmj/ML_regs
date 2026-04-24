# ─────────────────────────────────────────────────────────────
# 5D — Coverage and count per defect class per vendor
# ─────────────────────────────────────────────────────────────

CLASSES_PLOT = ["fod", "polish defect", "wax",
                "scratch", "water spot"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG)
fig.suptitle("5D — Coverage & Count per Defect Class: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

# ── Top left: mean coverage per class ─────────────────────
ax = axes[0][0]
styled_ax(ax)

x     = np.arange(len(CLASSES_PLOT))
width = 0.35

for vendor, color, offset in [("REO", VENDOR_A, -width/2),
                                ("GH",  VENDOR_B,  width/2)]:
    vdf  = df[df.SUB_VENDOR == vendor]
    vals = [vdf[vdf.predicted_defects == cls]
            ["total_coverage_excl_last"].mean()
            for cls in CLASSES_PLOT]

    bars = ax.bar(x + offset, vals, width,
                  color=color, alpha=0.85,
                  label=vendor, zorder=3)

    for bar, val in zip(bars, vals):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES_PLOT, rotation=20, ha="right")
ax.set_ylabel("Mean coverage % (excl. outer ring)")
ax.set_title("Mean coverage per class", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)

# ── Top right: median coverage per class ──────────────────
ax = axes[0][1]
styled_ax(ax)

for vendor, color, offset in [("REO", VENDOR_A, -width/2),
                                ("GH",  VENDOR_B,  width/2)]:
    vdf  = df[df.SUB_VENDOR == vendor]
    vals = [vdf[vdf.predicted_defects == cls]
            ["total_coverage_excl_last"].median()
            for cls in CLASSES_PLOT]

    bars = ax.bar(x + offset, vals, width,
                  color=color, alpha=0.85,
                  label=vendor, zorder=3)

    for bar, val in zip(bars, vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES_PLOT, rotation=20, ha="right")
ax.set_ylabel("Median coverage % (excl. outer ring)")
ax.set_title("Median coverage per class", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)

# ── Bottom left: mean defect count per class ──────────────
ax = axes[1][0]
styled_ax(ax)

for vendor, color, offset in [("REO", VENDOR_A, -width/2),
                                ("GH",  VENDOR_B,  width/2)]:
    vdf  = df[df.SUB_VENDOR == vendor]
    vals = [vdf[vdf.predicted_defects == cls]
            ["defect_count_excl_last"].mean()
            for cls in CLASSES_PLOT]

    bars = ax.bar(x + offset, vals, width,
                  color=color, alpha=0.85,
                  label=vendor, zorder=3)

    for bar, val in zip(bars, vals):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"{val:.0f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES_PLOT, rotation=20, ha="right")
ax.set_ylabel("Mean defect count (excl. outer ring)")
ax.set_title("Mean defect count per class", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)

# ── Bottom right: % clean mirrors per class ───────────────
ax = axes[1][1]
styled_ax(ax)

for vendor, color, offset in [("REO", VENDOR_A, -width/2),
                                ("GH",  VENDOR_B,  width/2)]:
    vdf  = df[df.SUB_VENDOR == vendor]
    vals = [(vdf[vdf.predicted_defects == cls]
             ["total_coverage_excl_last"] < 0.05).mean() * 100
            for cls in CLASSES_PLOT]

    bars = ax.bar(x + offset, vals, width,
                  color=color, alpha=0.85,
                  label=vendor, zorder=3)

    for bar, val in zip(bars, vals):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES_PLOT, rotation=20, ha="right")
ax.set_ylabel("% mirrors with coverage < 0.05%")
ax.set_title("% essentially clean mirrors per class",
             fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 105)

fig.text(0.5, -0.02,
         "No defect class excluded — shown separately. "
         "Outer ring (R9) excluded throughout.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 5E — Worst ring per defect class
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle("5E — Worst Ring per Defect Class: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

# ── Left: median worst ring per class ─────────────────────
ax = axes[0]
styled_ax(ax)

for vendor, color, offset in [("REO", VENDOR_A, -width/2),
                                ("GH",  VENDOR_B,  width/2)]:
    vdf  = df[df.SUB_VENDOR == vendor]
    vals = [vdf[vdf.predicted_defects == cls]
            ["worst_ring"].median()
            for cls in CLASSES_PLOT]

    bars = ax.bar(x + offset, vals, width,
                  color=color, alpha=0.85,
                  label=vendor, zorder=3)

    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f"R{val:.0f}",
                    ha="center", va="bottom",
                    fontsize=8, color=color,
                    fontweight="bold")

# Zone shading on y axis
ax.axhspan(0,   3, alpha=0.04, color="#3B82F6", zorder=0)
ax.axhspan(3,   6, alpha=0.04, color="#10B981", zorder=0)
ax.axhspan(6,   8.5, alpha=0.04, color="#EF4444", zorder=0)

for y, lbl in [(1.5, "inner"), (4.5, "mid"), (7.2, "outer")]:
    ax.text(len(CLASSES_PLOT) - 0.3, y, lbl,
            fontsize=7.5, color=MUTED,
            alpha=0.7, va="center")

ax.set_xticks(x)
ax.set_xticklabels(CLASSES_PLOT, rotation=20, ha="right")
ax.set_ylabel("Median worst ring  (0=center, 8=edge)")
ax.set_title("Median worst ring per class", fontsize=11)
ax.set_ylim(0, 9)
ax.set_yticks(range(9))
ax.set_yticklabels([f"R{i}" for i in range(9)])
ax.legend(fontsize=9)

# ── Right: worst ring distribution heatmap per class ──────
ax = axes[1]
ax.set_facecolor(BG_PANEL)

# Build matrix: class × ring → % of mirrors
vendors_combined = ["REO", "GH"]
n_classes = len(CLASSES_PLOT)

for vi, (vendor, cmap_name) in enumerate([
    ("REO", "Blues"),
    ("GH",  "Reds"),
]):
    vdf = df[df.SUB_VENDOR == vendor]

    matrix = np.zeros((n_classes, 9))
    for ci, cls in enumerate(CLASSES_PLOT):
        cls_df = vdf[vdf.predicted_defects == cls]
        if len(cls_df) == 0:
            continue
        for ri in range(9):
            matrix[ci, ri] = (cls_df["worst_ring"] == ri).mean() * 100

    # Offset heatmap — REO left half, GH right half
    x_offset = 0 if vendor == "REO" else 9.5
    im = ax.imshow(matrix,
                   cmap=cmap_name,
                   aspect="auto",
                   vmin=0, vmax=40,
                   extent=[x_offset, x_offset + 9,
                            -0.5, n_classes - 0.5])

    # Annotate cells
    for ci in range(n_classes):
        for ri in range(9):
            val = matrix[ci, ri]
            if val > 5:
                ax.text(x_offset + ri + 0.5,
                        ci, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=6.5,
                        color="white" if val > 20 else TEXT)

ax.set_yticks(range(n_classes))
ax.set_yticklabels(CLASSES_PLOT, fontsize=9)
ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5,
               5.5, 6.5, 7.5, 8.5,
               10, 11, 12, 13, 14,
               15, 16, 17, 18])
ax.set_xticklabels(
    [f"R{i}" for i in range(9)] +
    [f"R{i}" for i in range(9)],
    fontsize=7, rotation=45
)
ax.axvline(9.25, color=MUTED, lw=1.5,
           linestyle="--", alpha=0.6)
ax.text(4.5,  n_classes - 0.3, "REO",
        ha="center", color=VENDOR_A,
        fontsize=10, fontweight="bold")
ax.text(14.0, n_classes - 0.3, "GH",
        ha="center", color=VENDOR_B,
        fontsize=10, fontweight="bold")

ax.set_title("Worst ring distribution heatmap\n"
             "% of mirrors per class per ring",
             fontsize=11)
ax.grid(False)

fig.text(0.5, -0.03,
         "Wax worst ring peaks at R7-R8 (edge). "
         "REO FOD peaks at R7. "
         "Scratch and Polish Defect concentrate mid-to-outer (R5-R6).",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()
