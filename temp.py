# 5C — Radial profile per defect class

CLASSES_ALL = ["fod", "no defect", "polish defect",
               "wax", "scratch", "water spot"]

CLASS_COLORS = {
    "fod":           "#2196F3",
    "polish defect": "#FF9800",
    "wax":           "#9C27B0",
    "scratch":       "#F44336",
    "water spot":    "#00BCD4",
    "no defect":     "#9E9E9E",
}

ring_cols   = [f"coverage_ring_{i}" for i in range(9)]
ring_labels = [f"R{i}" for i in range(9)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("5C — Radial Coverage Profile per Defect Class",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

linestyles = {
    "fod":           "-",
    "polish defect": "--",
    "wax":           "-.",
    "scratch":       ":",
    "water spot":    (0, (3, 1)),
    "no defect":     (0, (5, 2)),
}

for ax, (vendor, color) in zip(axes, [("REO", VENDOR_A),
                                        ("GH",  VENDOR_B)]):
    styled_ax(ax)

    vdf = df[df.SUB_VENDOR == vendor]

    for cls in CLASSES_ALL:
        cls_df = vdf[vdf.predicted_defects == cls]
        n      = len(cls_df)

        # Skip classes with too few samples
        if n < 20:
            continue

        vals = cls_df[ring_cols].median().values

        ax.plot(range(9), vals,
                color=CLASS_COLORS[cls],
                lw=2.2,
                linestyle=linestyles[cls],
                marker="o", markersize=4,
                label=f"{cls}  (n={n:,})",
                zorder=4)
        ax.fill_between(range(9), vals,
                        alpha=0.06,
                        color=CLASS_COLORS[cls],
                        zorder=3)

    # Zone shading
    ax.axvspan(0,   2.5, alpha=0.04, color="#3B82F6", zorder=1)
    ax.axvspan(2.5, 5.5, alpha=0.04, color="#10B981", zorder=1)
    ax.axvspan(5.5, 8.5, alpha=0.04, color="#EF4444", zorder=1)

    for pos, lbl in [(1.0, "inner"), (4.0, "mid"), (7.0, "outer")]:
        ax.text(pos, 0.02, lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", fontsize=7.5,
                color=MUTED, alpha=0.6)

    ax.set_xticks(range(9))
    ax.set_xticklabels(ring_labels)
    ax.set_xlabel("Ring  (R0 = center → R8 = edge,  750px radius)")
    ax.set_ylabel("Median coverage %")
    ax.set_title(vendor, fontsize=12,
                 color=color, fontweight="semibold")
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, facecolor=BG_PANEL,
              edgecolor=BORDER, loc="upper left")

    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color(color)
    ax.spines["top"].set_linewidth(2.5)

fig.text(0.5, -0.03,
         "Median coverage per ring. Classes with n<20 excluded. "
         "Outer ring (R9) excluded. "
         "FOD and Wax show strongest edge concentration.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()
