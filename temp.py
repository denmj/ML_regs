# 5B — Class distribution over time + spike investigation

df["week"] = pd.to_datetime(df["serial_dt"]).dt.to_period("W").dt.start_time

CLASSES_DEFECT = ["fod", "polish defect", "wax", "scratch", "water spot"]

# Shades per class — consistent across both vendors
CLASS_COLORS = {
    "fod":           "#2196F3",
    "polish defect": "#FF9800",
    "wax":           "#9C27B0",
    "scratch":       "#F44336",
    "water spot":    "#00BCD4",
    "no defect":     "#9E9E9E",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG)
fig.suptitle("5B — Defect Class Distribution Over Time: REO vs GH",
             fontsize=13, fontweight="semibold", color=TEXT, y=1.01)

for row, (vendor, color) in enumerate([("REO", VENDOR_A),
                                        ("GH",  VENDOR_B)]):
    vdf = df[df.SUB_VENDOR == vendor]

    # ── Left: stacked area — class % per week ─────────────
    ax = axes[row][0]
    styled_ax(ax)

    weekly_total = vdf.groupby("week").size()
    weekly_class = (vdf.groupby(["week", "predicted_defects"])
                      .size()
                      .unstack(fill_value=0))

    # % per class per week
    weekly_pct = weekly_class.div(weekly_total, axis=0) * 100

    # Ensure all classes present
    for cls in CLASSES_DEFECT + ["no defect"]:
        if cls not in weekly_pct.columns:
            weekly_pct[cls] = 0

    # Stack order — no defect at bottom
    stack_order = ["no defect"] + CLASSES_DEFECT
    bottom = np.zeros(len(weekly_pct))

    for cls in stack_order:
        vals = weekly_pct[cls].values
        ax.fill_between(weekly_pct.index, bottom,
                        bottom + vals,
                        color=CLASS_COLORS[cls],
                        alpha=0.75, label=cls,
                        step="mid", zorder=3)
        bottom += vals

    # Spike markers
    for spike_date, lbl in [
        (pd.Timestamp("2026-01-05"), "Jan 5"),
        (pd.Timestamp("2026-03-09"), "Mar 9"),
    ]:
        if spike_date in weekly_pct.index:
            ax.axvline(spike_date, color=TEXT,
                       lw=1.2, linestyle=":",
                       alpha=0.7, zorder=5)
            ax.text(spike_date, 102, lbl,
                    color=TEXT, fontsize=7.5,
                    ha="center", va="top")

    ax.set_xlabel("Week")
    ax.set_ylabel("% of mirrors")
    ax.set_ylim(0, 105)
    ax.set_title(f"{vendor} — Weekly class composition",
                 fontsize=11, color=color,
                 fontweight="semibold")
    ax.tick_params(axis="x", rotation=45)

    # Legend only on first row
    if row == 0:
        ax.legend(fontsize=7.5, loc="upper left",
                  facecolor=BG_PANEL, edgecolor=BORDER,
                  ncol=2)

    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color(color)
    ax.spines["top"].set_linewidth(2.5)

    # ── Right: spike weeks vs normal weeks ────────────────
    ax = axes[row][1]
    styled_ax(ax)

    spike_weeks = [pd.Timestamp("2026-01-05"),
                   pd.Timestamp("2026-03-09")]

    spike_df  = vdf[vdf["week"].isin(spike_weeks)]
    normal_df = vdf[~vdf["week"].isin(spike_weeks)]

    x     = np.arange(len(CLASSES_DEFECT))
    width = 0.35

    spike_pcts  = [(spike_df["predicted_defects"] == cls).mean() * 100
                   for cls in CLASSES_DEFECT]
    normal_pcts = [(normal_df["predicted_defects"] == cls).mean() * 100
                   for cls in CLASSES_DEFECT]

    # Shades — spike darker, normal lighter
    spike_color  = color
    normal_color = "#A8D4F0" if vendor == "REO" else "#F5A99A"

    bars_s = ax.bar(x - width/2, spike_pcts,  width,
                    color=spike_color, alpha=0.90,
                    label="Spike weeks\n(Jan 5 + Mar 9)",
                    zorder=3)
    bars_n = ax.bar(x + width/2, normal_pcts, width,
                    color=normal_color, alpha=0.90,
                    label="Normal weeks",
                    zorder=3)

    for bars, vals, c in [(bars_s, spike_pcts,  spike_color),
                           (bars_n, normal_pcts, normal_color)]:
        for bar, val in zip(bars, vals):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}%",
                        ha="center", va="bottom",
                        fontsize=7.5, color=c,
                        rotation=90)

    # Delta annotation
    for i, (s, n, cls) in enumerate(zip(spike_pcts,
                                         normal_pcts,
                                         CLASSES_DEFECT)):
        diff = s - n
        if abs(diff) > 2:
            ax.text(i, max(s, n) + 2,
                    f"{'+' if diff > 0 else ''}{diff:.1f}%",
                    ha="center", fontsize=7.5,
                    color=TEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES_DEFECT, rotation=20,
                        ha="right", fontsize=9)
    ax.set_ylabel("% of mirrors")
    ax.set_title(f"{vendor} — Spike vs normal weeks",
                 fontsize=11, color=color,
                 fontweight="semibold")
    ax.legend(fontsize=8, facecolor=BG_PANEL,
              edgecolor=BORDER)
    ax.set_ylim(0, 65)

    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color(color)
    ax.spines["top"].set_linewidth(2.5)

fig.text(0.5, -0.02,
         "Spike weeks defined as Jan 5 and Mar 9 — "
         "weeks with highest average defect count per mirror. "
         "Delta shows class shift during spike weeks vs normal.",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.show()
