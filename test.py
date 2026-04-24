# ─────────────────────────────────────────────────────────────
# 7B — Worst mirrors — top 5% by coverage
# ─────────────────────────────────────────────────────────────

threshold_7b = df["total_coverage_excl_last"].quantile(0.95)
worst        = df[df.total_coverage_excl_last >= threshold_7b]
rest         = df[df.total_coverage_excl_last <  threshold_7b]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG)
fig.suptitle(
    f"7B — Worst Mirrors  (top 5% by coverage  |  "
    f"threshold: {threshold_7b:.3f}%  |  n={len(worst):,})",
    fontsize=13, fontweight="semibold",
    color=TEXT, y=1.01
)

# ── 1: Vendor split ───────────────────────────────────────
ax = axes[0][0]
styled_ax(ax)

vendor_pct_worst = worst.SUB_VENDOR.value_counts(normalize=True) * 100
vendor_pct_all   = df.SUB_VENDOR.value_counts(normalize=True)    * 100

x     = np.arange(2)
width = 0.35
vendors = ["REO", "GH"]

bars_w = ax.bar(x - width/2,
                [vendor_pct_worst.get(v, 0) for v in vendors],
                width, color=VENDOR_A, alpha=0.85,
                label="Worst 5%", zorder=3)
bars_a = ax.bar(x + width/2,
                [vendor_pct_all.get(v, 0) for v in vendors],
                width, color=MUTED, alpha=0.60,
                label="All mirrors", zorder=3)

for bars in [bars_w, bars_a]:
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=9, color=TEXT,
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(vendors, fontsize=11)
ax.set_ylabel("% of group")
ax.set_title("Vendor split\nworst 5% vs all mirrors",
             fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=8)

# ── 2: Class split ────────────────────────────────────────
ax = axes[0][1]
styled_ax(ax)

class_worst = (worst.predicted_defects
               .value_counts(normalize=True) * 100)
class_all   = (df.predicted_defects
               .value_counts(normalize=True) * 100)

classes = ["fod", "polish defect", "wax",
           "scratch", "water spot", "no defect"]
colors  = [CLASS_COLORS_MAP[c] for c in classes]
x       = np.arange(len(classes))

bars_w = ax.bar(x - width/2,
                [class_worst.get(c, 0) for c in classes],
                width, color=colors, alpha=0.85,
                label="Worst 5%", zorder=3)
bars_a = ax.bar(x + width/2,
                [class_all.get(c, 0) for c in classes],
                width, color=colors, alpha=0.40,
                label="All mirrors", zorder=3)

for bar in bars_w:
    val = bar.get_height()
    if val > 2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=7.5, color=TEXT,
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=20,
                    ha="right", fontsize=8)
ax.set_ylabel("% of group")
ax.set_title("Class split\nworst 5% vs all mirrors",
             fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=8)

# ── 3: Radial profile worst vs rest ──────────────────────
ax = axes[0][2]
styled_ax(ax)

ring_cols = [f"coverage_ring_{i}" for i in range(9)]

worst_radial = worst[ring_cols].median().values
rest_radial  = rest[ring_cols].median().values

ax.plot(range(9), worst_radial,
        color="#E74C3C", lw=2.2,
        marker="o", markersize=5,
        label=f"Worst 5%  (n={len(worst):,})",
        zorder=4)
ax.fill_between(range(9), worst_radial,
                alpha=0.10, color="#E74C3C")

ax.plot(range(9), rest_radial,
        color=MUTED, lw=2.2,
        marker="o", markersize=5,
        label=f"Rest  (n={len(rest):,})",
        zorder=4)
ax.fill_between(range(9), rest_radial,
                alpha=0.10, color=MUTED)

ax.axvspan(0,   2.5, alpha=0.04, color="#3B82F6")
ax.axvspan(2.5, 5.5, alpha=0.04, color="#10B981")
ax.axvspan(5.5, 8.5, alpha=0.04, color="#EF4444")

ax.set_xticks(range(9))
ax.set_xticklabels([f"R{i}" for i in range(9)])
ax.set_xlabel("Ring (R0=center → R8=edge)")
ax.set_ylabel("Median coverage %")
ax.set_title("Radial profile\nworst 5% vs rest",
             fontsize=11)
ax.legend(fontsize=8)

# ── 4: Key metrics comparison ─────────────────────────────
ax = axes[1][0]
styled_ax(ax)

metrics = {
    "Mean coverage %":    "total_coverage_excl_last",
    "Mean defect count":  "defect_count_excl_last",
    "Mean weighted sum":  "_weighted_sum",
    "Mean CE-R":          "nn_clark_evans_r",
    "Mean NN dist (px)":  "nn_mean_nnd",
}

worst_vals = [worst[col].mean() for col in metrics.values()]
rest_vals  = [rest[col].mean()  for col in metrics.values()]

# Normalize for visual comparison
max_vals   = [max(w, r) for w, r in zip(worst_vals, rest_vals)]
worst_norm = [w/m if m > 0 else 0
              for w, m in zip(worst_vals, max_vals)]
rest_norm  = [r/m if m > 0 else 0
              for r, m in zip(rest_vals,  max_vals)]

x = np.arange(len(metrics))
ax.barh(x + width/2, worst_norm, width,
        color="#E74C3C", alpha=0.85,
        label="Worst 5%", zorder=3)
ax.barh(x - width/2, rest_norm,  width,
        color=MUTED, alpha=0.60,
        label="Rest", zorder=3)

# Raw value annotations
for i, (wv, rv) in enumerate(zip(worst_vals, rest_vals)):
    ax.text(1.02, i + width/2,
            f"{wv:.2f}",
            va="center", fontsize=7.5,
            color="#E74C3C")
    ax.text(1.02, i - width/2,
            f"{rv:.2f}",
            va="center", fontsize=7.5,
            color=MUTED)

ax.set_yticks(x)
ax.set_yticklabels(list(metrics.keys()), fontsize=9)
ax.set_xlabel("Normalized value (1 = max)")
ax.set_title("Key metrics\nworst 5% vs rest",
             fontsize=11)
ax.set_xlim(0, 1.3)
ax.legend(fontsize=8)

# ── 5: Worst ring distribution ────────────────────────────
ax = axes[1][1]
styled_ax(ax)

bins = np.arange(-0.5, 9.5, 1)
ax.hist(worst.worst_ring.dropna(), bins=bins,
        color="#E74C3C", alpha=0.75, density=True,
        label=f"Worst 5%  "
              f"μ={worst.worst_ring.mean():.1f}")
ax.hist(rest.worst_ring.dropna(),  bins=bins,
        color=MUTED, alpha=0.60, density=True,
        label=f"Rest  "
              f"μ={rest.worst_ring.mean():.1f}")

ax.axvspan(0,   2.5, alpha=0.04, color="#3B82F6")
ax.axvspan(2.5, 5.5, alpha=0.04, color="#10B981")
ax.axvspan(5.5, 8.5, alpha=0.04, color="#EF4444")

ax.set_xticks(range(9))
ax.set_xticklabels([f"R{i}" for i in range(9)])
ax.set_xlabel("Worst ring index")
ax.set_ylabel("Density")
ax.set_title("Worst ring distribution\nworst 5% vs rest",
             fontsize=11)
ax.legend(fontsize=8)

# ── 6: Weekly worst mirror count ──────────────────────────
ax = axes[1][2]
styled_ax(ax)

df["week"] = (pd.to_datetime(df["serial_dt"])
              .dt.to_period("W").dt.start_time)

for vendor, color in [("REO", VENDOR_A), ("GH", VENDOR_B)]:
    weekly_worst = (
        df[(df.SUB_VENDOR == vendor) &
           (df.total_coverage_excl_last >= threshold_7b)]
        .groupby("week").size()
        .reset_index(name="count")
    )
    weekly_total = (
        df[df.SUB_VENDOR == vendor]
        .groupby("week").size()
        .reset_index(name="total")
    )
    weekly = weekly_worst.merge(weekly_total, on="week")
    weekly["pct"] = weekly["count"] / weekly["total"] * 100

    ax.plot(weekly["week"], weekly["pct"],
            color=color, lw=2.2,
            marker="o", markersize=5,
            label=f"{vendor}  "
                  f"YTD={len(df[(df.SUB_VENDOR==vendor) & (df.total_coverage_excl_last>=threshold_7b)])/len(df[df.SUB_VENDOR==vendor])*100:.1f}%",
            zorder=4)
    ax.fill_between(weekly["week"], weekly["pct"],
                    alpha=0.08, color=color)

ax.set_xlabel("Week")
ax.set_ylabel("% of weekly mirrors in worst 5%")
ax.set_title("Worst mirror rate over time",
             fontsize=11)
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=8)

fig.text(0.5, -0.02,
         f"Worst 5% defined as coverage ≥ {threshold_7b:.3f}%. "
         "Outer ring (R9) excluded.",
         ha="center", fontsize=8,
         color=MUTED, style="italic")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 7C — Clean mirrors — bottom 25% by coverage
# ─────────────────────────────────────────────────────────────

threshold_7c = df["total_coverage_excl_last"].quantile(0.25)
clean        = df[df.total_coverage_excl_last <= threshold_7c]
dirty        = df[df.total_coverage_excl_last >  threshold_7c]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG)
fig.suptitle(
    f"7C — Clean Mirrors  (bottom 25% by coverage  |  "
    f"threshold: ≤{threshold_7c:.3f}%  |  n={len(clean):,})",
    fontsize=13, fontweight="semibold",
    color=TEXT, y=1.01
)

# ── 1: Vendor split clean vs all ─────────────────────────
ax = axes[0][0]
styled_ax(ax)

vendor_pct_clean = (clean.SUB_VENDOR
                    .value_counts(normalize=True) * 100)
vendor_pct_all   = (df.SUB_VENDOR
                    .value_counts(normalize=True) * 100)

bars_c = ax.bar(x - width/2,
                [vendor_pct_clean.get(v, 0) for v in vendors],
                width, color=[VENDOR_A, VENDOR_B],
                alpha=0.85, label="Clean 25%", zorder=3)
bars_a = ax.bar(x + width/2,
                [vendor_pct_all.get(v, 0) for v in vendors],
                width, color=MUTED, alpha=0.50,
                label="All mirrors", zorder=3)

for bars in [bars_c, bars_a]:
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=9, color=TEXT,
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(vendors, fontsize=11)
ax.set_ylabel("% of group")
ax.set_title("Vendor split\nclean 25% vs all mirrors",
             fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=8)

# ── 2: Class split clean ──────────────────────────────────
ax = axes[0][1]
styled_ax(ax)

class_clean = (clean.predicted_defects
               .value_counts(normalize=True) * 100)
class_all   = (df.predicted_defects
               .value_counts(normalize=True) * 100)

bars_c = ax.bar(x - width/2,
                [class_clean.get(c, 0) for c in classes],
                width, color=colors, alpha=0.85,
                label="Clean 25%", zorder=3)
bars_a = ax.bar(x + width/2,
                [class_all.get(c, 0) for c in classes],
                width, color=colors, alpha=0.35,
                label="All mirrors", zorder=3)

for bar in bars_c:
    val = bar.get_height()
    if val > 2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=7.5, color=TEXT,
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=20,
                    ha="right", fontsize=8)
ax.set_ylabel("% of group")
ax.set_title("Class split\nclean 25% vs all mirrors",
             fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=8)

# ── 3: Radial profile clean vs dirty ─────────────────────
ax = axes[0][2]
styled_ax(ax)

clean_radial = clean[ring_cols].median().values
dirty_radial = dirty[ring_cols].median().values

ax.plot(range(9), clean_radial,
        color="#2ECC71", lw=2.2,
        marker="o", markersize=5,
        label=f"Clean 25%  (n={len(clean):,})",
        zorder=4)
ax.fill_between(range(9), clean_radial,
                alpha=0.10, color="#2ECC71")

ax.plot(range(9), dirty_radial,
        color=MUTED, lw=2.2,
        marker="o", markersize=5,
        label=f"Rest  (n={len(dirty):,})",
        zorder=4)
ax.fill_between(range(9), dirty_radial,
                alpha=0.10, color=MUTED)

ax.axvspan(0,   2.5, alpha=0.04, color="#3B82F6")
ax.axvspan(2.5, 5.5, alpha=0.04, color="#10B981")
ax.axvspan(5.5, 8.5, alpha=0.04, color="#EF4444")

ax.set_xticks(range(9))
ax.set_xticklabels([f"R{i}" for i in range(9)])
ax.set_xlabel("Ring (R0=center → R8=edge)")
ax.set_ylabel("Median coverage %")
ax.set_title("Radial profile\nclean 25% vs rest",
             fontsize=11)
ax.legend(fontsize=8)

# ── 4: REO clean vs GH clean comparison ──────────────────
ax = axes[1][0]
styled_ax(ax)

reo_clean = clean[clean.SUB_VENDOR == "REO"]
gh_clean  = clean[clean.SUB_VENDOR == "GH"]

metrics_clean = {
    "Coverage %":         "total_coverage_excl_last",
    "Defect count":       "defect_count_excl_last",
    "Small count":        "count_small_excl_last",
    "XLarge count":       "count_xlarge_excl_last",
    "CE-R":               "nn_clark_evans_r",
    "NN distance (px)":   "nn_mean_nnd",
}

reo_c_vals = [reo_clean[col].mean()
              for col in metrics_clean.values()]
gh_c_vals  = [gh_clean[col].mean()
              for col in metrics_clean.values()]

max_c_vals = [max(r, g) for r, g in
              zip(reo_c_vals, gh_c_vals)]
reo_c_norm = [r/m if m > 0 else 0
              for r, m in zip(reo_c_vals, max_c_vals)]
gh_c_norm  = [g/m if m > 0 else 0
              for g, m in zip(gh_c_vals,  max_c_vals)]

x_c = np.arange(len(metrics_clean))
ax.barh(x_c + width/2, reo_c_norm, width,
        color=VENDOR_A, alpha=0.85,
        label=f"REO clean  n={len(reo_clean):,}",
        zorder=3)
ax.barh(x_c - width/2, gh_c_norm,  width,
        color=VENDOR_B, alpha=0.85,
        label=f"GH clean   n={len(gh_clean):,}",
        zorder=3)

for i, (rv, gv) in enumerate(zip(reo_c_vals, gh_c_vals)):
    ax.text(1.02, i + width/2,
            f"{rv:.2f}",
            va="center", fontsize=7.5,
            color=VENDOR_A)
    ax.text(1.02, i - width/2,
            f"{gv:.2f}",
            va="center", fontsize=7.5,
            color=VENDOR_B)

ax.set_yticks(x_c)
ax.set_yticklabels(list(metrics_clean.keys()),
                    fontsize=9)
ax.set_xlabel("Normalized value (1 = max)")
ax.set_title("REO clean vs GH clean\nare they equally clean?",
             fontsize=11)
ax.set_xlim(0, 1.3)
ax.legend(fontsize=8)

# ── 5: Clean rate per vendor over time ────────────────────
ax = axes[1][1]
styled_ax(ax)

for vendor, color in [("REO", VENDOR_A), ("GH", VENDOR_B)]:
    weekly_clean = (
        df[(df.SUB_VENDOR == vendor) &
           (df.total_coverage_excl_last <= threshold_7c)]
        .groupby("week").size()
        .reset_index(name="count")
    )
    weekly_total = (
        df[df.SUB_VENDOR == vendor]
        .groupby("week").size()
        .reset_index(name="total")
    )
    weekly = weekly_clean.merge(weekly_total, on="week")
    weekly["pct"] = weekly["count"] / weekly["total"] * 100

    ytd_clean_pct = (
        len(df[(df.SUB_VENDOR==vendor) &
               (df.total_coverage_excl_last<=threshold_7c)]) /
        len(df[df.SUB_VENDOR==vendor]) * 100
    )

    ax.plot(weekly["week"], weekly["pct"],
            color=color, lw=2.2,
            marker="o", markersize=5,
            label=f"{vendor}  YTD={ytd_clean_pct:.1f}%",
            zorder=4)
    ax.fill_between(weekly["week"], weekly["pct"],
                    alpha=0.08, color=color)

ax.set_xlabel("Week")
ax.set_ylabel("% of weekly mirrors essentially clean")
ax.set_title("Clean mirror rate over time",
             fontsize=11)
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=8)

# ── 6: Summary insight panel ──────────────────────────────
ax = axes[1][2]
ax.set_facecolor(BG_CARD)
ax.axis("off")

ax.text(0.5, 0.97, "Clean Mirror Insights",
        transform=ax.transAxes,
        fontsize=11, color=TEXT,
        fontweight="bold",
        va="top", ha="center")

ax.plot([0.05, 0.95], [0.91, 0.91],
        transform=ax.transAxes,
        color=BORDER, linewidth=0.8)

insights = [
    (VENDOR_A, "REO clean mirrors",
     "REO's cleanest mirrors — what\n"
     "process condition produced them?\n"
     "Are they concentrated in specific\n"
     "weeks or random throughout?"),

    (VENDOR_B, "GH clean mirrors",
     "GH has more clean mirrors overall.\n"
     "Are GH's best mirrors cleaner\n"
     "than REO's best mirrors?\n"
     "Sets the quality ceiling."),

    (MUTED, "Key question for ISC",
     "If REO can produce clean mirrors\n"
     "— what's different about those\n"
     "production runs vs the dirty ones?\n"
     "That difference is the root cause."),
]

y = 0.87
for color, title, body in insights:
    ax.text(0.04, y, title,
            transform=ax.transAxes,
            fontsize=9, color=color,
            fontweight="bold", va="top")
    y -= 0.06
    ax.text(0.04, y, body,
            transform=ax.transAxes,
            fontsize=8.5, color=TEXT,
            va="top", linespacing=1.6)
    y -= 0.25

fig.text(0.5, -0.02,
         f"Clean defined as coverage ≤ {threshold_7c:.3f}% "
         f"(bottom 25th percentile). "
         "Outer ring (R9) excluded.",
         ha="center", fontsize=8,
         color=MUTED, style="italic")

plt.tight_layout()
plt.show()
