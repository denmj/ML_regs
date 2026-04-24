import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────
# HELPER — get filtered df slice
# ─────────────────────────────────────────────────────────────

def get_slice(vendor, defect_class, size):
    subset = df.copy()

    if vendor != "Both":
        subset = subset[subset.SUB_VENDOR == vendor]

    if defect_class != "All":
        subset = subset[subset.predicted_defects == defect_class]

    if size != "All":
        size_col = f"count_{size.lower()}_excl_last"
        subset = subset[subset[size_col] > 0]

    return subset


# ─────────────────────────────────────────────────────────────
# HELPER — compute radar metrics for a subset
# ─────────────────────────────────────────────────────────────

RADAR_METRICS = {
    "Coverage\n(mean %)":    ("total_coverage_excl_last", "mean",   True),
    "Defect\nCount":         ("defect_count_excl_last",   "mean",   True),
    "Size\nSeverity":        ("_weighted_sum",             "mean",   True),
    "Clustering\n(1-CE_R)":  ("nn_clark_evans_r",         "mean",   True),
    "Worst\nZone":           ("worst_ring",                "median", True),
}
# True = higher raw value is worse

def compute_metrics(subset):
    result = {}
    for label, (col, agg, higher_worse) in RADAR_METRICS.items():
        if col not in subset.columns:
            result[label] = np.nan
            continue
        val = (subset[col].mean() if agg == "mean"
               else subset[col].median())
        result[label] = val
    return result


def normalize_metrics(metrics_reo, metrics_gh):
    """
    Normalize each metric 0→1 across REO and GH combined.
    Higher = worse for all metrics.
    """
    labels = list(RADAR_METRICS.keys())
    norm_reo, norm_gh = [], []

    for label in labels:
        r = metrics_reo.get(label, np.nan)
        g = metrics_gh.get(label, np.nan)

        lo = min(r, g) if not np.isnan(r) and not np.isnan(g) else 0
        hi = max(r, g) if not np.isnan(r) and not np.isnan(g) else 1

        if hi == lo:
            norm_reo.append(0.5)
            norm_gh.append(0.5)
        else:
            norm_reo.append((r - lo) / (hi - lo))
            norm_gh.append((g - lo) / (hi - lo))

    return norm_reo, norm_gh


# ─────────────────────────────────────────────────────────────
# PLOT FUNCTION
# ─────────────────────────────────────────────────────────────

def plot_radar(vendor, defect_class, size):
    labels = list(RADAR_METRICS.keys())
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n,
                          endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(14, 6), facecolor=BG)

    # ── Left: radar ───────────────────────────────────────
    ax_radar = fig.add_subplot(121, polar=True)
    ax_radar.set_facecolor(BG_PANEL)
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)

    reo_sub = get_slice("REO", defect_class, size)
    gh_sub  = get_slice("GH",  defect_class, size)

    m_reo = compute_metrics(reo_sub)
    m_gh  = compute_metrics(gh_sub)

    norm_reo, norm_gh = normalize_metrics(m_reo, m_gh)

    reo_vals = norm_reo + norm_reo[:1]
    gh_vals  = norm_gh  + norm_gh[:1]

    ax_radar.plot(angles, reo_vals, color=VENDOR_A,
                  lw=2.2, zorder=4)
    ax_radar.fill(angles, reo_vals, color=VENDOR_A,
                  alpha=0.15, zorder=3)

    ax_radar.plot(angles, gh_vals, color=VENDOR_B,
                  lw=2.2, zorder=4)
    ax_radar.fill(angles, gh_vals, color=VENDOR_B,
                  alpha=0.15, zorder=3)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=9, color=TEXT)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["25%", "50%", "75%", "100%"],
                              fontsize=7, color=MUTED)
    ax_radar.grid(color=GRID, linewidth=0.6,
                  linestyle="--", alpha=0.7)
    ax_radar.spines["polar"].set_color(BORDER)

    ax_radar.legend(
        handles=[
            plt.matplotlib.patches.Patch(
                color=VENDOR_A, label=f"REO  n={len(reo_sub):,}"),
            plt.matplotlib.patches.Patch(
                color=VENDOR_B, label=f"GH   n={len(gh_sub):,}"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=9,
        facecolor=BG_PANEL,
        edgecolor=BORDER,
    )

    title = (f"Vendor comparison — "
             f"Class: {defect_class}  |  "
             f"Size: {size}")
    ax_radar.set_title(title, fontsize=11,
                       fontweight="semibold",
                       color=TEXT, pad=20)

    # ── Right: raw values table ───────────────────────────
    ax_table = fig.add_subplot(122)
    ax_table.set_facecolor(BG_CARD)
    ax_table.axis("off")

    ax_table.text(0.02, 0.97, "Metric",
                  transform=ax_table.transAxes,
                  fontsize=9, color=MUTED, va="top")
    ax_table.text(0.55, 0.97, "REO",
                  transform=ax_table.transAxes,
                  fontsize=9, color=VENDOR_A,
                  va="top", ha="center")
    ax_table.text(0.80, 0.97, "GH",
                  transform=ax_table.transAxes,
                  fontsize=9, color=VENDOR_B,
                  va="top", ha="center")
    ax_table.text(0.97, 0.97, "REO worse?",
                  transform=ax_table.transAxes,
                  fontsize=9, color=MUTED,
                  va="top", ha="right")

    ax_table.plot([0.01, 0.99], [0.92, 0.92],
                  transform=ax_table.transAxes,
                  color=GRID, linewidth=0.8)

    row_h = 0.13
    metric_formats = {
        "Coverage\n(mean %)":   "{:.4f}%",
        "Defect\nCount":        "{:.1f}",
        "Size\nSeverity":       "{:.0f}",
        "Clustering\n(1-CE_R)": "{:.3f}",
        "Worst\nZone":          "R{:.1f}",
    }

    for j, label in enumerate(labels):
        y      = 0.88 - j * row_h
        bg_col = BG_PANEL if j % 2 == 0 else BG_CARD
        r_val  = m_reo.get(label, np.nan)
        g_val  = m_gh.get(label,  np.nan)
        fmt    = metric_formats.get(label, "{:.3f}")

        ax_table.add_patch(
            plt.matplotlib.patches.FancyBboxPatch(
                (0, y - 0.05), 1, row_h,
                boxstyle="square,pad=0",
                facecolor=bg_col,
                transform=ax_table.transAxes,
                clip_on=True, zorder=0))

        # Clean label for table
        clean_label = label.replace("\n", " ")
        ax_table.text(0.02, y, clean_label,
                      transform=ax_table.transAxes,
                      fontsize=8.5, color=TEXT, va="center")

        r_str = fmt.format(r_val) if not np.isnan(r_val) else "N/A"
        g_str = fmt.format(g_val) if not np.isnan(g_val) else "N/A"

        ax_table.text(0.55, y, r_str,
                      transform=ax_table.transAxes,
                      fontsize=8.5, color=VENDOR_A,
                      va="center", ha="center")
        ax_table.text(0.80, y, g_str,
                      transform=ax_table.transAxes,
                      fontsize=8.5, color=VENDOR_B,
                      va="center", ha="center")

        # REO worse indicator
        if not np.isnan(r_val) and not np.isnan(g_val):
            reo_worse = r_val > g_val
            indicator = "▲ Yes" if reo_worse else "▼ No"
            ind_color = VENDOR_A if reo_worse else VENDOR_B
            ax_table.text(0.97, y, indicator,
                          transform=ax_table.transAxes,
                          fontsize=8.5, color=ind_color,
                          va="center", ha="right",
                          fontweight="bold")

    ax_table.set_title("Raw metric values",
                       fontsize=11, color=TEXT,
                       fontweight="semibold", pad=10)

    fig.text(0.5, -0.02,
             "Radar normalized 0→1 per metric — larger area = worse. "
             "Raw values shown in table. "
             "Outer ring (R9) excluded.",
             ha="center", fontsize=8,
             color=MUTED, style="italic")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# WIDGETS
# ─────────────────────────────────────────────────────────────

vendor_dd = widgets.Dropdown(
    options=["Both", "REO", "GH"],
    value="Both",
    description="Vendor:",
    style={"description_width": "80px"},
    layout=widgets.Layout(width="200px")
)

class_dd = widgets.Dropdown(
    options=["All", "fod", "polish defect",
             "wax", "scratch", "water spot"],
    value="All",
    description="Class:",
    style={"description_width": "80px"},
    layout=widgets.Layout(width="200px")
)

size_dd = widgets.Dropdown(
    options=["All", "small", "medium",
             "large", "xlarge", "xxlarge"],
    value="All",
    description="Size:",
    style={"description_width": "80px"},
    layout=widgets.Layout(width="200px")
)

# Sample count label
sample_label = widgets.Label(value="")

def update_label(*args):
    reo_n = len(get_slice("REO", class_dd.value, size_dd.value))
    gh_n  = len(get_slice("GH",  class_dd.value, size_dd.value))
    sample_label.value = (f"REO: {reo_n:,} mirrors  |  "
                          f"GH: {gh_n:,} mirrors")

# Wire up
out = widgets.interactive_output(
    plot_radar,
    {"vendor":       vendor_dd,
     "defect_class": class_dd,
     "size":         size_dd}
)

vendor_dd.observe(update_label, names="value")
class_dd.observe(update_label,  names="value")
size_dd.observe(update_label,   names="value")
update_label()

controls = widgets.HBox(
    [vendor_dd, class_dd, size_dd, sample_label],
    layout=widgets.Layout(
        gap="20px",
        align_items="center",
        padding="10px",
        border="1px solid #D1D5DB",
        border_radius="8px"
    )
)

display(controls, out)
