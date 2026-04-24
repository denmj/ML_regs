import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────
# METRIC DEFINITIONS
# higher_worse: True = higher raw = worse
#               False = lower raw = worse
#               None = no direction
# ─────────────────────────────────────────────────────────────

RADAR_METRICS = {
    "Coverage\n(mean %)":    ("total_coverage_excl_last", "mean",   True),
    "Defect\nCount":         ("defect_count_excl_last",   "mean",   True),
    "Size\nSeverity":        ("_weighted_sum",             "mean",   True),
    "Clustering\n(CE-R)":    ("nn_clark_evans_r",         "mean",   False),
    "NN Distance\n(px)":     ("nn_mean_nnd",              "mean",   False),
    "Worst\nZone":           ("worst_ring",                "median", None),
}

METRIC_FORMATS = {
    "Coverage\n(mean %)":  "{:.4f}%",
    "Defect\nCount":       "{:.1f}",
    "Size\nSeverity":      "{:.0f}",
    "Clustering\n(CE-R)":  "{:.3f}",
    "NN Distance\n(px)":   "{:.1f}px",
    "Worst\nZone":         "R{:.1f}",
}

METRIC_NOTES = {
    "Coverage\n(mean %)":  "higher = more surface affected",
    "Defect\nCount":       "higher = more defects",
    "Size\nSeverity":      "higher = larger defects",
    "Clustering\n(CE-R)":  "lower = more clustered",
    "NN Distance\n(px)":   "lower = defects closer together",
    "Worst\nZone":         "no direction — center or edge both bad",
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_slice(vendor, defect_class, size):
    subset = df.copy()
    if vendor != "Both":
        subset = subset[subset.SUB_VENDOR == vendor]
    if defect_class != "All":
        subset = subset[subset.predicted_defects == defect_class]
    if size != "All":
        size_col = f"count_{size.lower()}_excl_last"
        subset   = subset[subset[size_col] > 0]
    return subset


def compute_metrics(subset):
    result = {}
    for label, (col, agg, _) in RADAR_METRICS.items():
        if col not in subset.columns or len(subset) == 0:
            result[label] = np.nan
            continue
        result[label] = (subset[col].mean() if agg == "mean"
                         else subset[col].median())
    return result


def normalize_metrics(metrics_reo, metrics_gh):
    labels   = list(RADAR_METRICS.keys())
    norm_reo = []
    norm_gh  = []

    for label in labels:
        _, _, direction = RADAR_METRICS[label]
        r = metrics_reo.get(label, np.nan)
        g = metrics_gh.get(label,  np.nan)

        if np.isnan(r) or np.isnan(g):
            norm_reo.append(0.5)
            norm_gh.append(0.5)
            continue

        lo = min(r, g)
        hi = max(r, g)

        if hi == lo:
            norm_reo.append(0.5)
            norm_gh.append(0.5)
            continue

        r_norm = (r - lo) / (hi - lo)
        g_norm = (g - lo) / (hi - lo)

        # Invert so that larger radar area = worse
        if direction is False:
            r_norm = 1 - r_norm
            g_norm = 1 - g_norm

        norm_reo.append(r_norm)
        norm_gh.append(g_norm)

    return norm_reo, norm_gh


# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────

def plot_radar(vendor, defect_class, size):
    labels = list(RADAR_METRICS.keys())
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    reo_sub = get_slice("REO", defect_class, size)
    gh_sub  = get_slice("GH",  defect_class, size)
    m_reo   = compute_metrics(reo_sub)
    m_gh    = compute_metrics(gh_sub)

    norm_reo, norm_gh = normalize_metrics(m_reo, m_gh)

    reo_vals = norm_reo + norm_reo[:1]
    gh_vals  = norm_gh  + norm_gh[:1]

    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    fig.suptitle(
        f"Vendor Comparison  |  Class: {defect_class}  "
        f"|  Size: {size}",
        fontsize=12, fontweight="semibold",
        color=TEXT, y=1.01
    )

    # ── Left: radar ───────────────────────────────────────
    ax_radar = fig.add_subplot(131, polar=True)
    ax_radar.set_facecolor(BG_PANEL)
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)

    ax_radar.plot(angles, reo_vals,
                  color=VENDOR_A, lw=2.2, zorder=4)
    ax_radar.fill(angles, reo_vals,
                  color=VENDOR_A, alpha=0.15, zorder=3)

    ax_radar.plot(angles, gh_vals,
                  color=VENDOR_B, lw=2.2, zorder=4)
    ax_radar.fill(angles, gh_vals,
                  color=VENDOR_B, alpha=0.15, zorder=3)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=8, color=TEXT)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(
        ["25%", "50%", "75%", "100%"],
        fontsize=6, color=MUTED)
    ax_radar.grid(color=GRID, linewidth=0.6,
                  linestyle="--", alpha=0.7)
    ax_radar.spines["polar"].set_color(BORDER)

    # Class label in center
    dominant_class = (defect_class if defect_class != "All"
                      else "ALL")
    ax_radar.text(0, 0, dominant_class.upper(),
                  ha="center", va="center",
                  fontsize=10, fontweight="bold",
                  color=TEXT, zorder=10)

    ax_radar.legend(
        handles=[
            plt.matplotlib.patches.Patch(
                color=VENDOR_A,
                label=f"REO  n={len(reo_sub):,}"),
            plt.matplotlib.patches.Patch(
                color=VENDOR_B,
                label=f"GH   n={len(gh_sub):,}"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.4, 1.15),
        fontsize=8,
        facecolor=BG_PANEL,
        edgecolor=BORDER,
    )

    # ── Middle: REO metrics ───────────────────────────────
    ax_reo = fig.add_subplot(132)
    ax_reo.set_facecolor(BG_CARD)
    ax_reo.axis("off")

    ax_reo.text(0.5, 0.97, "REO",
                transform=ax_reo.transAxes,
                fontsize=12, color=VENDOR_A,
                fontweight="bold",
                va="top", ha="center")
    ax_reo.text(0.5, 0.90,
                f"n = {len(reo_sub):,} mirrors",
                transform=ax_reo.transAxes,
                fontsize=8, color=MUTED,
                va="top", ha="center")

    ax_reo.plot([0.05, 0.95], [0.86, 0.86],
                transform=ax_reo.transAxes,
                color=BORDER, linewidth=0.8)

    row_h = 0.12
    for j, label in enumerate(labels):
        y      = 0.82 - j * row_h
        bg_col = BG_PANEL if j % 2 == 0 else BG_CARD
        r_val  = m_reo.get(label, np.nan)
        fmt    = METRIC_FORMATS.get(label, "{:.3f}")
        note   = METRIC_NOTES.get(label, "")

        ax_reo.add_patch(
            plt.matplotlib.patches.FancyBboxPatch(
                (0, y - 0.04), 1, row_h,
                boxstyle="square,pad=0",
                facecolor=bg_col,
                transform=ax_reo.transAxes,
                clip_on=True, zorder=0))

        clean_label = label.replace("\n", " ")
        ax_reo.text(0.04, y, clean_label,
                    transform=ax_reo.transAxes,
                    fontsize=8, color=TEXT,
                    va="center")

        r_str = fmt.format(r_val) if not np.isnan(r_val) else "N/A"
        ax_reo.text(0.96, y, r_str,
                    transform=ax_reo.transAxes,
                    fontsize=8.5, color=VENDOR_A,
                    va="center", ha="right",
                    fontweight="bold")

    ax_reo.set_title("REO metrics",
                     fontsize=10, color=VENDOR_A,
                     fontweight="semibold", pad=8)

    # ── Right: GH metrics ─────────────────────────────────
    ax_gh = fig.add_subplot(133)
    ax_gh.set_facecolor(BG_CARD)
    ax_gh.axis("off")

    ax_gh.text(0.5, 0.97, "GH",
               transform=ax_gh.transAxes,
               fontsize=12, color=VENDOR_B,
               fontweight="bold",
               va="top", ha="center")
    ax_gh.text(0.5, 0.90,
               f"n = {len(gh_sub):,} mirrors",
               transform=ax_gh.transAxes,
               fontsize=8, color=MUTED,
               va="top", ha="center")

    ax_gh.plot([0.05, 0.95], [0.86, 0.86],
               transform=ax_gh.transAxes,
               color=BORDER, linewidth=0.8)

    for j, label in enumerate(labels):
        y      = 0.82 - j * row_h
        bg_col = BG_PANEL if j % 2 == 0 else BG_CARD
        g_val  = m_gh.get(label, np.nan)
        r_val  = m_reo.get(label, np.nan)
        fmt    = METRIC_FORMATS.get(label, "{:.3f}")
        _, _, direction = RADAR_METRICS[label]

        ax_gh.add_patch(
            plt.matplotlib.patches.FancyBboxPatch(
                (0, y - 0.04), 1, row_h,
                boxstyle="square,pad=0",
                facecolor=bg_col,
                transform=ax_gh.transAxes,
                clip_on=True, zorder=0))

        clean_label = label.replace("\n", " ")
        ax_gh.text(0.04, y, clean_label,
                   transform=ax_gh.transAxes,
                   fontsize=8, color=TEXT,
                   va="center")

        g_str = fmt.format(g_val) if not np.isnan(g_val) else "N/A"
        ax_gh.text(0.96, y, g_str,
                   transform=ax_gh.transAxes,
                   fontsize=8.5, color=VENDOR_B,
                   va="center", ha="right",
                   fontweight="bold")

    ax_gh.set_title("GH metrics",
                    fontsize=10, color=VENDOR_B,
                    fontweight="semibold", pad=8)

    fig.text(0.5, -0.03,
             "Radar: larger area = worse performance. "
             "Normalized per metric across both vendors. "
             "Outer ring (R9) excluded.",
             ha="center", fontsize=7.5,
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
    layout=widgets.Layout(width="180px")
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
    layout=widgets.Layout(width="180px")
)

sample_label = widgets.Label(value="")

def update_label(*args):
    reo_n = len(get_slice("REO", class_dd.value, size_dd.value))
    gh_n  = len(get_slice("GH",  class_dd.value, size_dd.value))
    sample_label.value = (f"REO: {reo_n:,} mirrors  |  "
                          f"GH: {gh_n:,} mirrors")

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
        gap="15px",
        align_items="center",
        padding="10px",
        border="1px solid #D1D5DB",
    )
)

display(controls, out)
