import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

IMAGE_BASE_PATH = "/mock/images/"   # ← replace with your path

VERDICT_THRESHOLDS = {
    "clean":       0.05,   # below → clean
    "review":      0.20,   # below → review
    # above → contaminated
}

CLASS_COLORS_MAP = {
    "fod":           "#2196F3",
    "polish defect": "#FF9800",
    "wax":           "#9C27B0",
    "scratch":       "#F44336",
    "water spot":    "#00BCD4",
    "no defect":     "#9E9E9E",
}


# ─────────────────────────────────────────────────────────────
# MOCK IMAGE GENERATOR
# ─────────────────────────────────────────────────────────────

def load_or_mock_image(fname, cx, cy, radius):
    """
    Try to load real image from IMAGE_BASE_PATH + fname.
    If not found — generate a mock dark-field mirror image.
    """
    path = os.path.join(IMAGE_BASE_PATH, fname)

    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mock — dark mirror with random bright specks
    size = 512
    mock = np.zeros((size, size, 3), dtype=np.uint8)
    rng  = np.random.default_rng(42)

    # Scale cx/cy/radius to mock size
    scale  = size / 2048
    mcx    = int(cx * scale)
    mcy    = int(cy * scale)
    mrad   = int(radius * scale)

    # Mirror disk
    cv2.circle(mock, (mcx, mcy), mrad, (18, 18, 20), -1)
    cv2.circle(mock, (mcx, mcy), mrad, (55, 55, 60),  3)

    # Random specks
    n_specks = rng.integers(10, 50)
    for _ in range(n_specks):
        r_norm = rng.uniform(0.1, 0.95)
        angle  = rng.uniform(0, 2 * np.pi)
        sx     = int(mcx + r_norm * mrad * np.cos(angle))
        sy     = int(mcy + r_norm * mrad * np.sin(angle))
        bright = int(rng.integers(180, 255))
        cv2.circle(mock, (sx, sy), 2,
                   (bright, bright, bright), -1)

    return mock


# ─────────────────────────────────────────────────────────────
# RADIAL HEATMAP HELPER
# ─────────────────────────────────────────────────────────────

def draw_radial_heatmap(ax, ring_values, worst_ring,
                         vendor_color):
    """
    Draw concentric ring mirror diagram colored by
    coverage value per ring. Single mirror version.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    ax.set_facecolor(BG_PANEL)
    ax.set_aspect("equal")

    cmap = cm.get_cmap("RdYlGn_r")
    vmax = max(ring_values) if max(ring_values) > 0 else 0.1
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    n = len(ring_values)
    for i in range(n - 1, -1, -1):
        r_outer = (i + 1) / n
        color   = cmap(norm(ring_values[i]))
        circle  = plt.Circle((0, 0), r_outer,
                              color=color, zorder=i + 1)
        ax.add_patch(circle)

        # Ring border
        border = plt.Circle((0, 0), r_outer,
                             fill=False,
                             edgecolor="white",
                             linewidth=0.5,
                             linestyle="--",
                             alpha=0.4,
                             zorder=i + 2)
        ax.add_patch(border)

        # Label
        r_mid     = ((i + 0.5) / n)
        cval      = norm(ring_values[i])
        txt_color = "white" if cval > 0.5 else "#333"
        ax.text(0, r_mid,
                f"R{i}\n{ring_values[i]:.3f}%",
                ha="center", va="center",
                fontsize=6, color=txt_color,
                fontweight="bold", zorder=20)

    # Worst ring highlight
    wr_outer = (worst_ring + 1) / n
    wr_inner = worst_ring / n
    theta    = np.linspace(0, 2 * np.pi, 100)
    ax.plot(wr_outer * np.cos(theta),
            wr_outer * np.sin(theta),
            color="white", lw=2.5,
            linestyle="-", zorder=30,
            alpha=0.9)

    # Mirror boundary
    boundary = plt.Circle((0, 0), 1.0,
                           fill=False,
                           edgecolor=vendor_color,
                           linewidth=2.0, zorder=25)
    ax.add_patch(boundary)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────

def get_verdict(coverage):
    if coverage <= VERDICT_THRESHOLDS["clean"]:
        return "✓  Clean", "#2ECC71"
    elif coverage <= VERDICT_THRESHOLDS["review"]:
        return "⚠  Review", "#F39C12"
    else:
        return "✕  Contaminated", "#E74C3C"


# ─────────────────────────────────────────────────────────────
# MAIN PLOT
# ─────────────────────────────────────────────────────────────

def plot_report_card(sn):
    sn = sn.strip()
    if not sn:
        print("Enter a mirror SN.")
        return

    # Lookup
    row = df[df["fname"].str.contains(sn, case=False, na=False)]
    if len(row) == 0:
        print(f"Mirror '{sn}' not found in dataset.")
        return

    row = row.iloc[0]

    vendor       = row["SUB_VENDOR"]
    vendor_color = VENDOR_A if vendor == "REO" else VENDOR_B
    fname        = row["fname"]
    date         = str(row["serial_dt"])[:10]
    pred_class   = row["predicted_defects"]
    probs        = row["predicted_probabilities"]
    confidence   = max(probs) if isinstance(probs, list) else 0.0
    coverage     = row["total_coverage_excl_last"]
    count        = row["defect_count_excl_last"]
    cx           = int(row["cx"])
    cy           = int(row["cy"])
    radius       = int(row["radius"])
    worst_ring   = int(row["worst_ring"]) if not np.isnan(
                       row["worst_ring"]) else 0
    ce_r         = row["nn_clark_evans_r"]
    nn_dist      = row["nn_mean_nnd"]

    count_small   = row.get("count_small_excl_last",   0)
    count_medium  = row.get("count_medium_excl_last",  0)
    count_large   = row.get("count_large_excl_last",   0)
    count_xlarge  = row.get("count_xlarge_excl_last",  0)
    count_xxlarge = row.get("count_xxlarge_excl_last", 0)

    ring_cols  = [f"coverage_ring_{i}" for i in range(9)]
    ring_vals  = [row[c] for c in ring_cols]

    verdict_text, verdict_color = get_verdict(coverage)

    # CE-R pattern
    if np.isnan(ce_r):
        pattern = "N/A"
    elif ce_r < 0.7:
        pattern = "clustered"
    elif ce_r <= 1.3:
        pattern = "random"
    else:
        pattern = "dispersed"

    # ── Layout ───────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.2, 1.0, 1.2],
        wspace=0.08,
        left=0.03, right=0.97,
        top=0.88,  bottom=0.05
    )

    fig.suptitle(
        f"Mirror Inspection Report  |  {fname}  "
        f"|  {vendor}  |  {date}",
        fontsize=12, fontweight="semibold",
        color=TEXT, y=0.97
    )

    # ── Left: raw image ───────────────────────────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_facecolor("#000000")

    img_rgb = load_or_mock_image(fname, cx, cy, radius)

    # Crop to mirror area with padding
    scale   = img_rgb.shape[0] / 2048
    mcx     = int(cx * scale)
    mcy     = int(cy * scale)
    mrad    = int(radius * scale)
    pad     = int(mrad * 0.12)
    y1      = max(0, mcy - mrad - pad)
    y2      = min(img_rgb.shape[0], mcy + mrad + pad)
    x1      = max(0, mcx - mrad - pad)
    x2      = min(img_rgb.shape[1], mcx + mrad + pad)
    cropped = img_rgb[y1:y2, x1:x2]

    ax_img.imshow(cropped)
    ax_img.axis("off")
    ax_img.set_title("Mirror Image",
                     fontsize=10, color=TEXT, pad=6)

    # ── Middle: radial heatmap ────────────────────────────
    ax_heat = fig.add_subplot(gs[1])
    draw_radial_heatmap(ax_heat, ring_vals,
                         worst_ring, vendor_color)
    ax_heat.set_title(
        f"Radial Coverage  (worst: R{worst_ring}={ring_vals[worst_ring]:.3f}%)",
        fontsize=10, color=TEXT, pad=6
    )

    # ── Right: metrics panel ──────────────────────────────
    ax_met = fig.add_subplot(gs[2])
    ax_met.set_facecolor(BG_CARD)
    ax_met.axis("off")

    # Class + confidence
    cls_color = CLASS_COLORS_MAP.get(pred_class, MUTED)
    ax_met.add_patch(plt.matplotlib.patches.FancyBboxPatch(
        (0.02, 0.84), 0.96, 0.13,
        boxstyle="round,pad=0.01",
        facecolor=cls_color,
        alpha=0.15,
        transform=ax_met.transAxes,
        clip_on=True, zorder=0
    ))
    ax_met.text(0.5, 0.93,
                pred_class.upper(),
                transform=ax_met.transAxes,
                fontsize=15, color=cls_color,
                fontweight="bold",
                va="center", ha="center")
    ax_met.text(0.5, 0.86,
                f"confidence: {confidence:.1%}",
                transform=ax_met.transAxes,
                fontsize=8.5, color=cls_color,
                va="center", ha="center")

    ax_met.plot([0.02, 0.98], [0.83, 0.83],
                transform=ax_met.transAxes,
                color=BORDER, linewidth=0.8)

    # Metrics
    metrics_rows = [
        ("Coverage",
         f"{coverage:.4f}%",
         "#E74C3C" if coverage > 0.05 else "#2ECC71"),
        ("Defect count",
         f"{int(count):,}",
         TEXT),
        ("Small  /  Medium",
         f"{int(count_small):,}  /  {int(count_medium):,}",
         TEXT),
        ("Large  /  XLarge",
         f"{int(count_large):,}  /  {int(count_xlarge):,}",
         TEXT),
        ("XXLarge",
         f"{int(count_xxlarge):,}",
         "#E74C3C" if count_xxlarge > 0 else TEXT),
        ("Clustering (CE-R)",
         f"{ce_r:.3f}  {pattern}" if not np.isnan(ce_r)
         else "N/A",
         TEXT),
        ("NN distance",
         f"{nn_dist:.0f} px" if not np.isnan(nn_dist)
         else "N/A",
         TEXT),
        ("Worst ring",
         f"R{worst_ring}",
         TEXT),
    ]

    row_h = 0.083
    y     = 0.80

    ax_met.text(0.04, y + 0.02,
                "METRICS",
                transform=ax_met.transAxes,
                fontsize=8, color=MUTED,
                fontweight="bold", va="top")

    ax_met.plot([0.02, 0.98], [y, y],
                transform=ax_met.transAxes,
                color=BORDER, linewidth=0.5)

    y -= 0.01
    for j, (label, value, val_color) in enumerate(metrics_rows):
        y -= row_h
        bg = BG_PANEL if j % 2 == 0 else BG_CARD
        ax_met.add_patch(
            plt.matplotlib.patches.FancyBboxPatch(
                (0.01, y - 0.02), 0.98, row_h,
                boxstyle="square,pad=0",
                facecolor=bg,
                transform=ax_met.transAxes,
                clip_on=True, zorder=0))

        ax_met.text(0.04, y + row_h/2 - 0.01,
                    label,
                    transform=ax_met.transAxes,
                    fontsize=8.5, color=MUTED,
                    va="center")
        ax_met.text(0.96, y + row_h/2 - 0.01,
                    value,
                    transform=ax_met.transAxes,
                    fontsize=8.5, color=val_color,
                    va="center", ha="right",
                    fontweight="bold")

    # Verdict
    y -= 0.06
    ax_met.plot([0.02, 0.98], [y + 0.04, y + 0.04],
                transform=ax_met.transAxes,
                color=BORDER, linewidth=0.8)

    ax_met.add_patch(
        plt.matplotlib.patches.FancyBboxPatch(
            (0.02, y - 0.06), 0.96, 0.09,
            boxstyle="round,pad=0.01",
            facecolor=verdict_color,
            alpha=0.15,
            transform=ax_met.transAxes,
            clip_on=True, zorder=0))

    ax_met.text(0.5, y - 0.015,
                verdict_text,
                transform=ax_met.transAxes,
                fontsize=13, color=verdict_color,
                fontweight="bold",
                va="center", ha="center")

    # Vendor badge
    ax_met.add_patch(
        plt.matplotlib.patches.FancyBboxPatch(
            (0.02, 0.01), 0.40, 0.055,
            boxstyle="round,pad=0.01",
            facecolor=vendor_color,
            alpha=0.15,
            transform=ax_met.transAxes,
            clip_on=True, zorder=0))
    ax_met.text(0.22, 0.038,
                vendor,
                transform=ax_met.transAxes,
                fontsize=9, color=vendor_color,
                fontweight="bold",
                va="center", ha="center")

    # Date badge
    ax_met.add_patch(
        plt.matplotlib.patches.FancyBboxPatch(
            (0.55, 0.01), 0.43, 0.055,
            boxstyle="round,pad=0.01",
            facecolor=BG_PANEL,
            transform=ax_met.transAxes,
            clip_on=True, zorder=0))
    ax_met.text(0.765, 0.038,
                date,
                transform=ax_met.transAxes,
                fontsize=9, color=MUTED,
                va="center", ha="center")

    plt.show()


# ─────────────────────────────────────────────────────────────
# WIDGET
# ─────────────────────────────────────────────────────────────

sn_input = widgets.Text(
    placeholder="Enter mirror SN or filename...",
    description="Mirror SN:",
    style={"description_width": "90px"},
    layout=widgets.Layout(width="350px")
)

search_btn = widgets.Button(
    description="Generate Report",
    button_style="primary",
    layout=widgets.Layout(width="150px")
)

status_label = widgets.Label(
    value=f"Dataset: {len(df):,} mirrors  |  "
          f"REO: {(df.SUB_VENDOR=='REO').sum():,}  |  "
          f"GH: {(df.SUB_VENDOR=='GH').sum():,}"
)

out = widgets.Output()

def on_search(b):
    out.clear_output()
    with out:
        plot_report_card(sn_input.value)

search_btn.on_click(on_search)

# Also trigger on Enter key
def on_submit(change):
    if change["new"].endswith("\n"):
        sn_input.value = sn_input.value.strip()
        out.clear_output()
        with out:
            plot_report_card(sn_input.value)

sn_input.observe(on_submit, names="value")

controls = widgets.HBox(
    [sn_input, search_btn],
    layout=widgets.Layout(
        gap="10px",
        align_items="center",
        padding="10px",
        border="1px solid #D1D5DB",
    )
)

display(status_label, controls, out)
