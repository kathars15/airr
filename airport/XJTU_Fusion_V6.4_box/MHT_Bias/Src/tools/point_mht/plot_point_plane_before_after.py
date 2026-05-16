# -*- coding: utf-8 -*-
"""Plot raw point records on radar-centered XY plane before and after filtering.

Edit the config block below directly. No CLI arguments.
"""

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================== Config ====================
POINT_FILE = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_121233\point_records.csv"
)
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
OUTPUT_PNG = OUTPUT_DIR / "point_plane_before_after.png"

AZIMUTH_MIN_DEG = 120.0
AZIMUTH_MAX_DEG = 135.0
TRUE_ONLY = True
# ===============================================


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def in_azimuth_window(value):
    if value is None:
        return False
    value = value % 360.0
    if AZIMUTH_MIN_DEG <= AZIMUTH_MAX_DEG:
        return AZIMUTH_MIN_DEG <= value <= AZIMUTH_MAX_DEG
    return value >= AZIMUTH_MIN_DEG or value <= AZIMUTH_MAX_DEG


def polar_to_xy(range_m, azimuth_deg):
    azimuth_rad = math.radians(azimuth_deg)
    x = range_m * math.sin(azimuth_rad)   # east
    y = range_m * math.cos(azimuth_rad)   # north
    return x, y


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rng = to_float(row.get("range"))
            az = to_float(row.get("azimuth"))
            if rng is None or az is None:
                continue
            rows.append(
                {
                    "range": rng,
                    "azimuth": az,
                    "is_true_point": str(row.get("is_true_point", "")).strip(),
                }
            )
    return rows


def filter_rows(rows):
    out = []
    for row in rows:
        if not in_azimuth_window(row["azimuth"]):
            continue
        if TRUE_ONLY and row["is_true_point"] != "1":
            continue
        out.append(row)
    return out


def split_xy(rows):
    xs, ys = [], []
    for row in rows:
        x, y = polar_to_xy(row["range"], row["azimuth"])
        xs.append(x)
        ys.append(y)
    return xs, ys


def set_equal_axes(ax, xs1, ys1, xs2, ys2):
    values = list(xs1) + list(ys1) + list(xs2) + list(ys2) + [0.0]
    vmax = max(abs(v) for v in values) if values else 1.0
    vmax = max(vmax, 10.0)
    ax.set_xlim(-vmax, vmax)
    ax.set_ylim(-vmax, vmax)
    ax.set_aspect("equal", adjustable="box")


def main():
    all_rows = load_rows(POINT_FILE)
    filtered_rows = filter_rows(all_rows)
    x_all, y_all = split_xy(all_rows)
    x_filtered, y_filtered = split_xy(filtered_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    axes[0].scatter(x_all, y_all, s=10, alpha=0.55, color="#1f77b4")
    axes[0].scatter([0], [0], s=60, color="#d62728", marker="x", label="radar center")
    axes[0].set_title("Before filtering")
    axes[0].set_xlabel("East X (m)")
    axes[0].set_ylabel("North Y (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if filtered_rows:
        time_index = list(range(len(filtered_rows)))
        scatter = axes[1].scatter(
            x_filtered,
            y_filtered,
            c=time_index,
            cmap="turbo",
            s=16,
            alpha=0.85,
        )
        cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Time order")
    else:
        axes[1].scatter(x_filtered, y_filtered, s=12, alpha=0.70, color="#2ca02c")
    axes[1].scatter([0], [0], s=60, color="#d62728", marker="x", label="radar center")
    axes[1].set_title(
        f"After filtering | az={AZIMUTH_MIN_DEG:.0f}-{AZIMUTH_MAX_DEG:.0f} deg"
        + (", true_only" if TRUE_ONLY else "")
    )
    axes[1].set_xlabel("East X (m)")
    axes[1].set_ylabel("North Y (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    set_equal_axes(axes[0], x_all, y_all, x_filtered, y_filtered)
    set_equal_axes(axes[1], x_all, y_all, x_filtered, y_filtered)

    fig.suptitle(
        f"Point records on radar-centered plane | total={len(all_rows)} filtered={len(filtered_rows)}",
        fontsize=12,
    )
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    print(f"[plot] input: {POINT_FILE}")
    print(f"[plot] total rows: {len(all_rows)}")
    print(f"[plot] filtered rows: {len(filtered_rows)}")
    print(f"[plot] output: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
