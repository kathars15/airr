from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "run_requirement_sim.py"
IMPACT_DIR = ROOT / "impact_outputs"

spec = importlib.util.spec_from_file_location("mht_requirement_sim", MAIN_SCRIPT)
sim = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)

BASELINE = {
    "sensor_count": 3,
    "sensor_pd": 0.7,
    "sensor_pos_std": 30.0,
    "sensor_fa_prob": 0.25,
    "mc_runs": 1,
}

num = 6
SWEEPS = {
    "sensor_count": {"min": 1, "max": 6, "num_points": num, "color": "#1f77b4", "as_int": True},
    "sensor_pd": {"min": 0.6, "max": 0.95, "num_points": num, "color": "#d62728", "as_int": False},
    "sensor_pos_std": {"min": 15, "max": 35, "num_points": num, "color": "#2ca02c", "as_int": False},
    "sensor_fa_prob": {"min": 0.10, "max": 0.35, "num_points": num, "color": "#9467bd", "as_int": False},
}

COMMON_PROGRESS = np.linspace(0.0, 1.0, 8)

METRIC_INFO = {
    "pd_out": ("Pd_out", "combined_pd_out.png", 0.90),
    "rmse": ("RMSE (m)", "combined_rmse.png", 15.0),
    "pfa_out": ("Pfa_out", "combined_pfa_out.png", 0.05),
}


def expand_range(min_value: float, max_value: float, num_points: int, as_int: bool) -> list[float]:
    if num_points <= 1 or abs(max_value - min_value) < 1e-12:
        return [int(round(min_value)) if as_int else float(min_value)]
    values = np.linspace(float(min_value), float(max_value), int(num_points))
    if as_int:
        return [int(round(value)) for value in values]
    return [float(value) for value in values]


def run_sweep(sweep_field: str, values: list[float], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[combined] start sweep {sweep_field}, values={values}")
    for idx, value in enumerate(values, start=1):
        print(f"[combined]   {sweep_field} {idx}/{len(values)} -> value={value}")
        config = sim.ScenarioConfig(
            sensor_count=int(BASELINE["sensor_count"]),
            sensor_pd=float(BASELINE["sensor_pd"]),
            sensor_pos_std_m=float(BASELINE["sensor_pos_std"]),
            sensor_fa_prob=float(BASELINE["sensor_fa_prob"]),
        )
        attr_name = {
            "sensor_count": "sensor_count",
            "sensor_pd": "sensor_pd",
            "sensor_pos_std": "sensor_pos_std_m",
            "sensor_fa_prob": "sensor_fa_prob",
        }[sweep_field]
        setattr(config, attr_name, int(value) if sweep_field == "sensor_count" else float(value))

        trial_metrics = []
        for run_idx in range(int(BASELINE["mc_runs"])):
            seed = (
                sim.RANDOM_SEED
                + idx * 100000
                + run_idx * 1009
                + config.sensor_count * 17
                + int(config.sensor_pd * 1000) * 19
                + int(config.sensor_pos_std_m * 10) * 23
                + int(config.sensor_fa_prob * 1000) * 29
            )
            metrics, _ = sim.run_single_trial(config, seed)
            trial_metrics.append(metrics)

        row = sim.summarize_trials(config, trial_metrics)
        print(
            f"[combined]     result: Pd_out={float(row['detection_rate']):.3f}, "
            f"RMSE={float(row['rmse_m']):.2f}m, Pfa_out={float(row['false_alarm_rate']):.3f}"
        )
        rows.append({
            "pd_out": float(row["detection_rate"]),
            "rmse": float(row["rmse_m"]),
            "pfa_out": float(row["false_alarm_rate"]),
        })
    return rows


def interpolate_to_common(values: list[float], target_len: int) -> list[float]:
    if len(values) == target_len:
        return list(values)
    if len(values) == 1:
        return [values[0]] * target_len
    src_x = np.linspace(0.0, 1.0, len(values))
    dst_x = np.linspace(0.0, 1.0, target_len)
    return list(np.interp(dst_x, src_x, values))


def plot_combined(series: dict, metric: str, ylabel: str, filename: str, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for name, cfg in series.items():
        y_values = interpolate_to_common(cfg[metric], len(COMMON_PROGRESS))
        value_range = cfg["value_range"]
        range_label = f"{value_range[0]}~{value_range[1]}"
        ax.plot(
            COMMON_PROGRESS,
            y_values,
            marker="o",
            linewidth=2.0,
            label=f"{name} ({range_label})",
            color=cfg["color"],
        )
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label="threshold")
    ax.set_title(f"{ylabel} under weak-input one-factor sweeps")
    ax.set_xlabel("normalized sweep progress")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = IMPACT_DIR / filename
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


def main() -> int:
    IMPACT_DIR.mkdir(parents=True, exist_ok=True)
    all_series = {}
    print("[combined] building combined impact plots...")
    for sweep_field, cfg in SWEEPS.items():
        values = expand_range(cfg["min"], cfg["max"], cfg["num_points"], cfg["as_int"])
        subdir = IMPACT_DIR / sweep_field.replace("sensor_pos_std", "sigma").replace("sensor_fa_prob", "pfa").replace("sensor_count", "count").replace("sensor_pd", "pd")
        rows = run_sweep(sweep_field, values, subdir)
        all_series[sweep_field] = {
            "pd_out": [row["pd_out"] for row in rows],
            "rmse": [row["rmse"] for row in rows],
            "pfa_out": [row["pfa_out"] for row in rows],
            "color": cfg["color"],
            "value_range": (values[0], values[-1]),
        }

    for metric, (ylabel, filename, threshold) in METRIC_INFO.items():
        plot_combined(all_series, metric, ylabel, filename, threshold)
    print("[combined] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
