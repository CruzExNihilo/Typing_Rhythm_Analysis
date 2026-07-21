from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np
import matplotlib

try:
    from analysis_tools.outlier_utils import mask_outliers
except ModuleNotFoundError:
    from outlier_utils import mask_outliers

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde  # Advanced density estimation
from statsmodels.api import nonparametric  # Advanced local regression (LOWESS)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
    }
)


def resolve_session_path(project_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend([
            Path.cwd() / candidate,
            project_root / candidate,
            project_root / "sessions" / candidate,
        ])

    for path in search_paths:
        resolved = path.resolve()
        if resolved.exists() and resolved.is_file() and resolved.suffix.lower() == ".json":
            return resolved

    return None


def resolve_session_files(project_root: Path, explicit_path: str | None):
    if explicit_path:
        path = resolve_session_path(project_root, explicit_path)
        if path is not None:
            return [path]

    sessions_dir = project_root / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*.json"))


def load_all_events(data_path: Path):
    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("events", [])


def optimize_lowess_frac(x, y):
    """
    Performs Leave-One-Out Cross-Validation to autonomously discover
    the mathematically optimal neighborhood fraction ('frac') for LOWESS.
    """
    fractions = np.linspace(0.15, 0.5, 8)
    best_frac = 0.3
    min_error = float("inf")

    # Step downsample indices for validation performance efficiency
    step = max(1, len(x) // 100)
    test_indices = np.arange(0, len(x), step)

    for f in fractions:
        total_sq_error = 0.0
        for idx in test_indices:
            # Mask out the validation target point
            train_mask = np.ones(len(x), dtype=bool)
            train_mask[idx] = False

            try:
                # Fit model on training neighborhood subset
                res = nonparametric.lowess(y[train_mask], x[train_mask], frac=f, it=0)
                # Predict value via local linear interpolation
                y_pred = np.interp(x[idx], res[:, 0], res[:, 1])
                total_sq_error += (y[idx] - y_pred) ** 2
            except:
                continue

        if total_sq_error < min_error and total_sq_error > 0:
            min_error = total_sq_error
            best_frac = f

    return best_frac


def build_advanced_metrics(events):
    input_events = [ev for ev in events if
                    ev.get("type") == "input" and ev.get("data", {}).get("inputType") == "insertText"]

    timestamps_ms = np.array([float(event.get("testMs", 0.0)) for event in input_events], dtype=float)
    timestamps_ms -= timestamps_ms[0]
    time_seconds = timestamps_ms / 1000.0

    intervals_sec = np.diff(time_seconds)
    intervals_sec = np.where(intervals_sec > 0, intervals_sec, np.nan)
    valid = np.isfinite(intervals_sec)

    time_valid = time_seconds[1:][valid]
    intervals_valid = intervals_sec[valid]
    clean_mask = mask_outliers(intervals_valid * 1000.0, min_value=20.0, max_value=2000.0)
    intervals_valid = intervals_valid[clean_mask]
    time_valid = time_valid[clean_mask]
    cpm_smoothed = np.convolve(60.0 / intervals_valid, np.ones(5) / 5, mode="same")
    cpm_smoothed = np.maximum(cpm_smoothed, 0)

    key_dict = {}
    dwell_times, dwell_timestamps = [], []

    for ev in events:
        ev_type, t_ms = ev.get("type"), float(ev.get("testMs", 0.0))
        code = ev.get("data", {}).get("code", "")
        if not code:
            continue
        if ev_type == "keydown":
            key_dict[code] = t_ms
        elif ev_type == "keyup" and code in key_dict:
            dwell_times.append((t_ms - key_dict[code]) / 1000.0)
            dwell_timestamps.append(t_ms / 1000.0)
            del key_dict[code]

    dwell_times = np.array(dwell_times, dtype=float)
    dwell_timestamps = np.array(dwell_timestamps, dtype=float)
    dwell_mask = mask_outliers(dwell_times * 1000.0, min_value=20.0, max_value=5000.0)
    dwell_times = dwell_times[dwell_mask]
    dwell_timestamps = dwell_timestamps[dwell_mask]

    return {
        "time_valid": time_valid,
        "cpm_smoothed": cpm_smoothed,
        "dwell_timestamps": np.array(dwell_timestamps),
        "dwell_times": np.array(dwell_times),
        "intervals_valid": intervals_valid
    }


def get_output_path(project_root: Path, stem: str) -> Path:
    result_dir = project_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return result_dir / f"{stem}_{timestamp}.pdf"


def plot_single_session_metrics(metrics, output_path: Path):
    time_valid, cpm_smoothed = metrics["time_valid"], metrics["cpm_smoothed"]
    dwell_timestamps, dwell_times = metrics["dwell_timestamps"], metrics["dwell_times"] * 1000.0
    intervals_valid = metrics["intervals_valid"] * 1000.0

    # FFT Calculations
    signal = cpm_smoothed - np.mean(cpm_smoothed)
    sample_dt = np.median(np.diff(time_valid)) if len(time_valid) > 1 else 0.01
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=sample_dt)
    power = (np.abs(fft_vals) ** 2) / len(signal)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5))
    fig.patch.set_facecolor("white")
    lbl_opts = {"fontsize": 11, "fontweight": "bold", "va": "bottom", "ha": "right"}

    # --- Subplot (a): Time Domain CPM ---
    ax = axes[0, 0]
    ax.plot(time_valid, cpm_smoothed, color="#1f4e79", linewidth=1.0, label="Raw Velocity")
    ax.axhline(np.mean(cpm_smoothed), color="#1f4e79", linestyle="--", linewidth=0.8, label="Mean Speed")
    ax.text(-0.15, 1.03, "(a)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Time-Domain Velocity Profile", pad=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Characters per minute (CPM)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)

    # --- Subplot (b): Power Spectral Density ---
    ax = axes[0, 1]
    ax.plot(freqs[1:], power[1:], color="#b22222", linewidth=0.9)
    ax.text(-0.15, 1.03, "(b)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Power Spectral Density Spectrum", pad=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (CPM²/Hz)")

    # --- Subplot (c): Dwell Time with Cross-Validated LOWESS ---
    ax = axes[1, 0]
    ax.scatter(dwell_timestamps, dwell_times, color="#2e7d32", s=4, alpha=0.3, edgecolors='none', label="Keypress Data")

    # Run the autonomous cross-validation loop to calculate the optimal fraction
    opt_frac = optimize_lowess_frac(dwell_timestamps, dwell_times)
    lowess_res = nonparametric.lowess(dwell_times, dwell_timestamps, frac=opt_frac)
    ax.plot(lowess_res[:, 0], lowess_res[:, 1], color="#d35400", linewidth=1.5, label=f"LOWESS (f={opt_frac:.2f})")

    ax.text(-0.15, 1.03, "(c)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Key Dwell Latency Regression", pad=8)
    ax.set_xlabel("Test Timeline (s)")
    ax.set_ylabel("Dwell Duration (ms)")

    valid_dwell = dwell_times[np.isfinite(dwell_times)]
    if valid_dwell.size:
        lower = max(0.0, float(np.nanmin(valid_dwell)) * 0.95)
        upper = float(np.nanmax(valid_dwell)) * 1.05
        if upper <= lower:
            upper = lower + 50.0
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0, 250)

    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # --- Subplot (d): Histogram with Auto-tuned KDE (Silverman's Rule) ---
    ax = axes[1, 1]
    clean_intervals = intervals_valid[intervals_valid < 600]
    if clean_intervals.size < 3:
        clean_intervals = intervals_valid
    ax.hist(clean_intervals, bins=25, density=True, color="#5e35b1", alpha=0.5, edgecolor='white', linewidth=0.5,
            label="Empirical")

    # Configure advanced KDE tracking utilizing automated silverman's tuning bandwidth
    kde = gaussian_kde(clean_intervals, bw_method="silverman")
    x_grid = np.linspace(clean_intervals.min(), clean_intervals.max(), 200)
    ax.plot(x_grid, kde(x_grid), color="#4a148c", linewidth=1.5, label="KDE Model")

    ax.text(-0.15, 1.03, "(d)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Inter-Keystroke Probability Density", pad=8)
    ax.set_xlabel("Key-to-Key Inter-Arrival Gap (ms)")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=False, fontsize=8)

    for row in axes:
        for ax in row:
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=3.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create a detailed single-session typing performance figure from a JSON session file.")
    parser.add_argument("session_file", nargs="?", help="Optional .json file to analyze. If omitted, the first file in sessions/ is used.")
    parser.add_argument("--output-name", help="Optional output filename without extension.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for the generated PDF.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    session_files = resolve_session_files(project_root, args.session_file)

    if not session_files:
        sessions_dir = project_root / "sessions"
        print(f"No .json files found inside '{sessions_dir}'.")
        print("Drop session JSON files into sessions/ or pass a specific file as an argument.")
        return

    data_path = session_files[0]
    events = load_all_events(data_path)
    metrics = build_advanced_metrics(events)
    output_stem = args.output_name or "typing_speed_analysis"
    output_path = get_output_path(project_root, output_stem)
    plot_single_session_metrics(metrics, output_path)
    print(f"Single-session analysis completed successfully as {output_path.name} using {data_path.name}")


if __name__ == "__main__":
    main()