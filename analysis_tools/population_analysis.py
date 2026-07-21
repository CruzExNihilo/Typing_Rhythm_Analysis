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
from scipy.stats import sem  # Standard Error of the Mean calculator
from scipy.stats import gaussian_kde
from statsmodels.api import nonparametric

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
        if resolved.exists():
            if resolved.is_file() and resolved.suffix.lower() == ".json":
                return resolved
            if resolved.is_dir():
                return resolved

    return None


def resolve_session_files(project_root: Path, explicit_paths):
    if explicit_paths:
        resolved = []
        for raw_path in explicit_paths:
            path = resolve_session_path(project_root, raw_path)
            if path is None:
                continue

            if path.is_file() and path.suffix.lower() == ".json":
                resolved.append(path)
            elif path.is_dir():
                resolved.extend(sorted(path.glob("*.json")))

        return sorted({path.resolve() for path in resolved})

    sessions_dir = project_root / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*.json"))


def extract_session_metrics(file_path: Path, target_timeline):
    """
    Extracts time-domain data, spectral profiles, dwell times, and 
    inter-keystroke intervals from an individual session file.
    """
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    events = data.get("events", [])

    input_events = [ev for ev in events if
                    ev.get("type") == "input" and ev.get("data", {}).get("inputType") == "insertText"]
    if len(input_events) < 10:
        return None

    # --- 1. Time Domain Processing ---
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
    cpm_raw = 60.0 / intervals_valid

    window = 5
    cpm_smoothed = np.convolve(cpm_raw, np.ones(window) / window, mode="same")
    cpm_smoothed = np.maximum(cpm_smoothed, 0)
    cpm_interp = np.interp(target_timeline, time_valid, cpm_smoothed, left=np.nan, right=np.nan)

    # --- 2. Spectral Analysis (FFT) ---
    signal = cpm_smoothed - np.mean(cpm_smoothed)
    sample_dt = np.median(np.diff(time_valid)) if len(time_valid) > 1 else 0.1
    target_freqs = np.linspace(0.05, 2.5, 100)

    if len(signal) < 2:
        return None

    fft_vals = np.fft.rfft(signal)
    raw_freqs = np.fft.rfftfreq(len(signal), d=sample_dt)
    raw_power = (np.abs(fft_vals) ** 2) / len(signal)
    power_interp = np.interp(target_freqs, raw_freqs[1:], raw_power[1:], left=0, right=0)

    # --- 3. Dwell Latency Tracking & Local Filtering ---
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
            dwell_times.append((t_ms - key_dict[code])) # Keep in ms
            dwell_timestamps.append(t_ms / 1000.0)
            del key_dict[code]

    dwell_times = np.array(dwell_times, dtype=float)
    dwell_timestamps = np.array(dwell_timestamps, dtype=float)
    dwell_mask = mask_outliers(dwell_times, min_value=20.0, max_value=5000.0)
    dwell_times = dwell_times[dwell_mask]
    dwell_timestamps = dwell_timestamps[dwell_mask]
    
    if len(dwell_times) > 5:
        lowess_fit = nonparametric.lowess(dwell_times, dwell_timestamps, frac=0.3)
        dwell_trend_interp = np.interp(target_timeline, lowess_fit[:, 0], lowess_fit[:, 1], left=np.nan, right=np.nan)
    else:
        dwell_trend_interp = np.full_like(target_timeline, np.nan)

    return {
        "cpm_profile": cpm_interp,
        "power_profile": power_interp,
        "dwell_trend": dwell_trend_interp,
        "raw_intervals_ms": intervals_valid * 1000.0
    }


def get_output_path(project_root: Path, stem: str) -> Path:
    result_dir = project_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return result_dir / f"{stem}_{timestamp}.pdf"


def main():
    parser = argparse.ArgumentParser(description="Aggregate typing sessions into a multi-panel performance profile.")
    parser.add_argument("session_files", nargs="*", help="Optional JSON session paths.")
    parser.add_argument("--output-name", help="Optional output filename without extension.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for the generated PDF.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    session_files = resolve_session_files(project_root, args.session_files)

    if not session_files:
        print("No session files located. Please add files to your sessions folder.")
        return

    print(f"Aggregating comprehensive metrics from {len(session_files)} files...")

    master_timeline = np.arange(0.0, 60.0, 0.1)
    target_freqs = np.linspace(0.05, 2.5, 100)

    all_cpm, all_power, all_dwell_trends, pooled_intervals = [], [], [], []

    for f in session_files:
        metrics = extract_session_metrics(f, master_timeline)
        if metrics is not None:
            all_cpm.append(metrics["cpm_profile"])
            all_power.append(metrics["power_profile"])
            all_dwell_trends.append(metrics["dwell_trend"])
            pooled_intervals.extend(metrics["raw_intervals_ms"])

    if not all_cpm or not all_power or not all_dwell_trends:
        print("No valid session metrics could be aggregated.")
        return

    # Convert matrices to structured numpy arrays for cross-session vector processing
    cpm_matrix = np.array(all_cpm)
    power_matrix = np.array(all_power)
    dwell_matrix = np.array(all_dwell_trends)
    pooled_intervals = np.array(pooled_intervals)

    # Calculate Population Parameters
    mean_cpm = np.nanmean(cpm_matrix, axis=0)
    sem_cpm = sem(cpm_matrix, axis=0, nan_policy='omit')
    std_cpm = np.nanstd(cpm_matrix, axis=0)

    mean_power = np.nanmean(power_matrix, axis=0)
    sem_power = sem(power_matrix, axis=0, nan_policy='omit')

    mean_dwell = np.nanmean(dwell_matrix, axis=0)
    sem_dwell = sem(dwell_matrix, axis=0, nan_policy='omit')

    # --- Rendering Upgraded Multi-Panel Graphics ---
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5))
    fig.patch.set_facecolor("white")
    lbl_opts = {"fontsize": 11, "fontweight": "bold", "va": "bottom", "ha": "right"}

    # (a) Aggregated Time Domain CPM
    ax = axes[0, 0]
    ax.plot(master_timeline, mean_cpm, color="#1f4e79", linewidth=1.2, label="Mean Velocity")
    ax.fill_between(master_timeline, mean_cpm - sem_cpm, mean_cpm + sem_cpm, color="#1f4e79", alpha=0.15, label="±1 SEM")
    ax.plot(master_timeline, mean_cpm + std_cpm, color="#b22222", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.plot(master_timeline, mean_cpm - std_cpm, color="#b22222", linestyle=":", linewidth=0.8, alpha=0.5, label="±1 SD")
    ax.text(-0.15, 1.03, "(a)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Population Velocity Profile", pad=8)
    ax.set_xlabel("Session Time (s)")
    ax.set_ylabel("Speed (CPM)")
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (b) Aggregated Power Spectral Density
    ax = axes[0, 1]
    ax.plot(target_freqs, mean_power, color="#b22222", linewidth=1.0, label="Mean Power")
    ax.fill_between(target_freqs, mean_power - sem_power, mean_power + sem_power, color="#b22222", alpha=0.15, label="±1 SEM")
    ax.text(-0.15, 1.03, "(b)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Population Spectral Density", pad=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (CPM²/Hz)")
    ax.set_xlim(0.05, 2.5)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)

    # (c) Aggregated Dwell Latency Tendency
    ax = axes[1, 0]
    ax.plot(master_timeline, mean_dwell, color="#d35400", linewidth=1.2, label="Mean Dwell Trend")
    ax.fill_between(master_timeline, mean_dwell - sem_dwell, mean_dwell + sem_dwell, color="#d35400", alpha=0.15, label="±1 SEM")
    ax.text(-0.15, 1.03, "(c)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Population Dwell Regressions", pad=8)
    ax.set_xlabel("Session Time (s)")
    ax.set_ylabel("Dwell Latency (ms)")
    ax.set_xlim(0, 60)

    valid_dwell = mean_dwell[np.isfinite(mean_dwell)]
    if valid_dwell.size:
        lower = max(0.0, float(np.nanmin(valid_dwell)) * 0.95)
        upper = float(np.nanmax(valid_dwell)) * 1.05
        if upper <= lower:
            upper = lower + 50.0
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0, 250)

    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (d) Pooled Inter-Keystroke Distribution (KDE Model)
    ax = axes[1, 1]
    clean_intervals = pooled_intervals[pooled_intervals < 600]
    if clean_intervals.size < 3:
        clean_intervals = pooled_intervals
    ax.hist(clean_intervals, bins=30, density=True, color="#5e35b1", alpha=0.4, edgecolor='white', linewidth=0.5, label="Empirical")
    kde = gaussian_kde(clean_intervals, bw_method="silverman")
    x_grid = np.linspace(clean_intervals.min(), clean_intervals.max(), 200)
    ax.plot(x_grid, kde(x_grid), color="#4a148c", linewidth=1.2, label="KDE Model")
    ax.text(-0.15, 1.03, "(d)", transform=ax.transAxes, **lbl_opts)
    ax.set_title("Pooled Inter-Arrival Density", pad=8)
    ax.set_xlabel("Inter-Keystroke Gap (ms)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(left=0, right=600)
    ax.legend(frameon=False, fontsize=8)

    for row in axes:
        for ax in row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)

    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=3.0)
    output_stem = args.output_name or "typing_speed_population_analysis"
    output_path = get_output_path(project_root, output_stem)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Complete population matrix rendering successful: {output_path.name}")


if __name__ == "__main__":
    main()