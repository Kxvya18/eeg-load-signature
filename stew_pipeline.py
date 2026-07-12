import argparse
import json
import os
import platform
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, iirnotch, sosfiltfilt
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def json_ready_array(x):
    """Return JSON-ready nested lists for numpy arrays or already-list values."""
    return x.tolist() if hasattr(x, "tolist") else x

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import mne
from asrpy import ASR
from ripser import ripser as compute_rips
from stew_topology_extensions import (
    compute_alpha_correlation,
    compute_envelope_correlation,
    compute_phase_locking_value,
    compute_circular_coordinate_summary,
    fc_similarity_to_distance,
    iaaft_surrogate,
    phase_randomized_surrogate,
    summarize_h1_representations,
    summarize_temporal_series,
)

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

DATA_DIR = PROJECT_ROOT / "STEW Dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "stew_tda_results_v3"

FS = 128
ADC_OFFSET = 4200.0
ADC_SCALE = 0.51

ALL_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

CONFIRMATORY_FRONTAL_CHANNELS = ["AF3", "F3", "FC5", "FC6", "F4", "AF4"]
REGION_CHANNELS = OrderedDict([
    ("confirmatory_frontal_roi", CONFIRMATORY_FRONTAL_CHANNELS),
    ("frontal", ["AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4"]),
    ("temporal", ["T7", "T8"]),
    ("posterior", ["P7", "O1", "O2", "P8"]),
    ("whole_head", ALL_CHANNELS),
])

FC_REGION_CHANNELS = OrderedDict([
    ("confirmatory_frontal_roi", CONFIRMATORY_FRONTAL_CHANNELS),
    ("frontal", ["AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4"]),
    ("posterior", ["P7", "O1", "O2", "P8"]),
    ("whole_head", ALL_CHANNELS),
])

ARTEFACT_ABS_THRESHOLD_UV = 100.0
HIGHPASS_HZ = 1.0
NOTCH_HZ = 50.0
ALPHA_LOW_HZ = 8.0
ALPHA_HIGH_HZ = 13.0

WINDOW_SECONDS = 4.0
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
SUBSAMPLE_STEP = 2

DEFAULT_TAU = 3
DEFAULT_D = 3
AMI_MAX_LAG = 50
AMI_N_BINS = 32
FNN_MAX_DIM = 8
FNN_RTOL = 15.0

POINT_CLOUD_NORMALIZATION = "unit_diameter"

LANDSCAPE_LAYERS = 3
LANDSCAPE_GRID_SIZE = 64
PERSISTENCE_IMAGE_SHAPE = (20, 20)
PERSISTENCE_IMAGE_SIGMA = 0.04

SURROGATE_N = 50
SURROGATE_IAAFT_ITER = 50
SURROGATE_METHODS = ("phase_randomized", "iaaft")
SURROGATE_COMPARE_FEATURES = (
    "max_persistence_H1",
    "landscape_total_area_H1",
    "persistence_image_mass_H1",
)

FC_CONNECTIVITY_METRIC = "alpha_envelope_correlation"
FC_CONNECTIVITY_METRICS = ("alpha_envelope_correlation", "alpha_plv", "alpha_correlation")


def design_butter(cutoff_hz, fs, btype, order=4):
    nyq = fs / 2.0
    return butter(order, cutoff_hz / nyq, btype=btype, output="sos")


def highpass_filter(x, cutoff_hz=HIGHPASS_HZ, fs=FS):
    sos = design_butter(cutoff_hz, fs, btype="highpass", order=4)
    return np.column_stack([sosfiltfilt(sos, x[:, ch]) for ch in range(x.shape[1])])


def bandpass_filter(x, low_hz, high_hz, fs=FS):
    nyq = fs / 2.0
    sos = butter(4, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return np.column_stack([sosfiltfilt(sos, x[:, ch]) for ch in range(x.shape[1])])


def notch_filter(x, notch_hz=NOTCH_HZ, fs=FS, q=30.0):
    w0 = notch_hz / (fs / 2.0)
    b, a = iirnotch(w0, q)
    return np.column_stack([filtfilt(b, a, x[:, ch]) for ch in range(x.shape[1])])


def load_subject(subject_id, data_dir=DATA_DIR):
    sid = f"sub{subject_id:02d}"
    lo_path = data_dir / f"{sid}_lo.txt"
    hi_path = data_dir / f"{sid}_hi.txt"
    raw_lo = np.loadtxt(lo_path)
    raw_hi = np.loadtxt(hi_path)
    if raw_lo.shape != (19200, 14) or raw_hi.shape != (19200, 14):
        raise ValueError(f"Unexpected shapes for {sid}: {raw_lo.shape}, {raw_hi.shape}")
    return raw_lo, raw_hi


def convert_to_uv(raw):
    return (raw - ADC_OFFSET) * ADC_SCALE


def average_reference(x):
    return x - np.mean(x, axis=1, keepdims=True)


def make_raw(data_uv):
    info = mne.create_info(ch_names=ALL_CHANNELS, sfreq=FS, ch_types=["eeg"] * len(ALL_CHANNELS))
    return mne.io.RawArray(data_uv.T, info, verbose="ERROR")


def asr_clean_pair(low_uv, high_uv):
    low_raw = make_raw(low_uv)
    high_raw = make_raw(high_uv)

    asr = ASR(sfreq=FS, cutoff=20)
    fit_exc = None
    clean_calibration = None
    sample_mask = None
    trim_used = 0

    # ASR can fail on a small minority of recordings due to internal window reshaping.
    # We retry with slightly trimmed calibration segments instead of discarding the subject.
    for trim_used in [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]:
        cal_uv = low_uv[:-trim_used] if trim_used > 0 else low_uv
        if len(cal_uv) < FS * 30:
            break
        cal_raw = make_raw(cal_uv)
        try:
            clean_calibration, sample_mask = asr.fit(cal_raw, return_clean_window=True)
            fit_exc = None
            break
        except Exception as exc:
            fit_exc = exc

    if fit_exc is not None:
        raise fit_exc

    low_clean = asr.transform(low_raw).get_data().T
    high_clean = asr.transform(high_raw).get_data().T

    calibration_clean_fraction_pct = float(np.mean(sample_mask) * 100.0)
    calibration_samples = int(sample_mask.sum())
    return low_clean, high_clean, {
        "status": "asr_applied" if trim_used == 0 else "asr_applied_trimmed_calibration",
        "calibration_clean_fraction_pct": calibration_clean_fraction_pct,
        "calibration_clean_samples": calibration_samples,
        "calibration_total_samples": int(sample_mask.size),
        "calibration_clean_minutes": float(calibration_samples / FS / 60.0),
        "clean_window_shape": list(clean_calibration.shape),
        "calibration_trimmed_samples": int(trim_used),
    }


def residual_artefact_metrics(x, threshold_uv=ARTEFACT_ABS_THRESHOLD_UV):
    abs_x = np.abs(x)
    return {
        "pct_samples_abs_gt_100uv": float(np.mean(abs_x > threshold_uv) * 100.0),
        "max_abs_uv": float(np.max(abs_x)),
        "rms_uv": float(np.sqrt(np.mean(np.square(x)))),
    }


def compute_ami(x, max_lag=AMI_MAX_LAG, n_bins=AMI_N_BINS):
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.zeros(max_lag)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    ami_vals = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        x1 = np.digitize(x[:-lag], bins) - 1
        x2 = np.digitize(x[lag:], bins) - 1
        x1 = np.clip(x1, 0, n_bins - 1)
        x2 = np.clip(x2, 0, n_bins - 1)
        ami_vals[lag - 1] = mutual_info_score(x1, x2)
    return ami_vals


def find_first_minimum(arr):
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            return i + 1
    return int(np.argmin(arr)) + 1


def estimate_tau(alpha_signal, max_lag=AMI_MAX_LAG):
    valid = alpha_signal[~np.isnan(alpha_signal)]
    if len(valid) < 4 * max_lag:
        return DEFAULT_TAU, np.zeros(max_lag)
    ami_curve = compute_ami(valid, max_lag=max_lag)
    tau = int(np.clip(find_first_minimum(ami_curve), 1, 20))
    return tau, ami_curve


def false_nearest_neighbours(x, tau, max_dim=FNN_MAX_DIM, rtol=FNN_RTOL):
    fnn_fracs = []
    n_total = len(x)

    for d in range(1, max_dim + 1):
        n_pts = n_total - d * tau
        if n_pts < 30:
            break

        X_d = np.column_stack([x[i * tau: i * tau + n_pts] for i in range(d)])
        n_pts_next = n_total - (d + 1) * tau
        if n_pts_next < 20:
            break

        X_d1 = np.column_stack([x[i * tau: i * tau + n_pts_next] for i in range(d + 1)])
        n_common = min(n_pts, n_pts_next)
        X_d = X_d[:n_common]
        X_d1 = X_d1[:n_common]

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X_d)
        distances, indices = nbrs.kneighbors(X_d)
        nn_dist = distances[:, 1]
        nn_idx = indices[:, 1]

        valid = nn_dist > 1e-10
        extra_dist = np.abs(X_d1[:, d] - X_d1[nn_idx, d])
        ratio = np.where(valid, extra_dist / (nn_dist + 1e-10), 0.0)
        fnn_frac = float(np.mean(ratio > rtol))
        fnn_fracs.append(fnn_frac)

        if fnn_frac < 0.05:
            fnn_fracs += [0.0] * (max_dim - d)
            break

    fnn_fracs = np.array(fnn_fracs)
    below = np.where(fnn_fracs < 0.10)[0]
    if len(below) > 0:
        optimal_d = int(below[0]) + 1
    else:
        optimal_d = int(np.argmin(fnn_fracs)) + 1

    optimal_d = int(np.clip(optimal_d, 2, 5))
    return fnn_fracs, optimal_d


def estimate_embedding_params(alpha_channel):
    signal = alpha_channel[::SUBSAMPLE_STEP]
    valid = signal[~np.isnan(signal)]
    if len(valid) < 200:
        return DEFAULT_TAU, DEFAULT_D, np.zeros(AMI_MAX_LAG), np.zeros(FNN_MAX_DIM)

    tau, ami_curve = estimate_tau(valid)
    fnn_fracs, optimal_d = false_nearest_neighbours(valid, tau=tau)
    return tau, optimal_d, ami_curve, fnn_fracs


def normalize_point_cloud(cloud, mode=POINT_CLOUD_NORMALIZATION):
    if mode == "raw":
        return cloud, 1.0
    if mode != "unit_diameter":
        raise ValueError(f"Unknown normalization mode: {mode}")

    diffs = cloud[:, None, :] - cloud[None, :, :]
    diam = float(np.max(np.linalg.norm(diffs, axis=-1)))
    if diam <= 0:
        return cloud.copy(), 1.0
    return cloud / diam, diam


def build_point_cloud_windows(alpha_channel, tau, d, window_samples=WINDOW_SAMPLES, subsample=SUBSAMPLE_STEP):
    signal_sub = alpha_channel[::subsample]
    eff_win = window_samples // subsample
    n_windows = len(signal_sub) // eff_win

    windows = []
    for w in range(n_windows):
        start_sub = w * eff_win
        end_sub = start_sub + eff_win
        win_data = signal_sub[start_sub:end_sub]
        start_sample = start_sub * subsample

        if np.any(np.isnan(win_data)):
            windows.append({
                "valid": False,
                "window_index": w,
                "window_start_sample": start_sample,
                "window_time_sec": float(start_sample / FS),
                "reason": "nan_in_window",
            })
            continue

        n_pts = eff_win - (d - 1) * tau
        if n_pts < 20:
            windows.append({
                "valid": False,
                "window_index": w,
                "window_start_sample": start_sample,
                "window_time_sec": float(start_sample / FS),
                "reason": "too_few_points",
            })
            continue

        cloud = np.column_stack([win_data[i * tau: i * tau + n_pts] for i in range(d)])
        windows.append({
            "valid": True,
            "window_index": w,
            "window_start_sample": start_sample,
            "window_time_sec": float(start_sample / FS),
            "signal_sub": win_data,
            "cloud": cloud,
        })
    return windows


def point_cloud_from_signal_sub(signal_sub, tau, d):
    signal_sub = np.asarray(signal_sub, dtype=float)
    n_pts = len(signal_sub) - (d - 1) * tau
    if n_pts < 20:
        return None
    return np.column_stack([signal_sub[i * tau: i * tau + n_pts] for i in range(d)])


def compute_persistence_features(cloud, distance_matrix=False):
    result = compute_rips(cloud, maxdim=1, distance_matrix=distance_matrix)
    dgms = result["dgms"]
    h0_diag = dgms[0]
    h1_diag = dgms[1]

    finite_h1 = h1_diag[np.isfinite(h1_diag[:, 1])]
    if len(finite_h1) == 0:
        lifetimes_h1 = np.array([0.0])
        max_persistence_h1 = 0.0
        total_persistence_h1 = 0.0
        entropy_h1 = 0.0
        n_significant_cycles = 0
    else:
        lifetimes_h1 = finite_h1[:, 1] - finite_h1[:, 0]
        max_persistence_h1 = float(lifetimes_h1.max())
        total_persistence_h1 = float(lifetimes_h1.sum())
        normed = lifetimes_h1 / (lifetimes_h1.sum() + 1e-10)
        entropy_h1 = float(-np.sum(normed * np.log(normed + 1e-10)))
        threshold = float(np.median(lifetimes_h1) + 2.0 * np.std(lifetimes_h1)) if len(lifetimes_h1) > 2 else 0.0
        n_significant_cycles = int(np.sum(lifetimes_h1 > threshold))

    finite_h0 = h0_diag[np.isfinite(h0_diag[:, 1])]
    if len(finite_h0) == 0:
        total_persistence_h0 = 0.0
        n_components_h0 = 1
    else:
        lifetimes_h0 = finite_h0[:, 1] - finite_h0[:, 0]
        total_persistence_h0 = float(lifetimes_h0.sum())
        half_max_filt = float(h0_diag[:, 0].max() / 2.0) if len(h0_diag) > 0 else 0.0
        n_components_h0 = int(np.sum((h0_diag[:, 0] <= half_max_filt) & (h0_diag[:, 1] > half_max_filt)))
        n_components_h0 = max(n_components_h0, 1)

    return {
        "max_persistence_H1": max_persistence_h1,
        "total_persistence_H1": total_persistence_h1,
        "persistence_entropy_H1": entropy_h1,
        "n_significant_cycles": n_significant_cycles,
        "total_persistence_H0": total_persistence_h0,
        "n_components_H0": n_components_h0,
        "mean_lifetime_H1": float(lifetimes_h1.mean()),
        "std_lifetime_H1": float(lifetimes_h1.std()),
        "n_bars_H1": int(len(finite_h1)),
        "diagrams": {"H0": h0_diag.tolist(), "H1": h1_diag.tolist()},
    }


def compute_surrogate_feature_summary(
    signal_sub,
    tau,
    d,
    normalization_mode,
    real_feature_map,
    surrogate_n=SURROGATE_N,
    iaaft_iterations=SURROGATE_IAAFT_ITER,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(42)

    method_dispatch = {
        "phase_randomized": lambda sig: phase_randomized_surrogate(sig, rng),
        "iaaft": lambda sig: iaaft_surrogate(sig, rng, n_iter=iaaft_iterations),
    }
    collected = {
        method: {feature: [] for feature in SURROGATE_COMPARE_FEATURES}
        for method in SURROGATE_METHODS
    }

    for method in SURROGATE_METHODS:
        generator = method_dispatch[method]
        for _ in range(surrogate_n):
            surrogate_signal = generator(signal_sub)
            cloud = point_cloud_from_signal_sub(surrogate_signal, tau=tau, d=d)
            if cloud is None:
                continue
            cloud_norm, _ = normalize_point_cloud(cloud, mode=normalization_mode)
            topology = compute_persistence_features(cloud_norm)
            representations = summarize_h1_representations(
                topology["diagrams"]["H1"],
                normalization_mode=normalization_mode,
                landscape_layers=LANDSCAPE_LAYERS,
                landscape_grid=LANDSCAPE_GRID_SIZE,
                image_shape=PERSISTENCE_IMAGE_SHAPE,
                image_sigma=PERSISTENCE_IMAGE_SIGMA,
            )
            surrogate_feature_map = {
                "max_persistence_H1": topology["max_persistence_H1"],
                "landscape_total_area_H1": representations["summary"]["landscape_total_area_H1"],
                "persistence_image_mass_H1": representations["summary"]["persistence_image_mass_H1"],
            }
            for feature_name in SURROGATE_COMPARE_FEATURES:
                collected[method][feature_name].append(float(surrogate_feature_map[feature_name]))

    summary = {}
    for method, feature_map in collected.items():
        for feature_name, values in feature_map.items():
            arr = np.asarray(values, dtype=float)
            if len(arr) == 0:
                mu = np.nan
                sigma = np.nan
                delta = np.nan
                zscore = np.nan
            else:
                mu = float(np.mean(arr))
                sigma = float(np.std(arr))
                delta = float(real_feature_map[feature_name] - mu)
                zscore = float(delta / (sigma + 1e-8))

            prefix = f"{method}_surrogate"
            summary[f"{prefix}_mean_{feature_name}"] = mu
            summary[f"{prefix}_std_{feature_name}"] = sigma
            summary[f"{prefix}_delta_{feature_name}"] = delta
            summary[f"{prefix}_z_{feature_name}"] = zscore

    summary["surrogate_n_per_method"] = surrogate_n
    return summary


def build_multichannel_windows(multichannel_data, window_samples=WINDOW_SAMPLES):
    x = np.asarray(multichannel_data, dtype=float)
    n_windows = x.shape[0] // window_samples
    windows = []
    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        win_data = x[start:end]
        if np.any(np.isnan(win_data)):
            windows.append({
                "valid": False,
                "window_index": w,
                "window_start_sample": start,
                "window_time_sec": float(start / FS),
                "reason": "nan_in_window",
            })
            continue
        windows.append({
            "valid": True,
            "window_index": w,
            "window_start_sample": start,
            "window_time_sec": float(start / FS),
            "window_data": win_data,
        })
    return windows


def compute_fc_similarity(window_data, metric=FC_CONNECTIVITY_METRIC):
    if metric == "alpha_envelope_correlation":
        return compute_envelope_correlation(window_data)
    if metric == "alpha_plv":
        return compute_phase_locking_value(window_data)
    if metric == "alpha_correlation":
        return compute_alpha_correlation(window_data)
    raise ValueError(f"Unknown FC connectivity metric: {metric}")


def analyze_fc_region(alpha_multichannel, region_name, channel_names, window_samples=WINDOW_SAMPLES, connectivity_metric=FC_CONNECTIVITY_METRIC):
    windows = build_multichannel_windows(alpha_multichannel, window_samples=window_samples)
    per_window = []
    valid_windows = []

    for entry in windows:
        if not entry["valid"]:
            per_window.append(None)
            continue

        similarity = compute_fc_similarity(entry["window_data"], metric=connectivity_metric)
        distance = fc_similarity_to_distance(similarity)
        topology = compute_persistence_features(distance, distance_matrix=True)
        representations = summarize_h1_representations(
            topology["diagrams"]["H1"],
            normalization_mode="unit_diameter",
            landscape_layers=LANDSCAPE_LAYERS,
            landscape_grid=LANDSCAPE_GRID_SIZE,
            image_shape=PERSISTENCE_IMAGE_SHAPE,
            image_sigma=PERSISTENCE_IMAGE_SIGMA,
        )
        offdiag = similarity[np.triu_indices_from(similarity, k=1)]
        feats = {
            "window_index": entry["window_index"],
            "window_start_sample": entry["window_start_sample"],
            "window_time_sec": entry["window_time_sec"],
            "fc_connectivity_metric": connectivity_metric,
            "similarity_matrix": json_ready_array(similarity),
            "diagrams": {
                "H0": json_ready_array(topology["diagrams"]["H0"]),
                "H1": json_ready_array(topology["diagrams"]["H1"]),
            },
            "fc_mean_similarity": float(np.mean(offdiag)) if len(offdiag) > 0 else np.nan,
            "fc_std_similarity": float(np.std(offdiag)) if len(offdiag) > 0 else np.nan,
            **{f"fc_{k}": v for k, v in topology.items() if k != "diagrams"},
            **{f"fc_{k}": v for k, v in representations["summary"].items()},
        }
        per_window.append(feats)
        valid_windows.append((entry, feats, similarity, topology["diagrams"], representations))

    if not valid_windows:
        return {
            "region_name": region_name,
            "channels": channel_names,
            "aggregated": {
                "median_fc_max_persistence_H1": np.nan,
                "mean_fc_max_persistence_H1": np.nan,
                "volatility_fc_max_persistence_H1": np.nan,
                "median_fc_landscape_total_area_H1": np.nan,
                "mean_fc_landscape_total_area_H1": np.nan,
                "volatility_fc_landscape_total_area_H1": np.nan,
                "median_fc_persistence_image_mass_H1": np.nan,
                "mean_fc_persistence_image_mass_H1": np.nan,
                "volatility_fc_persistence_image_mass_H1": np.nan,
                "mean_fc_mean_similarity": np.nan,
                "fc_connectivity_metric": connectivity_metric,
                "n_valid_windows": 0,
                "n_total_windows": len(windows),
                "pct_valid": 0.0,
            },
            "per_window": per_window,
            "representative_window": None,
        }

    fc_max = np.array([f["fc_max_persistence_H1"] for _, f, _, _, _ in valid_windows])
    fc_landscape = np.array([f["fc_landscape_total_area_H1"] for _, f, _, _, _ in valid_windows])
    fc_image = np.array([f["fc_persistence_image_mass_H1"] for _, f, _, _, _ in valid_windows])
    fc_similarity = np.array([f["fc_mean_similarity"] for _, f, _, _, _ in valid_windows])

    rep_idx = int(np.argmin(np.abs(fc_landscape - np.median(fc_landscape))))
    rep_entry, rep_feats, rep_similarity, rep_diags, rep_repr = valid_windows[rep_idx]

    aggregated = {}
    aggregated.update(summarize_temporal_series(fc_max, "fc_max_persistence_H1"))
    aggregated.update(summarize_temporal_series(fc_landscape, "fc_landscape_total_area_H1"))
    aggregated.update(summarize_temporal_series(fc_image, "fc_persistence_image_mass_H1"))
    aggregated.update(summarize_temporal_series(fc_similarity, "fc_mean_similarity"))
    aggregated["fc_connectivity_metric"] = connectivity_metric
    aggregated["n_valid_windows"] = int(len(valid_windows))
    aggregated["n_total_windows"] = int(len(windows))
    aggregated["pct_valid"] = float(len(valid_windows) / len(windows) * 100.0)

    representative_window = {
        "window_index": rep_entry["window_index"],
        "window_time_sec": rep_entry["window_time_sec"],
        "features": rep_feats,
        "similarity_matrix": json_ready_array(rep_similarity),
        "diagrams": rep_diags,
        "landscape_grid": json_ready_array(rep_repr["landscape"]["grid"]),
        "landscape_values": json_ready_array(rep_repr["landscape"]["landscape"]),
        "persistence_image_birth_grid": json_ready_array(rep_repr["persistence_image"]["birth_grid"]),
        "persistence_image_persistence_grid": json_ready_array(rep_repr["persistence_image"]["persistence_grid"]),
        "persistence_image_values": json_ready_array(rep_repr["persistence_image"]["image"]),
    }
    return {
        "region_name": region_name,
        "channels": channel_names,
        "aggregated": aggregated,
        "per_window": per_window,
        "representative_window": representative_window,
    }


def analyze_channel(
    alpha_channel,
    tau,
    d,
    normalization_mode=POINT_CLOUD_NORMALIZATION,
    surrogate_n=SURROGATE_N,
    iaaft_iterations=SURROGATE_IAAFT_ITER,
    window_samples=WINDOW_SAMPLES,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(42)
    windows = build_point_cloud_windows(alpha_channel, tau=tau, d=d, window_samples=window_samples)
    per_window = []
    valid_windows = []

    for entry in windows:
        if not entry["valid"]:
            per_window.append(None)
            continue

        cloud_norm, diameter = normalize_point_cloud(entry["cloud"], mode=normalization_mode)
        topology = compute_persistence_features(cloud_norm)
        representations = summarize_h1_representations(
            topology["diagrams"]["H1"],
            normalization_mode=normalization_mode,
            landscape_layers=LANDSCAPE_LAYERS,
            landscape_grid=LANDSCAPE_GRID_SIZE,
            image_shape=PERSISTENCE_IMAGE_SHAPE,
            image_sigma=PERSISTENCE_IMAGE_SIGMA,
        )
        alpha_power = float(np.mean(np.square(entry["signal_sub"])))
        alpha_rms = float(np.sqrt(alpha_power))
        alpha_envelope_mean = float(np.mean(np.abs(hilbert(entry["signal_sub"]))))
        real_feature_map = {
            "max_persistence_H1": topology["max_persistence_H1"],
            "landscape_total_area_H1": representations["summary"]["landscape_total_area_H1"],
            "persistence_image_mass_H1": representations["summary"]["persistence_image_mass_H1"],
        }
        surrogate_summary = compute_surrogate_feature_summary(
            entry["signal_sub"],
            tau=tau,
            d=d,
            normalization_mode=normalization_mode,
            real_feature_map=real_feature_map,
            surrogate_n=surrogate_n,
            iaaft_iterations=iaaft_iterations,
            rng=rng,
        )

        feats = {
            "window_index": entry["window_index"],
            "window_start_sample": entry["window_start_sample"],
            "window_time_sec": entry["window_time_sec"],
            "cloud_diameter": diameter,
            "alpha_power": alpha_power,
            "alpha_rms": alpha_rms,
            "alpha_envelope_mean": alpha_envelope_mean,
            **{k: v for k, v in topology.items() if k != "diagrams"},
            **representations["summary"],
            **surrogate_summary,
        }
        per_window.append(feats)
        valid_windows.append((entry, feats, topology["diagrams"], cloud_norm, representations))

    if not valid_windows:
        return {
            "aggregated": {
                "mean_max_persistence_H1": np.nan,
                "std_max_persistence_H1": np.nan,
                "median_max_persistence_H1": np.nan,
                "mean_total_persistence_H1": np.nan,
                "std_total_persistence_H1": np.nan,
                "mean_entropy_H1": np.nan,
                "std_entropy_H1": np.nan,
                "mean_n_significant_cycles": np.nan,
                "mean_total_persistence_H0": np.nan,
                "mean_alpha_power": np.nan,
                "median_alpha_power": np.nan,
                "mean_alpha_rms": np.nan,
                "mean_alpha_envelope": np.nan,
                "mean_cloud_diameter": np.nan,
                "mean_landscape_layer1_area_H1": np.nan,
                "mean_landscape_layer2_area_H1": np.nan,
                "mean_landscape_layer3_area_H1": np.nan,
                "mean_landscape_total_area_H1": np.nan,
                "std_landscape_total_area_H1": np.nan,
                "median_landscape_total_area_H1": np.nan,
                "mean_landscape_peak_H1": np.nan,
                "mean_landscape_l2_norm_H1": np.nan,
                "mean_persistence_image_mass_H1": np.nan,
                "std_persistence_image_mass_H1": np.nan,
                "median_persistence_image_mass_H1": np.nan,
                "mean_persistence_image_peak_H1": np.nan,
                "mean_persistence_image_entropy_H1": np.nan,
                "mean_persistence_image_l2_norm_H1": np.nan,
                "n_valid_windows": 0,
                "n_total_windows": len(windows),
                "pct_valid": 0.0,
                "normalization_mode": normalization_mode,
                "surrogate_n_per_method": surrogate_n,
            },
            "per_window": per_window,
            "representative_window": None,
        }

    mxp = np.array([f["max_persistence_H1"] for _, f, _, _, _ in valid_windows])
    tp = np.array([f["total_persistence_H1"] for _, f, _, _, _ in valid_windows])
    ent = np.array([f["persistence_entropy_H1"] for _, f, _, _, _ in valid_windows])
    nsc = np.array([f["n_significant_cycles"] for _, f, _, _, _ in valid_windows])
    h0p = np.array([f["total_persistence_H0"] for _, f, _, _, _ in valid_windows])
    alpha_power = np.array([f["alpha_power"] for _, f, _, _, _ in valid_windows])
    alpha_rms = np.array([f["alpha_rms"] for _, f, _, _, _ in valid_windows])
    alpha_env = np.array([f["alpha_envelope_mean"] for _, f, _, _, _ in valid_windows])
    cloud_diam = np.array([f["cloud_diameter"] for _, f, _, _, _ in valid_windows])
    landscape_layer1 = np.array([f["landscape_layer1_area_H1"] for _, f, _, _, _ in valid_windows])
    landscape_layer2 = np.array([f["landscape_layer2_area_H1"] for _, f, _, _, _ in valid_windows])
    landscape_layer3 = np.array([f["landscape_layer3_area_H1"] for _, f, _, _, _ in valid_windows])
    landscape_total = np.array([f["landscape_total_area_H1"] for _, f, _, _, _ in valid_windows])
    landscape_peak = np.array([f["landscape_peak_H1"] for _, f, _, _, _ in valid_windows])
    landscape_l2 = np.array([f["landscape_l2_norm_H1"] for _, f, _, _, _ in valid_windows])
    image_mass = np.array([f["persistence_image_mass_H1"] for _, f, _, _, _ in valid_windows])
    image_peak = np.array([f["persistence_image_peak_H1"] for _, f, _, _, _ in valid_windows])
    image_entropy = np.array([f["persistence_image_entropy_H1"] for _, f, _, _, _ in valid_windows])
    image_l2 = np.array([f["persistence_image_l2_norm_H1"] for _, f, _, _, _ in valid_windows])

    rep_idx = int(np.argmin(np.abs(mxp - np.median(mxp))))
    rep_entry, rep_feats, rep_diags, rep_cloud, rep_representations = valid_windows[rep_idx]

    aggregated = {
        "mean_max_persistence_H1": float(np.mean(mxp)),
        "std_max_persistence_H1": float(np.std(mxp)),
        "median_max_persistence_H1": float(np.median(mxp)),
        "mean_total_persistence_H1": float(np.mean(tp)),
        "std_total_persistence_H1": float(np.std(tp)),
        "mean_entropy_H1": float(np.mean(ent)),
        "std_entropy_H1": float(np.std(ent)),
        "mean_n_significant_cycles": float(np.mean(nsc)),
        "mean_total_persistence_H0": float(np.mean(h0p)),
        "mean_alpha_power": float(np.mean(alpha_power)),
        "median_alpha_power": float(np.median(alpha_power)),
        "mean_alpha_rms": float(np.mean(alpha_rms)),
        "mean_alpha_envelope": float(np.mean(alpha_env)),
        "mean_cloud_diameter": float(np.mean(cloud_diam)),
        "mean_landscape_layer1_area_H1": float(np.mean(landscape_layer1)),
        "mean_landscape_layer2_area_H1": float(np.mean(landscape_layer2)),
        "mean_landscape_layer3_area_H1": float(np.mean(landscape_layer3)),
        "mean_landscape_total_area_H1": float(np.mean(landscape_total)),
        "std_landscape_total_area_H1": float(np.std(landscape_total)),
        "median_landscape_total_area_H1": float(np.median(landscape_total)),
        "mean_landscape_peak_H1": float(np.mean(landscape_peak)),
        "mean_landscape_l2_norm_H1": float(np.mean(landscape_l2)),
        "mean_persistence_image_mass_H1": float(np.mean(image_mass)),
        "std_persistence_image_mass_H1": float(np.std(image_mass)),
        "median_persistence_image_mass_H1": float(np.median(image_mass)),
        "mean_persistence_image_peak_H1": float(np.mean(image_peak)),
        "mean_persistence_image_entropy_H1": float(np.mean(image_entropy)),
        "mean_persistence_image_l2_norm_H1": float(np.mean(image_l2)),
        "n_valid_windows": int(len(valid_windows)),
        "n_total_windows": int(len(windows)),
        "pct_valid": float(len(valid_windows) / len(windows) * 100.0),
        "normalization_mode": normalization_mode,
        "surrogate_n_per_method": int(surrogate_n),
    }
    aggregated.update(summarize_temporal_series(mxp, "max_persistence_H1"))
    aggregated.update(summarize_temporal_series(landscape_total, "landscape_total_area_H1"))
    aggregated.update(summarize_temporal_series(image_mass, "persistence_image_mass_H1"))
    aggregated.update(summarize_temporal_series(alpha_power, "alpha_power"))

    for method in SURROGATE_METHODS:
        for feature_name in SURROGATE_COMPARE_FEATURES:
            for stat_name in ["mean", "std", "delta", "z"]:
                key = f"{method}_surrogate_{stat_name}_{feature_name}"
                values = np.array([f[key] for _, f, _, _, _ in valid_windows], dtype=float)
                aggregated[f"mean_{key}"] = float(np.nanmean(values))
                aggregated[f"std_{key}"] = float(np.nanstd(values))
                aggregated[f"median_{key}"] = float(np.nanmedian(values))
                aggregated.update(summarize_temporal_series(values, key))

    circular_summary = compute_circular_coordinate_summary(rep_cloud)
    if circular_summary is not None:
        for key, value in circular_summary["summary"].items():
            aggregated[f"representative_{key}"] = value

    representative_window = {
        "window_index": rep_entry["window_index"],
        "window_time_sec": rep_entry["window_time_sec"],
        "features": rep_feats,
        "diagrams": rep_diags,
        "point_cloud": json_ready_array(rep_cloud),
        "landscape_grid": json_ready_array(rep_representations["landscape"]["grid"]),
        "landscape_values": json_ready_array(rep_representations["landscape"]["landscape"]),
        "persistence_image_birth_grid": json_ready_array(rep_representations["persistence_image"]["birth_grid"]),
        "persistence_image_persistence_grid": json_ready_array(rep_representations["persistence_image"]["persistence_grid"]),
        "persistence_image_values": json_ready_array(rep_representations["persistence_image"]["image"]),
    }
    if circular_summary is not None:
        representative_window["circular_coordinates"] = json_ready_array(circular_summary["theta"])
        representative_window["circular_coordinates_unwrapped"] = json_ready_array(circular_summary["unwrapped_theta"])
        representative_window["circular_summary"] = circular_summary["summary"]

    return {
        "aggregated": aggregated,
        "per_window": per_window,
        "representative_window": representative_window,
    }


def preprocess_subject_pair(subject_id):
    raw_lo, raw_hi = load_subject(subject_id)
    uv_lo = convert_to_uv(raw_lo)
    uv_hi = convert_to_uv(raw_hi)

    pre_metrics = {
        "lo_pre": residual_artefact_metrics(uv_lo),
        "hi_pre": residual_artefact_metrics(uv_hi),
    }

    lo_hp = highpass_filter(uv_lo)
    hi_hp = highpass_filter(uv_hi)
    lo_filt = notch_filter(lo_hp)
    hi_filt = notch_filter(hi_hp)

    lo_asr, hi_asr, asr_info = asr_clean_pair(lo_filt, hi_filt)
    lo_ref = average_reference(lo_asr)
    hi_ref = average_reference(hi_asr)

    post_metrics = {
        "lo_post": residual_artefact_metrics(lo_ref),
        "hi_post": residual_artefact_metrics(hi_ref),
    }

    alpha_lo = bandpass_filter(lo_ref, ALPHA_LOW_HZ, ALPHA_HIGH_HZ)
    alpha_hi = bandpass_filter(hi_ref, ALPHA_LOW_HZ, ALPHA_HIGH_HZ)

    return {
        "lo": alpha_lo,
        "hi": alpha_hi,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "asr_info": asr_info,
    }


def run_subject_pair(
    subject_id,
    normalization_mode=POINT_CLOUD_NORMALIZATION,
    surrogate_n=SURROGATE_N,
    iaaft_iterations=SURROGATE_IAAFT_ITER,
    window_seconds=WINDOW_SECONDS,
    fc_connectivity_metric=FC_CONNECTIVITY_METRIC,
):
    preprocessed = preprocess_subject_pair(subject_id)
    base_rng = np.random.default_rng(1000 + int(subject_id))
    window_samples = int(FS * float(window_seconds))

    results = {}
    for condition, alpha_all in [("lo", preprocessed["lo"]), ("hi", preprocessed["hi"])]:
        tau_per_channel = {}
        d_per_channel = {}
        ami_curves = {}
        fnn_curves = {}
        channel_results = {}

        for ch_idx, ch_name in enumerate(ALL_CHANNELS):
            tau_ch, d_ch, ami_ch, fnn_ch = estimate_embedding_params(alpha_all[:, ch_idx])
            tau_per_channel[ch_name] = int(tau_ch)
            d_per_channel[ch_name] = int(d_ch)
            ami_curves[ch_name] = json_ready_array(ami_ch)
            fnn_curves[ch_name] = json_ready_array(fnn_ch)

        tau_global = int(np.median(list(tau_per_channel.values())))
        d_global = int(np.median(list(d_per_channel.values())))

        for ch_idx, ch_name in enumerate(ALL_CHANNELS):
            channel_results[ch_name] = {
                **analyze_channel(
                    alpha_all[:, ch_idx],
                    tau=tau_per_channel[ch_name],
                    d=d_per_channel[ch_name],
                    normalization_mode=normalization_mode,
                    surrogate_n=surrogate_n,
                    iaaft_iterations=iaaft_iterations,
                    window_samples=window_samples,
                    rng=np.random.default_rng(base_rng.integers(0, 2**32 - 1)),
                ),
                "tau_used": tau_per_channel[ch_name],
                "d_used": d_per_channel[ch_name],
                "ami_curve": ami_curves[ch_name],
                "fnn_curve": fnn_curves[ch_name],
            }

        fc_region_results = {}
        for region_name, region_channels in FC_REGION_CHANNELS.items():
            idxs = [ALL_CHANNELS.index(ch) for ch in region_channels]
            fc_region_results[region_name] = analyze_fc_region(
                alpha_all[:, idxs],
                region_name,
                region_channels,
                window_samples=window_samples,
                connectivity_metric=fc_connectivity_metric,
            )

        cond_key = f"{condition}_post"
        cond_result = {
            "subject_id": subject_id,
            "condition": condition,
            "condition_label": "rest" if condition == "lo" else "task",
            "preprocessing_reference": "Lim et al. 2018 dataset; Safari et al. 2024 preprocessing sequence",
            "topology_extensions_reference": {
                "persistence_landscapes": "Bubenik 2015",
                "persistence_images": "Adams et al. 2017",
                "surrogate_testing": "Theiler et al. 1992",
                "iaaft_surrogates": "Schreiber and Schmitz 1996",
                "fc_topology": "Giusti et al. 2015; Yan et al. 2023",
                "circular_coordinates": "de Silva et al. 2011",
            },
            "preprocessing_sequence": [
                "ADC_to_microvolts",
                "highpass_1Hz",
                "line_noise_notch_50Hz",
                "ASR_on_subject_low_condition",
                "average_reference",
                "alpha_band_8_13Hz",
            ],
            "normalization_mode": normalization_mode,
            "landscape_layers": LANDSCAPE_LAYERS,
            "landscape_grid_size": LANDSCAPE_GRID_SIZE,
            "persistence_image_shape": list(PERSISTENCE_IMAGE_SHAPE),
            "persistence_image_sigma": PERSISTENCE_IMAGE_SIGMA,
            "surrogate_methods": list(SURROGATE_METHODS),
            "surrogate_n_per_method": surrogate_n,
            "surrogate_iaaft_iterations": iaaft_iterations,
            "fc_connectivity_metric": fc_connectivity_metric,
            "window_seconds": float(window_seconds),
            "window_samples": int(window_samples),
            "tau_global_median": tau_global,
            "d_global_median": d_global,
            "tau_per_channel": tau_per_channel,
            "d_per_channel": d_per_channel,
            "qc": {
                "pre_metrics": preprocessed["pre_metrics"][f"{condition}_pre"],
                "post_metrics": preprocessed["post_metrics"][cond_key],
                "asr": preprocessed["asr_info"],
            },
            "channels": channel_results,
            "regions": REGION_CHANNELS,
            "fc_regions": fc_region_results,
        }
        results[condition] = cond_result

    return results


def flatten_subject_condition(result, rating):
    row = {
        "subject_id": result["subject_id"],
        "condition": result["condition"],
        "condition_binary": 0 if result["condition"] == "lo" else 1,
        "subjective_rating": float(rating) if rating is not None else np.nan,
        "tau_global_median": result["tau_global_median"],
        "d_global_median": result["d_global_median"],
        "normalization_mode": result["normalization_mode"],
        "preprocessing_reference": result["preprocessing_reference"],
        "pre_pct_samples_abs_gt_100uv": result["qc"]["pre_metrics"]["pct_samples_abs_gt_100uv"],
        "post_pct_samples_abs_gt_100uv": result["qc"]["post_metrics"]["pct_samples_abs_gt_100uv"],
        "post_max_abs_uv": result["qc"]["post_metrics"]["max_abs_uv"],
        "post_rms_uv": result["qc"]["post_metrics"]["rms_uv"],
        "asr_calibration_clean_fraction_pct": result["qc"]["asr"]["calibration_clean_fraction_pct"],
    }

    for ch in ALL_CHANNELS:
        agg = result["channels"][ch]["aggregated"]
        row[f"tau_{ch}"] = result["tau_per_channel"][ch]
        row[f"d_{ch}"] = result["d_per_channel"][ch]
        for feat_name, feat_val in agg.items():
            if isinstance(feat_val, str):
                continue
            row[f"{ch}_{feat_name}"] = feat_val

    for region_name, region_result in result.get("fc_regions", {}).items():
        agg = region_result["aggregated"]
        for feat_name, feat_val in agg.items():
            if isinstance(feat_val, str):
                continue
            row[f"FCREGION_{region_name}_{feat_name}"] = feat_val

    lateralization_pairs = [("AF", "AF3", "AF4"), ("F", "F3", "F4"), ("FC", "FC5", "FC6")]
    lateralization_features = [
        "median_max_persistence_H1",
        "median_landscape_total_area_H1",
        "median_persistence_image_mass_H1",
        "median_alpha_power",
    ]
    for label, left_ch, right_ch in lateralization_pairs:
        for feature_name in lateralization_features:
            left_key = f"{left_ch}_{feature_name}"
            right_key = f"{right_ch}_{feature_name}"
            if left_key in row and right_key in row:
                row[f"LAT_{label}_{feature_name}"] = row[right_key] - row[left_key]
    return row


def run_full_pipeline(
    output_dir=DEFAULT_OUTPUT_DIR,
    subject_ids=None,
    normalization_mode=POINT_CLOUD_NORMALIZATION,
    surrogate_n=SURROGATE_N,
    iaaft_iterations=SURROGATE_IAAFT_ITER,
    window_seconds=WINDOW_SECONDS,
    fc_connectivity_metric=FC_CONNECTIVITY_METRIC,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_subject_dir = output_dir / "per_subject_raw"
    per_subject_dir.mkdir(exist_ok=True)

    if subject_ids is None:
        subject_ids = list(range(1, 49))

    ratings = np.loadtxt(str(DATA_DIR / "ratings.txt"), delimiter=",")
    ratings_dict = {int(row[0]): {"lo": float(row[1]), "hi": float(row[2])} for row in ratings}

    feature_rows = []
    preproc_rows = []
    embedding_rows = []
    failed_pairs = []

    print("\nRunning STEW TDA pipeline v3")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {output_dir}")
    print(f"Channels: {ALL_CHANNELS}")
    print(f"Regions: {dict(REGION_CHANNELS)}")
    print(f"Normalization: {normalization_mode}")
    print(f"Surrogates per method: {surrogate_n}")
    print(f"Window seconds: {window_seconds}")
    print(f"FC connectivity metric: {fc_connectivity_metric}")

    pbar = tqdm(subject_ids, desc="Subjects", unit="subject")
    for sid in pbar:
        try:
            results = run_subject_pair(
                sid,
                normalization_mode=normalization_mode,
                surrogate_n=surrogate_n,
                iaaft_iterations=iaaft_iterations,
                window_seconds=window_seconds,
                fc_connectivity_metric=fc_connectivity_metric,
            )
            for cond in ["lo", "hi"]:
                result = results[cond]
                with open(per_subject_dir / f"sub{sid:02d}_{cond}.json", "w") as f:
                    json.dump(result, f)

                feature_rows.append(flatten_subject_condition(result, ratings_dict.get(sid, {}).get(cond)))
                preproc_rows.append({
                    "subject_id": sid,
                    "condition": cond,
                    "pre_pct_samples_abs_gt_100uv": result["qc"]["pre_metrics"]["pct_samples_abs_gt_100uv"],
                    "post_pct_samples_abs_gt_100uv": result["qc"]["post_metrics"]["pct_samples_abs_gt_100uv"],
                    "post_max_abs_uv": result["qc"]["post_metrics"]["max_abs_uv"],
                    "post_rms_uv": result["qc"]["post_metrics"]["rms_uv"],
                    "asr_status": result["qc"]["asr"]["status"],
                    "asr_calibration_clean_fraction_pct": result["qc"]["asr"]["calibration_clean_fraction_pct"],
                    "asr_calibration_clean_minutes": result["qc"]["asr"]["calibration_clean_minutes"],
                    "error": "",
                })

                embed_row = {
                    "subject_id": sid,
                    "condition": cond,
                    "tau_global_median": result["tau_global_median"],
                    "d_global_median": result["d_global_median"],
                }
                for ch in ALL_CHANNELS:
                    embed_row[f"tau_{ch}"] = result["tau_per_channel"][ch]
                    embed_row[f"d_{ch}"] = result["d_per_channel"][ch]
                embedding_rows.append(embed_row)
        except Exception as exc:
            failed_pairs.append({"subject_id": sid, "error": str(exc)})
            print(f"\nERROR subject {sid:02d}: {exc}")
            for cond in ["lo", "hi"]:
                preproc_rows.append({
                    "subject_id": sid,
                    "condition": cond,
                    "pre_pct_samples_abs_gt_100uv": -1,
                    "post_pct_samples_abs_gt_100uv": -1,
                    "post_max_abs_uv": -1,
                    "post_rms_uv": -1,
                    "asr_status": "failed",
                    "asr_calibration_clean_fraction_pct": -1,
                    "asr_calibration_clean_minutes": -1,
                    "error": str(exc),
                })

    features = pd.DataFrame(feature_rows)
    embed = pd.DataFrame(embedding_rows)
    preproc = pd.DataFrame(preproc_rows)

    features.to_csv(output_dir / "subject_level_features.csv", index=False)
    embed.to_csv(output_dir / "embedding_params.csv", index=False)
    preproc.to_csv(output_dir / "preprocessing_log.csv", index=False)

    config = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "fs": FS,
        "channels": ALL_CHANNELS,
        "confirmatory_frontal_channels": CONFIRMATORY_FRONTAL_CHANNELS,
        "region_channels": dict(REGION_CHANNELS),
        "fc_region_channels": dict(FC_REGION_CHANNELS),
        "fc_connectivity_metric": fc_connectivity_metric,
        "preprocessing_sequence": [
            "ADC_to_microvolts",
            "highpass_1Hz",
            "line_noise_notch_50Hz",
            "ASR_on_subject_low_condition",
            "average_reference",
            "alpha_band_8_13Hz",
        ],
        "references": {
            "dataset": "Lim et al. 2018, IEEE TNSRE, PMID 30281467",
            "preprocessing_sequence_reported": "Safari et al. 2024, Scientific Reports s41598-024-59652-w",
            "persistence_landscapes": "Bubenik 2015, JMLR 16:77-102",
            "persistence_images": "Adams et al. 2017, JMLR 18:1-35",
            "surrogate_testing": "Theiler et al. 1992, Physica D 58:77-94",
            "iaaft_surrogates": "Schreiber and Schmitz 1996, Phys Rev Lett 77:635-638",
            "fc_topology_motivation": "Giusti et al. 2015, PNAS 112:11589-11594; Yan et al. 2023, IEEE TIM 72:1-11",
            "circular_coordinates": "de Silva et al. 2011, Discrete Comput Geom 45:737-759",
        },
        "window_seconds": float(window_seconds),
        "window_samples": int(FS * float(window_seconds)),
        "subsample_step": SUBSAMPLE_STEP,
        "normalization_mode": normalization_mode,
        "landscape_layers": LANDSCAPE_LAYERS,
        "landscape_grid_size": LANDSCAPE_GRID_SIZE,
        "persistence_image_shape": list(PERSISTENCE_IMAGE_SHAPE),
        "persistence_image_sigma": PERSISTENCE_IMAGE_SIGMA,
        "surrogate_methods": list(SURROGATE_METHODS),
        "surrogate_n_per_method": surrogate_n,
        "surrogate_iaaft_iterations": iaaft_iterations,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mne": mne.__version__,
        "failed_subjects": failed_pairs,
    }

    with open(output_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved feature table: {features.shape}")
    print(f"Saved embedding table: {embed.shape}")
    print(f"Saved preprocessing log: {preproc.shape}")
    print(f"Failed subjects: {len(failed_pairs)}")
    return features


def quick_check(
    subject_id=1,
    normalization_mode=POINT_CLOUD_NORMALIZATION,
    surrogate_n=SURROGATE_N,
    iaaft_iterations=SURROGATE_IAAFT_ITER,
    window_seconds=WINDOW_SECONDS,
    fc_connectivity_metric=FC_CONNECTIVITY_METRIC,
):
    results = run_subject_pair(
        subject_id,
        normalization_mode=normalization_mode,
        surrogate_n=surrogate_n,
        iaaft_iterations=iaaft_iterations,
        window_seconds=window_seconds,
        fc_connectivity_metric=fc_connectivity_metric,
    )
    print(f"\nQuick check subject {subject_id:02d}")
    for cond in ["lo", "hi"]:
        result = results[cond]
        print(f"[{result['condition_label'].upper()}]")
        print(f"  ASR calibration clean fraction: {result['qc']['asr']['calibration_clean_fraction_pct']:.1f}%")
        print(f"  Post-ASR abs>100uV: {result['qc']['post_metrics']['pct_samples_abs_gt_100uv']:.2f}%")
        print(f"  Global tau median: {result['tau_global_median']}")
        print(f"  Global d median: {result['d_global_median']}")
        for ch in ["F4", "F3", "T7", "O1"]:
            agg = result["channels"][ch]["aggregated"]
            print(
                f"    {ch}: median_H1={agg['median_max_persistence_H1']:.4f} "
                f"landscape={agg['median_landscape_total_area_H1']:.4f} "
                f"pi_mass={agg['median_persistence_image_mass_H1']:.4f} "
                f"alpha_power={agg['median_alpha_power']:.4f} "
                f"valid={agg['n_valid_windows']}"
            )
        fc_agg = result["fc_regions"]["whole_head"]["aggregated"]
        print(
            f"    FC whole_head: median_landscape={fc_agg['median_fc_landscape_total_area_H1']:.4f} "
            f"volatility={fc_agg['volatility_fc_landscape_total_area_H1']:.4f} "
            f"mean_similarity={fc_agg['mean_fc_mean_similarity']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="STEW topology pipeline v3")
    parser.add_argument("mode", nargs="?", default="full", choices=["full", "sample", "check"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--normalization", default=POINT_CLOUD_NORMALIZATION, choices=["raw", "unit_diameter"])
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--surrogates", type=int, default=SURROGATE_N)
    parser.add_argument("--iaaft-iterations", type=int, default=SURROGATE_IAAFT_ITER)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--fc-metric", default=FC_CONNECTIVITY_METRIC, choices=list(FC_CONNECTIVITY_METRICS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "check":
        quick_check(
            subject_id=args.subject,
            normalization_mode=args.normalization,
            surrogate_n=args.surrogates,
            iaaft_iterations=args.iaaft_iterations,
            window_seconds=args.window_seconds,
            fc_connectivity_metric=args.fc_metric,
        )
    elif args.mode == "sample":
        run_full_pipeline(
            output_dir=args.output_dir,
            subject_ids=list(range(1, 6)),
            normalization_mode=args.normalization,
            surrogate_n=args.surrogates,
            iaaft_iterations=args.iaaft_iterations,
            window_seconds=args.window_seconds,
            fc_connectivity_metric=args.fc_metric,
        )
    else:
        run_full_pipeline(
            output_dir=args.output_dir,
            normalization_mode=args.normalization,
            surrogate_n=args.surrogates,
            iaaft_iterations=args.iaaft_iterations,
            window_seconds=args.window_seconds,
            fc_connectivity_metric=args.fc_metric,
        )
