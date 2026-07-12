import numpy as np


def _trapz(y, x):
    integrator = getattr(np, "trapezoid", np.trapz)
    return integrator(y, x)
from scipy.signal import hilbert
from ripser import ripser as compute_rips


def _finite_h1_pairs(h1_diagram):
    arr = np.asarray(h1_diagram, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected an H1 diagram with shape (n_bars, 2).")
    finite = arr[np.isfinite(arr[:, 1])]
    return finite[finite[:, 1] > finite[:, 0]]


def _default_axis_limits(finite_pairs, normalization_mode):
    if normalization_mode == "unit_diameter":
        return 0.0, 1.0, 0.0, 1.0

    if len(finite_pairs) == 0:
        return 0.0, 1.0, 0.0, 1.0

    persistence = finite_pairs[:, 1] - finite_pairs[:, 0]
    birth_max = float(np.max(finite_pairs[:, 0])) if len(finite_pairs) else 1.0
    death_max = float(np.max(finite_pairs[:, 1])) if len(finite_pairs) else 1.0
    pers_max = float(np.max(persistence)) if len(persistence) else 1.0

    axis_max = max(death_max, birth_max, pers_max, 1e-6)
    return 0.0, axis_max, 0.0, max(pers_max, axis_max)


def compute_persistence_landscape(
    h1_diagram,
    normalization_mode="unit_diameter",
    n_layers=3,
    n_grid=64,
):
    finite_pairs = _finite_h1_pairs(h1_diagram)
    t_min, t_max, _, _ = _default_axis_limits(finite_pairs, normalization_mode)
    if t_max <= t_min:
        t_max = t_min + 1.0

    grid = np.linspace(t_min, t_max, n_grid)
    if len(finite_pairs) == 0:
        landscape = np.zeros((n_layers, n_grid), dtype=float)
    else:
        births = finite_pairs[:, 0][:, None]
        deaths = finite_pairs[:, 1][:, None]
        values = np.maximum(0.0, np.minimum(grid[None, :] - births, deaths - grid[None, :]))
        sorted_vals = np.sort(values, axis=0)[::-1]
        landscape = np.zeros((n_layers, n_grid), dtype=float)
        use_layers = min(n_layers, sorted_vals.shape[0])
        landscape[:use_layers] = sorted_vals[:use_layers]

    layer_areas = [float(_trapz(layer, grid)) for layer in landscape]
    total_area = float(np.sum(layer_areas))
    l2 = float(np.sqrt(_trapz(np.sum(np.square(landscape), axis=0), grid)))
    peak = float(np.max(landscape)) if landscape.size else 0.0

    summary = {
        "landscape_layer1_area_H1": layer_areas[0] if len(layer_areas) > 0 else 0.0,
        "landscape_layer2_area_H1": layer_areas[1] if len(layer_areas) > 1 else 0.0,
        "landscape_layer3_area_H1": layer_areas[2] if len(layer_areas) > 2 else 0.0,
        "landscape_total_area_H1": total_area,
        "landscape_peak_H1": peak,
        "landscape_l2_norm_H1": l2,
    }
    return {
        "grid": grid,
        "landscape": landscape,
        "summary": summary,
    }


def compute_persistence_image(
    h1_diagram,
    normalization_mode="unit_diameter",
    shape=(20, 20),
    sigma=0.04,
):
    finite_pairs = _finite_h1_pairs(h1_diagram)
    birth_min, birth_max, pers_min, pers_max = _default_axis_limits(finite_pairs, normalization_mode)
    if birth_max <= birth_min:
        birth_max = birth_min + 1.0
    if pers_max <= pers_min:
        pers_max = pers_min + 1.0

    n_birth, n_pers = shape
    birth_grid = np.linspace(birth_min, birth_max, n_birth)
    pers_grid = np.linspace(pers_min, pers_max, n_pers)
    birth_centers, pers_centers = np.meshgrid(birth_grid, pers_grid, indexing="xy")
    image = np.zeros((n_pers, n_birth), dtype=float)

    if len(finite_pairs) > 0:
        births = finite_pairs[:, 0]
        persistence = finite_pairs[:, 1] - finite_pairs[:, 0]
        sigma_sq = max(sigma, 1e-6) ** 2
        for birth, pers in zip(births, persistence):
            weight = max(float(pers), 0.0)
            image += weight * np.exp(
                -((birth_centers - birth) ** 2 + (pers_centers - pers) ** 2) / (2.0 * sigma_sq)
            )

    mass = float(np.sum(image))
    peak = float(np.max(image)) if image.size else 0.0
    l2 = float(np.sqrt(np.sum(np.square(image))))
    if mass > 0:
        probs = image / mass
        entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    else:
        entropy = 0.0

    summary = {
        "persistence_image_mass_H1": mass,
        "persistence_image_peak_H1": peak,
        "persistence_image_entropy_H1": entropy,
        "persistence_image_l2_norm_H1": l2,
    }
    return {
        "birth_grid": birth_grid,
        "persistence_grid": pers_grid,
        "image": image,
        "summary": summary,
    }


def summarize_h1_representations(
    h1_diagram,
    normalization_mode="unit_diameter",
    landscape_layers=3,
    landscape_grid=64,
    image_shape=(20, 20),
    image_sigma=0.04,
):
    landscape = compute_persistence_landscape(
        h1_diagram,
        normalization_mode=normalization_mode,
        n_layers=landscape_layers,
        n_grid=landscape_grid,
    )
    image = compute_persistence_image(
        h1_diagram,
        normalization_mode=normalization_mode,
        shape=image_shape,
        sigma=image_sigma,
    )
    return {
        "landscape": landscape,
        "persistence_image": image,
        "summary": {
            **landscape["summary"],
            **image["summary"],
        },
    }


def phase_randomized_surrogate(signal, rng):
    x = np.asarray(signal, dtype=float)
    x_centered = x - np.mean(x)
    spectrum = np.fft.rfft(x_centered)
    amplitudes = np.abs(spectrum)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spectrum))
    phases[0] = 0.0
    if len(x) % 2 == 0 and len(phases) > 1:
        phases[-1] = 0.0
    randomized = amplitudes * np.exp(1j * phases)
    y = np.fft.irfft(randomized, n=len(x))
    return np.real(y + np.mean(x))


def iaaft_surrogate(signal, rng, n_iter=30):
    x = np.asarray(signal, dtype=float)
    sorted_x = np.sort(x)
    target_magnitude = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)

    for _ in range(n_iter):
        spectrum = np.fft.rfft(y)
        phase = np.angle(spectrum)
        adjusted = target_magnitude * np.exp(1j * phase)
        y = np.fft.irfft(adjusted, n=len(x))
        ranks = np.argsort(np.argsort(y))
        y = sorted_x[ranks]

    return np.asarray(y, dtype=float)


def summarize_temporal_series(values, prefix):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"mean_{prefix}": np.nan,
            f"std_{prefix}": np.nan,
            f"median_{prefix}": np.nan,
            f"cv_{prefix}": np.nan,
            f"volatility_{prefix}": np.nan,
            f"slope_{prefix}": np.nan,
            f"early_late_delta_{prefix}": np.nan,
        }

    idx = np.arange(len(arr), dtype=float)
    if len(arr) > 1:
        slope = float(np.polyfit(idx, arr, 1)[0])
        volatility = float(np.mean(np.abs(np.diff(arr))))
        split = max(1, len(arr) // 3)
        early = float(np.mean(arr[:split]))
        late = float(np.mean(arr[-split:]))
        early_late_delta = late - early
    else:
        slope = 0.0
        volatility = 0.0
        early_late_delta = 0.0

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    return {
        f"mean_{prefix}": mean_val,
        f"std_{prefix}": std_val,
        f"median_{prefix}": float(np.median(arr)),
        f"cv_{prefix}": float(std_val / (abs(mean_val) + 1e-8)),
        f"volatility_{prefix}": volatility,
        f"slope_{prefix}": slope,
        f"early_late_delta_{prefix}": early_late_delta,
    }


def compute_envelope_correlation(window_data):
    x = np.asarray(window_data, dtype=float)
    envelopes = np.abs(hilbert(x, axis=0))
    corr = np.corrcoef(envelopes, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_phase_locking_value(window_data):
    x = np.asarray(window_data, dtype=float)
    phase = np.angle(hilbert(x, axis=0))
    n_ch = phase.shape[1]
    plv = np.eye(n_ch, dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            diff = phase[:, i] - phase[:, j]
            value = float(np.abs(np.mean(np.exp(1j * diff))))
            plv[i, j] = value
            plv[j, i] = value
    return plv


def compute_alpha_correlation(window_data):
    x = np.asarray(window_data, dtype=float)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def fc_similarity_to_distance(similarity):
    sim = np.asarray(similarity, dtype=float)
    dist = 1.0 - np.clip(sim, 0.0, 1.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist


def _lift_cocycle_values(cocycle, coeff):
    lifted = {}
    for i, j, val in np.asarray(cocycle, dtype=float):
        i = int(i)
        j = int(j)
        raw = float(val)
        raw = ((raw + coeff / 2.0) % coeff) - coeff / 2.0
        lifted[(i, j)] = raw / coeff
        lifted[(j, i)] = -raw / coeff
    return lifted


def compute_circular_coordinate_summary(cloud, coeff=47):
    x = np.asarray(cloud, dtype=float)
    if x.ndim != 2 or len(x) < 10:
        return None

    result = compute_rips(x, maxdim=1, do_cocycles=True, coeff=coeff)
    diagrams = result["dgms"]
    cocycles = result.get("cocycles", [[], []])
    h1 = np.asarray(diagrams[1], dtype=float)
    finite_mask = np.isfinite(h1[:, 1]) if h1.size else np.array([], dtype=bool)
    finite = h1[finite_mask] if h1.size else np.empty((0, 2), dtype=float)
    if len(finite) == 0 or len(cocycles) < 2 or len(cocycles[1]) == 0:
        return None

    lifetimes = finite[:, 1] - finite[:, 0]
    top_idx = int(np.argmax(lifetimes))
    birth, death = finite[top_idx]
    cocycle = cocycles[1][top_idx]
    epsilon = float(birth + 0.5 * (death - birth))

    diffs = x[:, None, :] - x[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    edges = [(i, j) for i in range(len(x)) for j in range(i + 1, len(x)) if dist[i, j] <= epsilon]
    if len(edges) < len(x) - 1:
        return None

    lifted = _lift_cocycle_values(cocycle, coeff=coeff)
    anchor = 0
    col_map = {node: idx for idx, node in enumerate(range(1, len(x)))}
    A = np.zeros((len(edges), len(x) - 1), dtype=float)
    b = np.zeros(len(edges), dtype=float)

    for row_idx, (i, j) in enumerate(edges):
        if i != anchor:
            A[row_idx, col_map[i]] = -1.0
        if j != anchor:
            A[row_idx, col_map[j]] = 1.0
        b[row_idx] = lifted.get((i, j), 0.0)

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    theta = np.zeros(len(x), dtype=float)
    theta[1:] = np.mod(sol, 1.0)

    unwrapped = np.unwrap(theta * 2.0 * np.pi) / (2.0 * np.pi)
    steps = np.diff(unwrapped)
    bins = np.linspace(0.0, 1.0, 9)
    hist, _ = np.histogram(theta, bins=bins)
    probs = hist / max(np.sum(hist), 1)
    occupancy_entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0]))) if np.any(probs > 0) else 0.0
    slope = float(np.polyfit(np.arange(len(unwrapped), dtype=float), unwrapped, 1)[0]) if len(unwrapped) > 1 else 0.0
    residual = unwrapped - (slope * np.arange(len(unwrapped), dtype=float) + np.mean(unwrapped))
    residual_std = float(np.std(residual))

    return {
        "birth": float(birth),
        "death": float(death),
        "epsilon": epsilon,
        "theta": theta,
        "unwrapped_theta": unwrapped,
        "summary": {
            "circular_lifetime_H1": float(death - birth),
            "circular_mean_abs_step": float(np.mean(np.abs(steps))) if len(steps) > 0 else 0.0,
            "circular_step_std": float(np.std(steps)) if len(steps) > 0 else 0.0,
            "circular_occupancy_entropy": occupancy_entropy,
            "circular_unwrapped_slope": slope,
            "circular_linearity_residual_std": residual_std,
            "circular_total_unwrapped_range": float(np.max(unwrapped) - np.min(unwrapped)) if len(unwrapped) > 0 else 0.0,
        },
    }
