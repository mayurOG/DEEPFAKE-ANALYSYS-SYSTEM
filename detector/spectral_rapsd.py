import numpy as np
from PIL import Image
import cv2


def compute_rapsd(image_input, max_dim=512) -> dict:
    """
    Computes the 1D Radially Averaged Power Spectral Distribution (RAPSD)
    and checks for 1/f^alpha natural image power-law deviation.

    SOTA Basis: Generative AI models (GANs, Latent Diffusion, Transposed Convolutions)
    leave distinct high-frequency periodic lattice peaks and slope anomalies
    in the Fourier spectrum (Durall et al., CVPR; Frank et al., ICML).
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("L"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img_np = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        else:
            img_np = image_input
    else:
        img_np = np.array(Image.open(image_input).convert("L"))

    # Resize if too large for consistent radial comparison and speed
    h, w = img_np.shape
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img_np.shape

    # Apply Hann window to reduce edge discontinuity leakage in FFT
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    img_windowed = (img_np.astype(np.float32) - np.mean(img_np)) * window

    # 2D Fast Fourier Transform
    fft2 = np.fft.fft2(img_windowed)
    fft_shifted = np.fft.fftshift(fft2)
    power_spectrum = np.abs(fft_shifted) ** 2

    # Radial integration
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_radius = min(cx, cy)

    radial_sum = np.bincount(radii.ravel(), weights=power_spectrum.ravel())
    radial_count = np.bincount(radii.ravel())

    # Avoid division by zero
    valid_mask = (radial_count > 0) & (np.arange(len(radial_count)) <= max_radius)
    radii_bins = np.arange(len(radial_count))[valid_mask][1:]  # Exclude DC component (r=0)
    radial_profile = radial_sum[valid_mask][1:] / np.maximum(radial_count[valid_mask][1:], 1)

    # Normalize radial profile to [0, 1] range for unified plotting
    norm_profile = (radial_profile - np.min(radial_profile)) / (
        np.max(radial_profile) - np.min(radial_profile) + 1e-12
    )

    # Fit natural 1/f^alpha power-law in log-log scale
    log_r = np.log(radii_bins.astype(np.float64))
    log_power = np.log(np.maximum(radial_profile, 1e-12))

    # Linear regression in log-log domain (ignoring extreme low/high freqs)
    mid_start = max(1, int(len(radii_bins) * 0.1))
    mid_end = max(mid_start + 2, int(len(radii_bins) * 0.85))
    slope, intercept = np.polyfit(log_r[mid_start:mid_end], log_power[mid_start:mid_end], deg=1)

    natural_log_fit = slope * log_r + intercept
    natural_curve = np.exp(natural_log_fit)
    norm_natural_curve = (natural_curve - np.min(natural_curve)) / (
        np.max(natural_curve) - np.min(natural_curve) + 1e-12
    )

    # Calculate Spectral Divergence Index (SDI) and High-Frequency Residual Energy
    residual = np.abs(norm_profile - norm_natural_curve)
    high_freq_idx = int(len(norm_profile) * 0.6)
    high_freq_anomaly_energy = float(np.mean(residual[high_freq_idx:]))

    # Check for synthetic periodic frequency spikes (GAN/diffusion signature)
    grad2 = np.abs(np.diff(norm_profile, n=2))
    spike_roughness = float(np.mean(grad2[int(len(grad2) * 0.4):]))

    # Heuristic spectral risk score
    spectral_risk = min(1.0, (high_freq_anomaly_energy * 3.5) + (spike_roughness * 15.0))

    verdict = "Suspicious Synthetic Spectrum" if spectral_risk > 0.45 else "Natural Fourier Decay"

    return {
        "radii": radii_bins.tolist(),
        "observed_energy": norm_profile.tolist(),
        "natural_baseline": norm_natural_curve.tolist(),
        "spectral_slope_alpha": round(float(-slope), 3),
        "high_freq_anomaly_energy": round(high_freq_anomaly_energy, 4),
        "spike_roughness": round(spike_roughness, 4),
        "spectral_risk_score": round(spectral_risk, 3),
        "verdict": verdict,
        "is_abnormal": spectral_risk > 0.45,
    }
