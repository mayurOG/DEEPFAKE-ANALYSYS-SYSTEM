import cv2
import numpy as np
from PIL import Image
import scipy.signal


def extract_face_rois(face_img_rgb: np.ndarray):
    """
    Extracts forehead, left cheek, and right cheek sub-regions
    from a cropped face image.
    """
    h, w = face_img_rgb.shape[:2]
    # Forehead: top-middle area
    fh = face_img_rgb[int(0.12 * h):int(0.28 * h), int(0.28 * w):int(0.72 * w)]
    # Left cheek
    lc = face_img_rgb[int(0.48 * h):int(0.68 * h), int(0.15 * w):int(0.38 * w)]
    # Right cheek
    rc = face_img_rgb[int(0.48 * h):int(0.68 * h), int(0.62 * w):int(0.85 * w)]
    return fh, lc, rc


def compute_chrom_pulse(rgb_roi: np.ndarray) -> float:
    """
    Computes chrominance-based (CHROM) hemodynamic blood volume pulse proxy.
    X = 3R - 2G, Y = 1.5R + G - 1.5B (de Haan & Jeanne, IEEE TBME).
    """
    if rgb_roi.size == 0:
        return 0.0
    r = np.mean(rgb_roi[:, :, 0])
    g = np.mean(rgb_roi[:, :, 1])
    b = np.mean(rgb_roi[:, :, 2])
    # Normalize by luminance to remove global lighting drift
    denom = r + g + b + 1e-6
    rn, gn, bn = r / denom, g / denom, b / denom
    x = 3.0 * rn - 2.0 * gn
    y = 1.5 * rn + gn - 1.5 * bn
    return float(x - 0.7 * y)


def analyze_video_rppg(frames: list, fps: float = 30.0) -> dict:
    """
    Extracts multi-region rPPG signals across video frames,
    computes cardiac pulse rhythmicity and inter-region cross-coherence.
    """
    if len(frames) < 15:
        return {
            "error": "Insufficient frames for hemodynamic pulse analysis (minimum 15 required).",
            "is_valid": False,
        }

    fh_signals = []
    lc_signals = []
    rc_signals = []

    for frame in frames:
        if isinstance(frame, Image.Image):
            f_rgb = np.array(frame.convert("RGB"))
        elif isinstance(frame, np.ndarray):
            f_rgb = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            continue

        fh, lc, rc = extract_face_rois(f_rgb)
        fh_signals.append(compute_chrom_pulse(fh))
        lc_signals.append(compute_chrom_pulse(lc))
        rc_signals.append(compute_chrom_pulse(rc))

    n_samples = len(fh_signals)
    if n_samples < 15:
        return {"error": "Could not extract sufficient facial ROIs.", "is_valid": False}

    # Bandpass filter in physiological cardiac range: 0.75 Hz to 2.5 Hz (45 to 150 BPM)
    nyq = 0.5 * fps
    lowcut = min(0.75, 0.4 * nyq)
    highcut = min(2.5, 0.9 * nyq)

    # Detrend
    sig_lc = scipy.signal.detrend(np.array(lc_signals))
    sig_rc = scipy.signal.detrend(np.array(rc_signals))
    sig_fh = scipy.signal.detrend(np.array(fh_signals))

    try:
        b, a = scipy.signal.butter(2, [lowcut / nyq, highcut / nyq], btype="band")
        filt_lc = scipy.signal.filtfilt(b, a, sig_lc)
        filt_rc = scipy.signal.filtfilt(b, a, sig_rc)
        filt_fh = scipy.signal.filtfilt(b, a, sig_fh)
    except Exception:
        filt_lc = sig_lc
        filt_rc = sig_rc
        filt_fh = sig_fh

    # Combined BVP pulse wave
    combined_bvp = (filt_lc + filt_rc + filt_fh) / 3.0

    # Cross-regional Pearson coherence
    def corr(a, b):
        sa, sb = np.std(a), np.std(b)
        if sa < 1e-7 or sb < 1e-7:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    c_lr = corr(filt_lc, filt_rc)
    c_lf = corr(filt_lc, filt_fh)
    c_rf = corr(filt_rc, filt_fh)
    coherence_score = max(0.0, float(np.mean([c_lr, c_lf, c_rf])))

    # Power Spectral Density & Estimated Heart Rate (BPM)
    freqs, psd = scipy.signal.welch(combined_bvp, fs=fps, nperseg=min(len(combined_bvp), 64))
    cardiac_mask = (freqs >= 0.75) & (freqs <= 2.5)
    if np.any(cardiac_mask) and np.max(psd[cardiac_mask]) > 1e-9:
        peak_freq = freqs[cardiac_mask][np.argmax(psd[cardiac_mask])]
        estimated_bpm = round(float(peak_freq * 60.0), 1)
        peak_power = np.max(psd[cardiac_mask])
        total_power = np.sum(psd[cardiac_mask]) + 1e-9
        snr = float(peak_power / total_power)
    else:
        estimated_bpm = 0.0
        snr = 0.0

    # Biological liveness determination:
    # Deepfake / GAN / diffusion faces produce uncoupled or noisy signals
    is_live = (coherence_score > 0.45) and (snr > 0.20)
    liveness_confidence = round(min(1.0, (coherence_score * 0.6) + (snr * 0.4)), 3)

    verdict = "Authentic Biological Hemodynamics" if is_live else "Synthetic Biometric Pulse Disruption"

    timestamps = [round(i / fps, 2) for i in range(len(combined_bvp))]
    norm_bvp = ((combined_bvp - np.mean(combined_bvp)) / (np.std(combined_bvp) + 1e-7)).tolist()

    return {
        "is_valid": True,
        "coherence_score": round(coherence_score, 3),
        "estimated_bpm": estimated_bpm,
        "snr": round(snr, 3),
        "liveness_confidence": liveness_confidence,
        "verdict": verdict,
        "is_live": is_live,
        "pulse_wave": norm_bvp,
        "timestamps": timestamps,
        "regions_correlation": {
            "left_right_cheeks": round(c_lr, 3),
            "left_cheek_forehead": round(c_lf, 3),
            "right_cheek_forehead": round(c_rf, 3),
        },
    }


def analyze_static_face_micro_vasculature(face_img: Image.Image) -> dict:
    """
    For static single images, analyzes micro-vascular color variation and
    capillary hemoglobin diffusion entropy across cheek and forehead regions.
    """
    f_rgb = np.array(face_img.convert("RGB"))
    fh, lc, rc = extract_face_rois(f_rgb)

    def roi_entropy(roi):
        if roi.size == 0:
            return 0.0
        # Color variance in green channel relative to red (hemoglobin absorption band)
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 0].astype(np.float32)
        ratio = g / (r + 1e-6)
        return float(np.std(ratio))

    e_fh = roi_entropy(fh)
    e_lc = roi_entropy(lc)
    e_rc = roi_entropy(rc)
    mean_vascular_diffusion = float(np.mean([e_fh, e_lc, e_rc]))

    # AI face smoothing typically destroys vascular micro-entropy (< 0.02)
    is_synthetic_smooth = mean_vascular_diffusion < 0.025
    verdict = (
        "Potential Generative Skin Smoothing (Low Vascular Detail)"
        if is_synthetic_smooth
        else "Natural Capillary Vascular Texture"
    )

    return {
        "mean_vascular_entropy": round(mean_vascular_diffusion, 4),
        "is_synthetic_smooth": is_synthetic_smooth,
        "verdict": verdict,
    }
