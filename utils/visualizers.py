import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio

matplotlib.use('Agg')


def generate_ela_image(
    image: Image.Image, quality: int = 90, scale: float = 15.0
) -> Image.Image:
    """
    Generates an Error Level Analysis (ELA) heatmap image.
    ELA highlights areas of an image that have different compression levels.
    """
    img = image.convert("RGB")

    # Save image to memory at a specific JPEG compression level
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Re-open compressed image
    compressed_img = Image.open(buffer)

    # Calculate absolute difference between original and compressed
    ela_img = ImageChops.difference(img, compressed_img)

    # Scale up the difference brightness for visual enhancement
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1

    scale_factor = (255.0 / max_diff) * (scale / 10.0)
    enhancer = ImageEnhance.Brightness(ela_img)
    ela_enhanced = enhancer.enhance(scale_factor)

    return ela_enhanced


def generate_fft_spectrum(image: Image.Image) -> Image.Image:
    """
    Generates a 2D Fourier Transform (FFT) Magnitude Spectrum map.
    Identifies high-frequency periodic grid artifacts common in GANs & Diffusion
    """
    # Convert image to grayscale numpy array
    gray_img = image.convert("L")
    img_arr = np.array(gray_img, dtype=np.float32)

    # 2D Fast Fourier Transform
    fft2 = np.fft.fft2(img_arr)
    fft_shifted = np.fft.fftshift(fft2)

    # Compute log magnitude spectrum
    magnitude_spectrum = np.log(1 + np.abs(fft_shifted))

    # Plot nicely with matplotlib colormap
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(magnitude_spectrum, cmap='inferno')
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)


def generate_mel_spectrogram(audio_source) -> Image.Image:
    """
    Generates a Mel-Spectrogram visualization for an audio file or waveform.
    """
    try:
        from utils.audio_utils import load_audio

        if isinstance(audio_source, tuple) and len(audio_source) == 2:
            waveform, sr = audio_source
        elif isinstance(audio_source, torch.Tensor):
            waveform = audio_source
            sr = 16000
        else:
            waveform, sr = load_audio(audio_source)

        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Try MelSpectrogram via torchaudio
        try:
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=1024, hop_length=512, n_mels=64
            )
            mel_spec = mel_transform(waveform)
            mel_spec_db = torchaudio.transforms.AmplitudeToDB()(
                mel_spec
            ).squeeze().cpu().numpy()
        except Exception:
            # Fallback using scipy spectrogram
            import scipy.signal
            audio_1d = waveform.squeeze().cpu().numpy()
            _, _, Sxx = scipy.signal.spectrogram(audio_1d, fs=sr, nperseg=1024, noverlap=512)
            mel_spec_db = 10 * np.log10(np.maximum(Sxx, 1e-10))

        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.imshow(
            mel_spec_db, origin='lower', aspect='auto', cmap='magma'
        )
        ax.set_title(
            "Mel-Frequency Spectrogram (Acoustic Forensic Analysis)",
            color='white', fontsize=12, pad=10
        )
        ax.set_xlabel("Time Frames", color='#aaaaaa')
        ax.set_ylabel("Mel Frequency Hz", color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        plt.colorbar(im, ax=ax, format='%+2.0f dB')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(
            buf, format='png', facecolor=fig.get_facecolor(),
            bbox_inches='tight', dpi=150
        )
        buf.seek(0)
        plt.close(fig)

        return Image.open(buf)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(
            0.5, 0.5, f"Spectrogram unavailable: {str(e)}",
            color="red", ha="center"
        )
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)


def plot_rapsd_curve(rapsd_metrics: dict):
    """
    Renders an interactive Plotly dual-line curve of the Radially Averaged
    Power Spectral Distribution comparing input vs. natural 1/f^alpha baseline.
    """
    import plotly.graph_objects as go

    radii = rapsd_metrics.get("radii", [])
    observed = rapsd_metrics.get("observed_energy", [])
    natural = rapsd_metrics.get("natural_baseline", [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=radii, y=observed,
        mode="lines",
        name="Observed Spectrum",
        line=dict(color="#38bdf8", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=radii, y=natural,
        mode="lines",
        name="Natural 1/f^alpha Baseline",
        line=dict(color="#f43f5e", width=2, dash="dash")
    ))
    fig.update_layout(
        title="Radially Averaged Power Spectral Distribution (RAPSD)",
        xaxis_title="Radial Spatial Frequency (r)",
        yaxis_title="Normalized Log Power",
        paper_bgcolor="rgba(15,23,42,0.8)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_rppg_wave(rppg_metrics: dict):
    """
    Renders an interactive Plotly line chart of the blood volume pulse (BVP) wave.
    """
    import plotly.graph_objects as go

    timestamps = rppg_metrics.get("timestamps", [])
    pulse_wave = rppg_metrics.get("pulse_wave", [])
    bpm = rppg_metrics.get("estimated_bpm", 0)
    coherence = rppg_metrics.get("coherence_score", 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=pulse_wave,
        mode="lines",
        name="Hemodynamic BVP Pulse Wave",
        line=dict(color="#ec4899", width=2.5)
    ))
    fig.update_layout(
        title=f"Remote Photoplethysmography (rPPG) Pulse Wave | Est. HR: {bpm} BPM | Coherence: {coherence:.2f}",
        xaxis_title="Time (Seconds)",
        yaxis_title="Normalized Blood Volume Pulse",
        paper_bgcolor="rgba(15,23,42,0.8)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_fusion_bar(fusion_metrics: dict):
    """
    Renders an interactive Plotly bar chart comparing evidence across modalities.
    """
    import plotly.graph_objects as go

    vectors = fusion_metrics.get("evidence_vectors", [])
    if not vectors:
        return None
    modalities = [v["modality"] for v in vectors]
    scores = [v["p_fake"] for v in vectors]
    colors = ["#ef4444" if s >= 0.55 else ("#22c55e" if s <= 0.45 else "#eab308") for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=modalities, y=scores,
        marker_color=colors,
        text=[f"{s*100:.1f}%" for s in scores],
        textposition="auto"
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title="Multi-Modal Forensic Consensus & Modality Fake Probability",
        yaxis_title="Synthetic Probability P(Fake)",
        yaxis=dict(range=[0, 1.05]),
        paper_bgcolor="rgba(15,23,42,0.8)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
