import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import torch
from dotenv import load_dotenv
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.audio_utils import load_audio, resample_waveform
from utils.visualizers import generate_mel_spectrogram

matplotlib.use('Agg')

# Load token
load_dotenv()
token = os.getenv("HF_TOKEN")

# Load model
model_id = "mo-thecreator/Deepfake-audio-detection"
try:
    extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
    model = AutoModelForAudioClassification.from_pretrained(model_id, token=token)
    model.eval()
    label_map = {int(k): v for k, v in model.config.id2label.items()}
except Exception as e:
    extractor = None
    model = None
    label_map = {0: "Fake", 1: "Real"}


def detect_audio(audio_file, chunk_seconds=3):
    """
    Detects whether an audio recording contains synthetic/deepfake speech.
    Supports audio file paths, bytes, or file-like objects.
    Returns:
        verdict (str): "Fake" or "Real"
        avg_conf (str): String representation of average confidence (0.00-1.00)
        segment_preds (list): List of dicts for each time chunk
        mel_img (PIL.Image): Spectrogram visualization
        time_series_data (list): List of dicts for time-series plotting
    """
    if model is None or extractor is None:
        raise RuntimeError("Audio classification model failed to initialize.")

    waveform, sr = load_audio(audio_file)

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != 16000:
        waveform = resample_waveform(waveform, sr, 16000)
        sr = 16000

    waveform = waveform.squeeze()
    if waveform.ndim == 0 or len(waveform) == 0:
        raise ValueError("Audio contains no readable audio samples.")

    duration = waveform.shape[0] / sr
    chunk_samples = int(chunk_seconds * sr)

    segment_preds = []
    fake_segments = 0
    time_series_data = []

    for i in range(0, len(waveform), chunk_samples):
        chunk = waveform[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            # Pad short audio chunk with zeros to meet model window size
            padding = torch.zeros(chunk_samples - len(chunk))
            chunk = torch.cat([chunk, padding])

        inputs = extractor(
            chunk.cpu().numpy(), sampling_rate=sr, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            confidence = probs[label_id].item()
            label = label_map.get(label_id, "Unknown").capitalize()

        start = i / sr
        end = min((i + chunk_samples) / sr, duration)
        mid_time = (start + end) / 2.0

        # Estimate fake probability (index 0 is typically fake in mo-thecreator model)
        if label.lower() == "fake":
            fake_p = confidence
        elif label.lower() == "real":
            fake_p = 1.0 - confidence
        else:
            fake_p = float(probs[0].item())

        segment_preds.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "label": label,
            "confidence": f"{confidence:.2f}",
            "fake_prob": round(fake_p, 4)
        })

        time_series_data.append({
            "timestamp": round(mid_time, 2),
            "fake_probability": round(fake_p, 4),
            "label": label
        })

        if label.lower() == "fake":
            fake_segments += 1

    verdict = "Fake" if fake_segments >= 1 else "Real"
    avg_conf = (
        np.mean([float(seg["confidence"]) for seg in segment_preds])
        if segment_preds else 0.0
    )

    try:
        mel_img = generate_mel_spectrogram(audio_file)
    except Exception:
        mel_img = generate_mel_spectrogram((waveform.unsqueeze(0), sr))

    return verdict, f"{avg_conf:.2f}", segment_preds, mel_img, time_series_data


def get_waveform_image(audio_path):
    """
    Renders an amplitude waveform image from an audio file or waveform tensor.
    """
    if isinstance(audio_path, tuple) and len(audio_path) == 2:
        waveform, sr = audio_path
    elif isinstance(audio_path, torch.Tensor):
        waveform = audio_path
    else:
        waveform, sr = load_audio(audio_path)

    if isinstance(waveform, torch.Tensor):
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0).cpu().numpy()
        else:
            waveform = waveform.cpu().numpy()
    elif isinstance(waveform, np.ndarray):
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(waveform, color="#38bdf8", alpha=0.85, linewidth=0.8)
    ax.set_title("Audio Amplitude Waveform", fontsize=10, color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(
        buf, format='png', facecolor=fig.get_facecolor(),
        bbox_inches='tight', dpi=150
    )
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)

