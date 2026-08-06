import torch
import torchaudio
import os
import io
from dotenv import load_dotenv
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.visualizers import generate_mel_spectrogram

matplotlib.use('Agg')

try:
    if hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# Load token
load_dotenv()
token = os.getenv("HF_TOKEN")

# Load model
model_id = "mo-thecreator/Deepfake-audio-detection"
extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
model = AutoModelForAudioClassification.from_pretrained(model_id, token=token)
model.eval()

label_map = {int(k): v for k, v in model.config.id2label.items()}


def detect_audio(audio_file, chunk_seconds=3):
    waveform, sr = torchaudio.load(audio_file)

    # Mono conversion
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    waveform = waveform.squeeze()
    duration = waveform.shape[0] / sr
    chunk_samples = int(chunk_seconds * sr)

    segment_preds = []
    fake_segments = 0
    time_series_data = []

    for i in range(0, len(waveform), chunk_samples):
        chunk = waveform[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            if len(segment_preds) == 0:
                # Pad short audio chunk with zeros if it is the only segment
                padding = torch.zeros(chunk_samples - len(chunk))
                chunk = torch.cat([chunk, padding])
            else:
                break

        inputs = extractor(
            chunk.numpy(), sampling_rate=sr, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            confidence = probs[label_id].item()
            label = label_map[label_id].capitalize()

        start = i / sr
        end = min((i + chunk_samples) / sr, duration)
        mid_time = (start + end) / 2.0

        # Estimate fake probability
        fake_p = confidence if label.lower() == "fake" else (1.0 - confidence)

        segment_preds.append({
            "start": start,
            "end": end,
            "label": label,
            "confidence": f"{confidence:.2f}",
            "fake_prob": fake_p
        })

        time_series_data.append({
            "timestamp": mid_time,
            "fake_probability": fake_p,
            "label": label
        })

        if label.lower() == "fake":
            fake_segments += 1

    verdict = "Fake" if fake_segments >= 1 else "Real"
    avg_conf = (
        np.mean([float(seg["confidence"]) for seg in segment_preds])
        if segment_preds else 0.0
    )

    mel_img = generate_mel_spectrogram(audio_file)

    return verdict, f"{avg_conf:.2f}", segment_preds, mel_img, time_series_data


def get_waveform_image(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy()

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
