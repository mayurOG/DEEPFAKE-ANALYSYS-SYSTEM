import os
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
from transformers import AutoImageProcessor, SiglipForImageClassification
from dotenv import load_dotenv

# Load token and model
load_dotenv()
token = os.getenv("HF_TOKEN")
model_id = "prithivMLmods/deepfake-detector-model-v1"
processor = AutoImageProcessor.from_pretrained(model_id, token=token)
model = SiglipForImageClassification.from_pretrained(model_id, token=token)
model.eval()

label_map = {"0": "Fake", "1": "Real"}


def detect_video(video_file, frame_skip=15):
    """
    Analyzes video frame-by-frame, extracts audio for multimodal analysis,
    and returns frame timelines, metrics, and fake frame thumbnails.
    """
    # Save uploaded video to temp file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        if hasattr(video_file, "read"):
            temp_file.write(video_file.read())
            if hasattr(video_file, "seek"):
                video_file.seek(0)
        else:
            temp_file.write(open(video_file, "rb").read())
        temp_path = temp_file.name

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_count = 0
    real_count = 0
    fake_count = 0
    fake_frames = []
    annotated_frames = []
    timeline_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        timestamp = frame_count / fps
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            verdict = label_map[str(label_id)]
            confidence = probs[label_id].item()
            fake_prob = probs[0].item()

        # Add visual overlay on frame
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"{verdict} ({confidence:.2f})",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255) if verdict == "Fake" else (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        annotated_frames.append(annotated)
        timeline_data.append({
            "frame": frame_count,
            "timestamp": timestamp,
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": fake_prob
        })

        if verdict == "Fake":
            fake_count += 1
            fake_frames.append(annotated)
        else:
            real_count += 1

    cap.release()

    total_analyzed = real_count + fake_count
    verdict = "Fake" if fake_count >= 2 else "Real"
    fake_pct = (fake_count / total_analyzed * 100) if total_analyzed > 0 else 0.0

    # 1. Novelty: Biometric Remote Photoplethysmography (rPPG) Pulse Coherence
    biometric_rppg_res = None
    try:
        from detector.biometric_rppg import analyze_video_rppg
        biometric_rppg_res = analyze_video_rppg(annotated_frames, fps=fps)
    except Exception:
        biometric_rppg_res = None

    # 2. Novelty: C2PA Video Container Provenance
    c2pa_res = None
    try:
        from detector.c2pa_provenance import audit_c2pa_provenance
        if os.path.exists(temp_path):
            v_bytes = open(temp_path, "rb").read(500000)
            c2pa_res = audit_c2pa_provenance(v_bytes)
    except Exception:
        c2pa_res = None

    # Attempt audio track analysis from video file
    audio_analysis = None
    try:
        from detector.audio_detector import detect_audio
        audio_analysis = detect_audio(temp_path)
    except Exception:
        audio_analysis = None

    # 3. Novelty: Calibrated Multi-Modal Evidence Fusion
    calibrated_fusion_res = None
    try:
        from detector.calibrated_fusion import CalibratedForensicFusion
        aud_fake_p = None
        if audio_analysis:
            a_verdict = audio_analysis[0]
            a_conf = float(audio_analysis[1])
            aud_fake_p = a_conf if a_verdict == "Fake" else (1.0 - a_conf)

        rppg_liveness = biometric_rppg_res.get("liveness_confidence") if biometric_rppg_res and biometric_rppg_res.get("is_valid") else None

        calibrated_fusion_res = CalibratedForensicFusion.fuse_evidence(
            neural_visual_fake_p=(fake_pct / 100.0),
            neural_audio_fake_p=aud_fake_p,
            biometric_liveness_score=rppg_liveness,
            c2pa_result=c2pa_res,
        )
    except Exception:
        calibrated_fusion_res = None

    summary = {
        "total_analyzed_frames": total_analyzed,
        "real_frames": real_count,
        "fake_frames": fake_count,
        "fake_percentage": f"{fake_pct:.1f}%",
        "video_fps": f"{fps:.1f}",
        "biometric_rppg": biometric_rppg_res,
        "c2pa_provenance": c2pa_res,
        "calibrated_fusion": calibrated_fusion_res,
    }

    # Clean up temp file
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return (
        verdict,
        total_analyzed,
        fake_count,
        summary,
        fake_frames,
        annotated_frames,
        timeline_data,
        audio_analysis
    )
