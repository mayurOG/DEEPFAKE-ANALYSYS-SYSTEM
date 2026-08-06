import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from facenet_pytorch import MTCNN
from dotenv import load_dotenv
from utils.visualizers import generate_ela_image, generate_fft_spectrum
from detector.facial_forensics import analyze_image_quality_and_noise

# Load token
load_dotenv()
token = os.getenv("HF_TOKEN")

# Load HuggingFace model
model_id = "prithivMLmods/deepfake-detector-model-v1"
processor = AutoImageProcessor.from_pretrained(model_id, token=token)
model = SiglipForImageClassification.from_pretrained(model_id, token=token)
model.eval()
model.to("cpu")  # Force CPU

label_map = {"0": "Fake", "1": "Real"}

# Face detector
mtcnn = MTCNN(keep_all=True, device="cpu")


def detect_image(input_img, fake_threshold=0.5):
    """
    Analyzes an image using Siglip Transformer, MTCNN Face detection,
    ELA, 2D FFT, and SRM noise filtering.
    """
    if isinstance(input_img, np.ndarray):
        img = Image.fromarray(input_img).convert("RGB")
    elif isinstance(input_img, Image.Image):
        img = input_img.convert("RGB")
    else:
        img = Image.open(input_img).convert("RGB")

    # Clone original image to draw on
    annotated_img = img.copy()

    # Detect faces
    try:
        boxes, _ = mtcnn.detect(img)
    except Exception:
        boxes = None

    face_count = len(boxes) if boxes is not None else 0

    faces = []
    face_thumbs = []

    if boxes is not None:
        draw = ImageDraw.Draw(annotated_img)
        for box in boxes:
            box_int = [int(b) for b in box]
            draw.rectangle(box_int, outline="#ef4444", width=3)
            try:
                face_crop = img.crop(box_int)
                faces.append(face_crop)
                face_thumbs.append(face_crop.resize((224, 224)))
            except Exception:
                pass

    # If no faces detected, fallback to analyzing full image
    if not faces:
        faces = [img]
        face_thumbs = [img.resize((224, 224))]

    fake_probs = []
    real_probs = []

    for face in faces:
        inputs = processor(images=face, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            # Label map: 0 = Fake, 1 = Real
            fake_prob = probs[0].item()
            real_prob = probs[1].item()
            fake_probs.append(fake_prob)
            real_probs.append(real_prob)

    avg_fake_prob = float(np.mean(fake_probs))
    avg_real_prob = float(np.mean(real_probs))

    # Apply sensitivity threshold
    is_fake = avg_fake_prob >= fake_threshold
    verdict = "Fake" if is_fake else "Real"
    primary_conf = avg_fake_prob if is_fake else avg_real_prob

    # Generate explainable AI visualizers & quality metrics
    ela_img = generate_ela_image(img)
    fft_img = generate_fft_spectrum(img)
    quality_metrics = analyze_image_quality_and_noise(img)

    raw_details = {
        "fake_probability": avg_fake_prob,
        "real_probability": avg_real_prob,
        "primary_confidence": primary_conf,
        "face_count": face_count,
        "threshold_used": fake_threshold,
        "quality_metrics": quality_metrics
    }

    return (
        verdict,
        f"{primary_conf:.2f}",
        face_count,
        annotated_img,
        face_thumbs,
        ela_img,
        fft_img,
        raw_details
    )
