import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional


def segment_corneal_highlight(eye_roi: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float, np.ndarray]:
    """
    Finds the centroid and peak intensity of corneal specular highlight in an eye ROI.
    Returns:
        (center_x, center_y), peak_intensity, annotated_eye
    """
    if eye_roi is None or eye_roi.size == 0:
        return None, 0.0, eye_roi

    gray = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2GRAY)
    # Highlight is typically in top 2% of brightness
    thresh_val = max(200, int(np.percentile(gray, 98)))
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Clean small noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = eye_roi.copy()

    if not contours:
        return None, float(np.max(gray)), annotated

    # Largest specular highlight contour
    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] > 0:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 255, 255), 1)  # Yellow circle
        cv2.drawMarker(annotated, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 4, 1)
        return (cx, cy), float(np.max(gray)), annotated

    return None, float(np.max(gray)), annotated


def analyze_corneal_lighting(face_input) -> Dict[str, Any]:
    """
    Evaluates bi-ocular corneal specular highlight consistency across eyes.
    SOTA Forensic Principle: Physical lights in the scene must reflect at
    geometrically coherent vectors relative to the pupil center in both eyes
    (O'Brien & Farid, IEEE T-IFS). GANs and Diffusion models frequently fail this constraint.
    """
    if isinstance(face_input, Image.Image):
        face_rgb = np.array(face_input.convert("RGB"))
    elif isinstance(face_input, np.ndarray):
        face_rgb = face_input if face_input.shape[2] == 3 else cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
    else:
        face_rgb = np.array(Image.open(face_input).convert("RGB"))

    h, w = face_rgb.shape[:2]

    # Anthropometric facial eye bounding coordinates
    # Left eye (viewer's left / subject's right)
    le_y1, le_y2 = int(0.24 * h), int(0.42 * h)
    le_x1, le_x2 = int(0.18 * w), int(0.44 * w)

    # Right eye (viewer's right / subject's left)
    re_y1, re_y2 = int(0.24 * h), int(0.42 * h)
    re_x1, re_x2 = int(0.56 * w), int(0.82 * w)

    left_eye_roi = face_rgb[le_y1:le_y2, le_x1:le_x2]
    right_eye_roi = face_rgb[re_y1:re_y2, re_x1:re_x2]

    if left_eye_roi.size == 0 or right_eye_roi.size == 0:
        return {
            "is_valid": False,
            "error": "Eye regions could not be localized.",
            "consistency_score": 0.5,
            "verdict": "Indeterminate",
        }

    left_center = (left_eye_roi.shape[1] / 2.0, left_eye_roi.shape[0] / 2.0)
    right_center = (right_eye_roi.shape[1] / 2.0, right_eye_roi.shape[0] / 2.0)

    l_hl, l_peak, l_annotated = segment_corneal_highlight(left_eye_roi)
    r_hl, r_peak, r_annotated = segment_corneal_highlight(right_eye_roi)

    has_both_highlights = (l_hl is not None) and (r_hl is not None)

    if has_both_highlights:
        # Displacement vectors from eye center to specular highlight
        vec_l = np.array([l_hl[0] - left_center[0], l_hl[1] - left_center[1]])
        vec_r = np.array([r_hl[0] - right_center[0], r_hl[1] - right_center[1]])

        norm_l = np.linalg.norm(vec_l) + 1e-6
        norm_r = np.linalg.norm(vec_r) + 1e-6

        # Directional cosine similarity of light source reflection vector
        cos_sim = float(np.dot(vec_l, vec_r) / (norm_l * norm_r))
        cos_sim = max(-1.0, min(1.0, cos_sim))

        # Angle disparity in degrees
        angle_diff_deg = float(np.degrees(np.arccos(cos_sim)))

        # Intensity ratio
        intensity_ratio = float(min(l_peak, r_peak) / max(l_peak, r_peak, 1e-6))

        # Overall reflection consistency score
        # In real faces, light angles differ by < 25 degrees
        angular_score = max(0.0, 1.0 - (angle_diff_deg / 60.0))
        consistency_score = round(float(0.7 * angular_score + 0.3 * intensity_ratio), 3)

        is_suspicious = consistency_score < 0.45 or angle_diff_deg > 45.0
        verdict = (
            "Divergent Corneal Lighting Geometry (Synthetic Artifact)"
            if is_suspicious
            else "Coherent Bi-Ocular Corneal Lighting"
        )
    else:
        angle_diff_deg = 0.0
        consistency_score = 0.5
        verdict = "Specular Highlights Absent / Weakly Lit"
        is_suspicious = False

    # Resize annotated eyes for UI display
    l_thumb = cv2.resize(l_annotated, (120, 80))
    r_thumb = cv2.resize(r_annotated, (120, 80))

    return {
        "is_valid": True,
        "has_both_highlights": has_both_highlights,
        "consistency_score": consistency_score,
        "angle_disparity_deg": round(angle_diff_deg, 1),
        "left_peak_intensity": round(l_peak, 1),
        "right_peak_intensity": round(r_peak, 1),
        "is_suspicious": is_suspicious,
        "verdict": verdict,
        "left_eye_annotated": Image.fromarray(l_thumb),
        "right_eye_annotated": Image.fromarray(r_thumb),
    }
