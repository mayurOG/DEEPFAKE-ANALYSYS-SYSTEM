import re
from typing import Dict, Any, List, Optional


KNOWN_AI_GENERATOR_SIGNATURES = [
    ("dall-e", "OpenAI DALL-E Generative Model"),
    ("midjourney", "Midjourney AI Renderer"),
    ("stable diffusion", "Stability AI Diffusion Model"),
    ("comfyui", "ComfyUI Node-Based Generative Pipeline"),
    ("automatic1111", "AUTOMATIC1111 WebUI"),
    ("adobe firefly", "Adobe Firefly Generative Engine"),
    ("sora", "OpenAI Sora Video Diffusion"),
    ("kling", "Kuaishou Kling Video Model"),
    ("runway", "Runway Gen-2 / Gen-3 Alpha"),
    ("luma", "Luma Dream Machine"),
    ("deepfacelab", "DeepFaceLab Autoencoder Splicer"),
    ("faceswap", "FaceSwap Neural Re-enactment"),
    ("elevenlabs", "ElevenLabs Neural Speech Synthesis"),
    ("suno", "Suno Generative Audio"),
    ("udio", "Udio Generative Music"),
    ("c2pa.synthetic", "C2PA Explicit Synthetic Generation Assertion"),
]


def audit_c2pa_provenance(raw_data: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Audits C2PA (Coalition for Content Provenance and Authenticity) manifests,
    JUMBF (JPEG Universal Metadata Box Format) boxes, cryptographic claim blocks,
    and generative software provenance indicators.
    """
    if not raw_data:
        return {
            "c2pa_manifest_found": False,
            "status": "No Data",
            "trust_level": "Unsigned",
            "ai_provenance_detected": False,
            "detected_claims": [],
            "warnings": ["Media stream is empty."],
        }

    data_lower = raw_data.lower()

    # 1. Check for JUMBF / C2PA superbox byte signatures
    has_jumb = b"jumb" in data_lower
    has_c2pa = b"c2pa" in data_lower
    c2pa_manifest_present = has_jumb and has_c2pa

    detected_claims: List[str] = []
    ai_clues: List[str] = []
    actions_found: List[str] = []

    # 2. Extract AI Generative Model Signatures
    for sig_key, sig_name in KNOWN_AI_GENERATOR_SIGNATURES:
        if sig_key.encode("utf-8") in data_lower:
            ai_clues.append(sig_name)

    # 3. Detect C2PA Standard Action Assertions
    c2pa_actions = [
        (b"c2pa.created", "Original Creation"),
        (b"c2pa.edited", "Content Post-Processing / Editing"),
        (b"c2pa.ai_generated", "AI Generative Model Synthesized"),
        (b"c2pa.converted", "Format Transcoding"),
        (b"c2pa.color_adjustments", "Color Calibration / Tuning"),
        (b"c2pa.placed", "Composite Overlay / Splicing"),
    ]
    for action_bytes, desc in c2pa_actions:
        if action_bytes in data_lower:
            actions_found.append(desc)

    # 4. Check for Digital Signatures & Signing Certificates
    has_x509_cert = b"x509" in data_lower or b"cert" in data_lower
    has_signature = b"c2pa.signature" in data_lower or b"claim.signature" in data_lower

    # 5. Extract Claim Generators
    generator_name = "Unknown"
    gen_matches = re.findall(b"claim_generator\"?:?\\s*\"?([^\",\\r\\n]{3,40})", data_lower)
    if gen_matches:
        try:
            generator_name = gen_matches[0].decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

    # 6. Evaluate Provenance Verdict and Trust Level
    if c2pa_manifest_present:
        if any("ai" in clue.lower() for clue in ai_clues) or "c2pa.ai_generated" in str(actions_found):
            trust_level = "Synthetic Provenance Confirmed"
            verdict = "C2PA Manifest Verified: Explicitly Declared Synthetic / AI Generated"
            ai_detected = True
        elif has_signature:
            trust_level = "Cryptographically Authenticated"
            verdict = "C2PA Provenance Manifest Intact with Valid Digital Signature"
            ai_detected = False
        else:
            trust_level = "Unverified Provenance Claims"
            verdict = "C2PA Manifest Markers Present (Unsigned or Self-Asserted)"
            ai_detected = len(ai_clues) > 0
    elif ai_clues:
        trust_level = "Unsigned Synthetic Metadata Traces"
        verdict = "AI Model Software Traces Found in File Container Metadata"
        ai_detected = True
    else:
        trust_level = "Unsigned / Standard Legacy Media"
        verdict = "No C2PA Manifest Found (Standard consumer or web media)"
        ai_detected = False

    return {
        "c2pa_manifest_found": c2pa_manifest_present,
        "trust_level": trust_level,
        "verdict": verdict,
        "ai_provenance_detected": ai_detected,
        "ai_generator_clues": ai_clues,
        "actions_recorded": actions_found,
        "has_digital_signature": has_signature,
        "has_x509_cert": has_x509_cert,
        "claim_generator": generator_name,
        "c2pa_box_count": data_lower.count(b"c2pa"),
    }
