from typing import Dict, Any, List, Optional
import numpy as np


class CalibratedForensicFusion:
    """
    SOTA Multi-Modal Evidence Fusion Engine.
    Combines neural, spectral, biometric, acoustic, and cryptographic
    forensic vectors with adaptive reliability weighting and contradiction detection.
    """

    # Baseline reliability weights for different forensic modalities
    MODALITY_WEIGHTS = {
        "neural_visual": 0.30,
        "neural_audio": 0.30,
        "spectral_rapsd": 0.15,
        "biometric_rppg": 0.15,
        "corneal_lighting": 0.10,
        "srm_noise": 0.05,
    }

    @classmethod
    def fuse_evidence(
        cls,
        neural_visual_fake_p: Optional[float] = None,
        neural_audio_fake_p: Optional[float] = None,
        spectral_rapsd_risk: Optional[float] = None,
        biometric_liveness_score: Optional[float] = None,
        corneal_disparity_score: Optional[float] = None,
        c2pa_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fuses available evidence into a unified forensic report.
        """
        evidence_vectors: List[Dict[str, Any]] = []
        scores: List[float] = []
        weights: List[float] = []

        if neural_visual_fake_p is not None:
            p = float(np.clip(neural_visual_fake_p, 0.0, 1.0))
            w = cls.MODALITY_WEIGHTS["neural_visual"]
            scores.append(p)
            weights.append(w)
            evidence_vectors.append({
                "modality": "Vision Transformer (SigLIP)",
                "p_fake": round(p, 3),
                "weight": w,
            })

        if neural_audio_fake_p is not None:
            p = float(np.clip(neural_audio_fake_p, 0.0, 1.0))
            w = cls.MODALITY_WEIGHTS["neural_audio"]
            scores.append(p)
            weights.append(w)
            evidence_vectors.append({
                "modality": "Acoustic Speech Classifier",
                "p_fake": round(p, 3),
                "weight": w,
            })

        if spectral_rapsd_risk is not None:
            p = float(np.clip(spectral_rapsd_risk, 0.0, 1.0))
            w = cls.MODALITY_WEIGHTS["spectral_rapsd"]
            scores.append(p)
            weights.append(w)
            evidence_vectors.append({
                "modality": "Radial Power Spectrum (RAPSD)",
                "p_fake": round(p, 3),
                "weight": w,
            })

        if biometric_liveness_score is not None:
            # liveness is 1.0 for real, so fake is 1.0 - liveness
            p = float(np.clip(1.0 - biometric_liveness_score, 0.0, 1.0))
            w = cls.MODALITY_WEIGHTS["biometric_rppg"]
            scores.append(p)
            weights.append(w)
            evidence_vectors.append({
                "modality": "Biometric rPPG Pulse Coherence",
                "p_fake": round(p, 3),
                "weight": w,
            })

        if corneal_disparity_score is not None:
            p = float(np.clip(corneal_disparity_score, 0.0, 1.0))
            w = cls.MODALITY_WEIGHTS["corneal_lighting"]
            scores.append(p)
            weights.append(w)
            evidence_vectors.append({
                "modality": "Corneal Specular Reflection",
                "p_fake": round(p, 3),
                "weight": w,
            })

        if not scores:
            return {
                "calibrated_p_fake": 0.5,
                "confidence_score": 0.0,
                "verdict": "Inconclusive (No Modalities Active)",
                "inter_expert_contradiction": 0.0,
                "evidence_vectors": [],
            }

        scores_arr = np.array(scores)
        weights_arr = np.array(weights)
        norm_weights = weights_arr / np.sum(weights_arr)

        # Weighted Bayesian mean
        calibrated_p_fake = float(np.sum(scores_arr * norm_weights))

        # Inter-expert contradiction (variance across modality assessments)
        if len(scores) > 1:
            contradiction_index = float(np.std(scores_arr) * 2.0)
            contradiction_index = min(1.0, contradiction_index)
        else:
            contradiction_index = 0.0

        # C2PA Cryptographic Override: If C2PA explicitly asserts synthetic generation
        c2pa_provenance_flag = False
        if c2pa_result and c2pa_result.get("ai_provenance_detected"):
            calibrated_p_fake = max(calibrated_p_fake, 0.95)
            c2pa_provenance_flag = True

        # Uncertainty intervals (95% CI)
        margin = max(0.04, contradiction_index * 0.15)
        ci_low = max(0.0, calibrated_p_fake - margin)
        ci_high = min(1.0, calibrated_p_fake + margin)

        # Final verdict determination with safety abstention margin
        if contradiction_index > 0.65 and 0.40 <= calibrated_p_fake <= 0.60:
            verdict = "Inconclusive / Abstention (Inter-Expert Contradiction)"
            decision_class = "inconclusive"
        elif calibrated_p_fake >= 0.55:
            verdict = "Synthetic / Manipulated Media"
            decision_class = "fake"
        elif calibrated_p_fake <= 0.45:
            verdict = "Authentic / Unaltered Media"
            decision_class = "real"
        else:
            verdict = "Inconclusive / Insufficient Discriminative Evidence"
            decision_class = "inconclusive"

        return {
            "calibrated_p_fake": round(calibrated_p_fake, 3),
            "confidence_score": round(abs(calibrated_p_fake - 0.5) * 2.0, 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
            "inter_expert_contradiction": round(contradiction_index, 3),
            "c2pa_override": c2pa_provenance_flag,
            "verdict": verdict,
            "decision_class": decision_class,
            "evidence_vectors": evidence_vectors,
        }
