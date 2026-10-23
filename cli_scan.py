#!/usr/bin/env python3
import sys
import os
import argparse

from detector.image_detector import detect_image
from detector.audio_detector import detect_audio
from detector.video_detector import detect_video
from detector.metadata_detector import analyze_metadata
from detector.report_generator import generate_html_report, generate_json_report


def main():
    parser = argparse.ArgumentParser(
        description="🛡️ Deepfake Forensic Intelligence CLI Utility v2.0"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input media file (Image, Video, or Audio)"
    )
    parser.add_argument(
        "--output-dir", "-o", default="./forensic_reports",
        help="Directory to save generated forensic reports"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="Fake sensitivity threshold (0.1 to 0.9)"
    )

    args = parser.parse_args()
    file_path = args.input

    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' does not exist.")
        sys.exit(1)

    ext = file_path.split(".")[-1].lower()
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.basename(file_path)

    print(f"🛡️ Analyzing: {filename}...")

    if ext in ["jpg", "jpeg", "png", "webp"]:
        verdict, conf_str, face_count, _, _, _, _, raw_details = detect_image(
            file_path, fake_threshold=args.threshold
        )
        metadata = analyze_metadata(file_path)
        media_type = "Image"
        metrics = {
            "Faces Detected": face_count,
            "Fake Probability": f"{raw_details['fake_probability'] * 100:.1f}%",
            "Threshold Used": f"{args.threshold:.2f}"
        }
        if raw_details.get("rapsd_metrics"):
            metrics["RAPSD Spectral Verdict"] = raw_details["rapsd_metrics"]["verdict"]
        if raw_details.get("c2pa_provenance"):
            metrics["C2PA Provenance Trust"] = raw_details["c2pa_provenance"]["trust_level"]
        if raw_details.get("corneal_metrics") and raw_details["corneal_metrics"].get("is_valid"):
            metrics["Corneal Lighting Verdict"] = raw_details["corneal_metrics"]["verdict"]

    elif ext in ["mp4", "avi", "mov"]:
        v_res = detect_video(file_path)
        verdict, total_analyzed, fake_count, summary = v_res[:4]
        conf_str = summary["fake_percentage"]
        metadata = {"video_fps": summary.get("video_fps", "30.0")}
        media_type = "Video"
        metrics = {
            "total_analyzed_frames": summary["total_analyzed_frames"],
            "real_frames": summary["real_frames"],
            "fake_frames": summary["fake_frames"],
            "fake_percentage": summary["fake_percentage"],
        }
        if summary.get("biometric_rppg") and summary["biometric_rppg"].get("is_valid"):
            metrics["rPPG Biological Liveness"] = summary["biometric_rppg"]["verdict"]
            metrics["Estimated Heart Rate"] = f"{summary['biometric_rppg']['estimated_bpm']} BPM"
        if summary.get("c2pa_provenance"):
            metrics["C2PA Container Trust"] = summary["c2pa_provenance"]["trust_level"]

    elif ext in ["wav", "mp3", "flac"]:
        verdict, conf_str, segment_preds, _, _ = detect_audio(file_path)
        metadata = None
        media_type = "Audio"
        metrics = {
            "Total Segments": len(segment_preds),
            "Average Confidence": conf_str
        }
    else:
        print(f"❌ Unsupported file extension '.{ext}'")
        sys.exit(1)

    print("\n==========================================")
    print(f"  VERDICT: {verdict.upper()}")
    print(f"  CONFIDENCE: {conf_str}")
    for k, v in metrics.items():
        if k not in ["total_analyzed_frames", "real_frames", "fake_frames"]:
            print(f"  • {k}: {v}")
    print("==========================================\n")

    # Generate Reports
    html_out = generate_html_report(
        filename, media_type, verdict, conf_str, metrics, metadata
    )
    json_out = generate_json_report(
        filename, media_type, verdict, conf_str, metrics, metadata
    )

    html_path = os.path.join(args.output_dir, f"{filename}_report.html")
    json_path = os.path.join(args.output_dir, f"{filename}_report.json")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_out)

    print(f"✅ Reports successfully generated in '{args.output_dir}':")
    print(f"  📄 HTML: {html_path}")
    print(f"  ⚙️ JSON: {json_path}")


if __name__ == "__main__":
    main()
