import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from PIL import Image
import tempfile
import base64

import importlib
import detector.audio_detector
import utils.visualizers
importlib.reload(detector.audio_detector)
importlib.reload(utils.visualizers)

from detector.image_detector import detect_image
from detector.video_detector import detect_video
from detector.audio_detector import detect_audio, get_waveform_image
from detector.metadata_detector import analyze_metadata
from detector.report_generator import generate_html_report, generate_json_report
from utils.ui_components import (
    apply_custom_theme, render_header, render_verdict_card
)
from utils.visualizers import (
    generate_ela_image, generate_fft_spectrum,
    plot_rapsd_curve, plot_rppg_wave, plot_fusion_bar
)
from detector.spectral_rapsd import compute_rapsd
from detector.biometric_rppg import analyze_video_rppg, analyze_static_face_micro_vasculature
from detector.c2pa_provenance import audit_c2pa_provenance
from detector.corneal_lighting import analyze_corneal_lighting
from detector.calibrated_fusion import CalibratedForensicFusion

# Configure Streamlit page
st.set_page_config(
    page_title="🛡️ Deepfake Forensic Intelligence Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Glassmorphic Dark Theme
apply_custom_theme()

# Top Header
render_header()

# === SIDEBAR CONFIGURATION ===
st.sidebar.markdown("### ⚙️ Forensic Calibration")
fake_threshold = st.sidebar.slider(
    "🎯 Fake Sensitivity Threshold",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="Lower threshold increases detection sensitivity for deepfakes."
)

video_frame_skip = st.sidebar.slider(
    "🎞️ Video Frame Skip",
    min_value=5, max_value=60, value=15, step=5,
    help="Number of frames to skip during video inspection."
)

audio_chunk_sec = st.sidebar.slider(
    "🎵 Audio Chunk Duration (sec)",
    min_value=1, max_value=5, value=3, step=1,
    help="Duration of audio segments for time-series scanning."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Forensic Models & Engines Active")
st.sidebar.info("""
- **Vision**: `SiglipForImageClassification`
- **Faces**: `MTCNN` (FaceNet-PyTorch)
- **Audio**: `AutoModelForAudioClassification`
- **SOTA Fourier**: `Azimuthal Radial RAPSD (1/f^α)`
- **SOTA Biometrics**: `Remote rPPG Cardiac Coherence`
- **SOTA Optics**: `Bi-Ocular Corneal Specular Geometry`
- **SOTA Provenance**: `C2PA Content Credentials Auditor`
- **SOTA Decision**: `Calibrated Multi-Modal Fusion (95% CI)`
- **Noise Forensics**: `SRM Filter & Laplacian Blur`
""")

# === MAIN TABS NAVIGATION ===
tab_single, tab_cam, tab_batch, tab_compare, tab_lab, tab_report = st.tabs([
    "🎯 Single File Scanner",
    "📷 Live Webcam Scanner",
    "⚡ Batch Forensic Suite",
    "🆚 Dual File Comparison",
    "🔬 Deep Forensic Lab",
    "📄 Forensic Report Export"
])

# Store analysis results in session state for cross-tab availability
if "latest_analysis" not in st.session_state:
    st.session_state["latest_analysis"] = None

# ==========================================
# TAB 1: SINGLE FILE FORENSIC SCANNER
# ==========================================
with tab_single:
    st.markdown(
        '<div class="section-title">📁 Select Media Type for Forensic Inspection</div>',
        unsafe_allow_html=True
    )
    option = st.radio(
        "Media Category:",
        ["Image", "Video (Multimodal)", "Audio"],
        horizontal=True
    )

    # ------------------ IMAGE SCANNER ------------------
    if option == "Image":
        uploaded_img = st.file_uploader(
            "Upload Image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_img"
        )

        if uploaded_img:
            st.markdown("---")
            with st.spinner("🔬 Running Neural Deepfake & Spectral Forensic Pipeline..."):
                img_bytes = uploaded_img.read()
                uploaded_img.seek(0)

                # Image Deepfake Model Detection
                res = detect_image(uploaded_img, fake_threshold=fake_threshold)
                verdict, conf_str, face_count, annotated_img, face_thumbs = res[:5]
                ela_img, fft_img, raw_details = res[5:]

                # EXIF Metadata Detection
                uploaded_img.seek(0)
                metadata = analyze_metadata(uploaded_img)
                conf_val = float(conf_str)

                # Save to session state
                q_info = raw_details.get("quality_metrics", {})
                st.session_state["latest_analysis"] = {
                    "filename": uploaded_img.name,
                    "media_type": "Image",
                    "verdict": verdict,
                    "confidence": f"{conf_val * 100:.1f}%",
                    "summary_metrics": {
                        "Faces Detected": face_count,
                        "Fake Probability": f"{raw_details['fake_probability'] * 100:.1f}%",
                        "Real Probability": f"{raw_details['real_probability'] * 100:.1f}%",
                        "Laplacian Blur Score": q_info.get("laplacian_blur_score", "N/A"),
                        "SRM Noise Energy": q_info.get("srm_noise_energy", "N/A"),
                        "Sensitivity Threshold": f"{fake_threshold:.2f}"
                    },
                    "metadata": metadata
                }

            # Render Verdict Card
            fake_p_str = f"{raw_details['fake_probability'] * 100:.1f}%"
            render_verdict_card(
                verdict, conf_val,
                details=f"Analyzed {face_count} face(s). Fake prob: {fake_p_str}"
            )

            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Verdict Status", verdict,
                delta="Manipulated" if verdict == "Fake" else "Clean"
            )
            col2.metric("Primary Confidence", f"{conf_val * 100:.1f}%")
            col3.metric("Faces Detected", face_count)
            col4.metric(
                "EXIF Status",
                "Present" if metadata["has_exif"] else "Missing / Stripped"
            )

            # Image Visuals Row
            st.markdown("### 🖼️ Forensic Visual Analysis")
            img_col1, img_col2, img_col3 = st.columns(3)

            with img_col1:
                st.markdown("##### 📷 Annotated Bounding Boxes")
                st.image(annotated_img, use_container_width=True)

            with img_col2:
                st.markdown("##### 🔍 Error Level Analysis (ELA)")
                st.image(
                    ela_img, use_container_width=True,
                    caption="Highlights JPEG re-compression variance"
                )

            with img_col3:
                st.markdown("##### ⚡ 2D FFT Spatial Spectrum")
                st.image(
                    fft_img, use_container_width=True,
                    caption="High-frequency periodic grid artifacts"
                )

            # Quality & Noise Metrics Expander
            q_metrics = raw_details.get("quality_metrics", {})
            if q_metrics:
                st.markdown("### 📊 Image Quality & SRM Noise Residual Metrics")
                qm_c1, qm_c2, qm_c3 = st.columns(3)
                qm_c1.metric(
                    "Laplacian Blur Score", q_metrics.get("laplacian_blur_score")
                )
                qm_c2.metric(
                    "Skin Chroma Variance", q_metrics.get("chroma_variance")
                )
                qm_c3.metric(
                    "SRM Noise Residual", q_metrics.get("srm_noise_energy")
                )

            # Crop thumbnails if faces detected
            if face_thumbs:
                st.markdown("### 👤 Isolated Facial Crops")
                cols = st.columns(min(len(face_thumbs), 4))
                for idx, thumb in enumerate(face_thumbs):
                    with cols[idx % 4]:
                        st.image(
                            thumb, caption=f"Face Region #{idx + 1}",
                            use_container_width=True
                        )

            # Metadata & Forensic Warnings Expander
            with st.expander("📄 View EXIF & Header Forensic Inspection Details"):
                if metadata["warnings"]:
                    st.error("⚠️ **Forensic Warnings Detected:**")
                    for warn in metadata["warnings"]:
                        st.markdown(f"- `{warn}`")

                meta_df = pd.DataFrame([
                    {"Property": k, "Value": str(v)}
                    for k, v in metadata.items()
                    if k not in ["exif_data", "warnings"]
                ])
                st.dataframe(meta_df, use_container_width=True)

            # SOTA Forensic Intelligence Suite (RAPSD, Corneal Lighting, C2PA, Calibrated Fusion)
            with st.expander("🔬 SOTA Forensic Intelligence (RAPSD, Corneal Lighting, C2PA, Calibrated Fusion)", expanded=True):
                # 1. Calibrated Multi-Modal Fusion
                fusion_data = raw_details.get("calibrated_fusion")
                if fusion_data:
                    st.markdown("#### ⚡ Calibrated Evidence Fusion & Uncertainty Bounds")
                    f_col1, f_col2, f_col3 = st.columns(3)
                    f_col1.metric("Calibrated P(Fake)", f"{fusion_data['calibrated_p_fake'] * 100:.1f}%")
                    f_col2.metric("95% Confidence Interval", f"[{fusion_data['ci_low']:.2f}, {fusion_data['ci_high']:.2f}]")
                    f_col3.metric("Inter-Expert Contradiction", f"{fusion_data['inter_expert_contradiction']:.2f}")

                    fusion_bar_fig = plot_fusion_bar(fusion_data)
                    if fusion_bar_fig:
                        st.plotly_chart(fusion_bar_fig, use_container_width=True)

                sota_col1, sota_col2 = st.columns(2)

                # 2. RAPSD Spectral Profile
                with sota_col1:
                    rapsd = raw_details.get("rapsd_metrics")
                    if rapsd:
                        st.markdown("#### 📐 Radially Averaged Power Spectrum (RAPSD)")
                        rapsd_fig = plot_rapsd_curve(rapsd)
                        st.plotly_chart(rapsd_fig, use_container_width=True)
                        st.caption(
                            f"**Spectral Slope (α):** `{rapsd['spectral_slope_alpha']}` | "
                            f"**High-Freq Anomaly:** `{rapsd['high_freq_anomaly_energy']}` | "
                            f"**Verdict:** `{rapsd['verdict']}`"
                        )

                # 3. Corneal Specular Lighting
                with sota_col2:
                    corneal = raw_details.get("corneal_metrics")
                    if corneal and corneal.get("is_valid"):
                        st.markdown("#### 👁️ Corneal Specular Reflection Consistency")
                        eye_c1, eye_c2 = st.columns(2)
                        with eye_c1:
                            st.image(corneal["left_eye_annotated"], caption="Left Eye Specular Highlight", use_container_width=True)
                        with eye_c2:
                            st.image(corneal["right_eye_annotated"], caption="Right Eye Specular Highlight", use_container_width=True)
                        st.caption(
                            f"**Lighting Consistency:** `{corneal['consistency_score'] * 100:.1f}%` | "
                            f"**Vector Disparity:** `{corneal['angle_disparity_deg']}°` | "
                            f"**Verdict:** `{corneal['verdict']}`"
                        )

                # 4. C2PA Content Credentials
                c2pa = raw_details.get("c2pa_provenance")
                if c2pa:
                    st.markdown("#### 📜 C2PA Content Credentials & Provenance Manifest")
                    c_badge = "🟢 SIGNED / PROVENANCE PRESENT" if c2pa["c2pa_manifest_found"] else "⚪ UNSIGNED STANDARD MEDIA"
                    if c2pa.get("ai_provenance_detected"):
                        c_badge = "🔴 SYNTHETIC AI PROVENANCE DETECTED"
                    st.markdown(f"**Provenance Trust Status:** `{c_badge}` — *{c2pa['verdict']}*")
                    if c2pa.get("ai_generator_clues"):
                        st.warning(f"🤖 **Generative Model Signatures Detected:** {', '.join(c2pa['ai_generator_clues'])}")

    # ------------------ VIDEO SCANNER ------------------
    elif option == "Video (Multimodal)":
        uploaded_vid = st.file_uploader(
            "Upload Video (MP4 / AVI / MOV)",
            type=["mp4", "avi", "mov"],
            key="single_vid"
        )

        if uploaded_vid:
            st.markdown("---")
            video_bytes = uploaded_vid.read()
            uploaded_vid.seek(0)

            # Video Preview
            b64_video = base64.b64encode(video_bytes).decode()
            st.markdown("### 🎞️ Uploaded Video Preview")
            v_html = f"""
                <video controls style='width: 100%; max-width: 600px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); margin-bottom: 20px;'>
                    <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                </video>
            """
            st.markdown(v_html, unsafe_allow_html=True)

            with st.spinner("🔍 Running Frame-by-Frame & Audio Inspection..."):
                vid_res = detect_video(uploaded_vid, frame_skip=video_frame_skip)
                verdict, total_analyzed, fake_count, summary = vid_res[:4]
                fake_frames, annotated_frames, timeline_data, audio_analysis = vid_res[4:]

                fake_pct_val = (
                    (fake_count / total_analyzed) if total_analyzed > 0 else 0.0
                )
                conf_val = fake_pct_val if verdict == "Fake" else (1.0 - fake_pct_val)

                st.session_state["latest_analysis"] = {
                    "filename": uploaded_vid.name,
                    "media_type": "Video",
                    "verdict": verdict,
                    "confidence": f"{conf_val * 100:.1f}%",
                    "summary_metrics": summary,
                    "metadata": {"video_fps": summary.get("video_fps", "30.0")}
                }

            # Verdict Display
            render_verdict_card(
                verdict, conf_val,
                details=f"Analyzed {total_analyzed} frames. Flagged {fake_count} fake frames ({summary['fake_percentage']})."
            )

            # Interactive Plotly Timeline Chart
            if timeline_data:
                st.markdown("### 📈 Temporal Deepfake Probability Timeline")
                df_timeline = pd.DataFrame(timeline_data)

                fig = px.line(
                    df_timeline, x="timestamp", y="fake_probability",
                    labels={
                        "timestamp": "Time (Seconds)",
                        "fake_probability": "Fake Probability Score"
                    },
                    title="Frame-by-Frame Deepfake Detection Profile",
                    markers=True
                )
                fig.add_hline(
                    y=fake_threshold, line_dash="dash", line_color="red",
                    annotation_text=f"Threshold ({fake_threshold})"
                )
                fig.update_traces(line_color="#38bdf8", marker_color="#818cf8")
                fig.update_layout(
                    paper_bgcolor="rgba(15,23,42,0.8)",
                    plot_bgcolor="rgba(15,23,42,0.8)",
                    font_color="#e2e8f0"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Multimodal Audio Analysis Section
            if audio_analysis:
                st.markdown("### 🎙️ Extracted Multimodal Audio Analysis")
                a_verdict, a_conf, a_segments, a_mel, a_ts = audio_analysis
                a_badge = (
                    "🔴 FAKE AUDIO" if a_verdict == "Fake" else "🟢 REAL AUDIO"
                )
                st.markdown(
                    f"**Audio Track Status:** {a_badge} (Conf: `{a_conf}`)"
                )
                st.image(
                    a_mel, use_container_width=True,
                    caption="Extracted Audio Mel-Spectrogram"
                )

            # Display Detected Fake Frames
            if fake_frames:
                st.markdown("### 🖼️ Flagged Manipulated Frames")
                cols = st.columns(4)
                for idx, f_img in enumerate(fake_frames[:12]):
                    with cols[idx % 4]:
                        st.image(
                            f_img, caption=f"Fake Frame #{idx + 1}",
                            use_container_width=True
                        )

            # SOTA Biometric (rPPG Pulse), C2PA, and Calibrated Fusion Expander
            rppg_res = summary.get("biometric_rppg")
            c2pa_vid = summary.get("c2pa_provenance")
            vid_fusion = summary.get("calibrated_fusion")

            with st.expander("🔬 SOTA Biometric (rPPG Pulse) & Provenance Intelligence", expanded=True):
                if vid_fusion:
                    st.markdown("#### ⚡ Calibrated Forensic Consensus & Uncertainty Bounds")
                    vf_c1, vf_c2, vf_c3 = st.columns(3)
                    vf_c1.metric("Calibrated P(Fake)", f"{vid_fusion['calibrated_p_fake'] * 100:.1f}%")
                    vf_c2.metric("95% Confidence Interval", f"[{vid_fusion['ci_low']:.2f}, {vid_fusion['ci_high']:.2f}]")
                    vf_c3.metric("Inter-Modal Contradiction", f"{vid_fusion['inter_expert_contradiction']:.2f}")

                    v_fusion_fig = plot_fusion_bar(vid_fusion)
                    if v_fusion_fig:
                        st.plotly_chart(v_fusion_fig, use_container_width=True)

                if rppg_res and rppg_res.get("is_valid"):
                    st.markdown("#### 💓 Remote Photoplethysmography (rPPG) Hemodynamic Cardiac Coherence")
                    rppg_fig = plot_rppg_wave(rppg_res)
                    st.plotly_chart(rppg_fig, use_container_width=True)
                    st.caption(
                        f"**Biological Liveness Confidence:** `{rppg_res['liveness_confidence'] * 100:.1f}%` | "
                        f"**Cross-Cheek Phase Coherence:** `{rppg_res['coherence_score']}` | "
                        f"**Estimated Heart Rate:** `{rppg_res['estimated_bpm']} BPM` | "
                        f"**Verdict:** `{rppg_res['verdict']}`"
                    )

                if c2pa_vid:
                    st.markdown("#### 📜 C2PA Video Container Credentials")
                    v_c2pa_badge = "🟢 VERIFIED C2PA CONTAINER" if c2pa_vid["c2pa_manifest_found"] else "⚪ STANDARD CONTAINER"
                    if c2pa_vid.get("ai_provenance_detected"):
                        v_c2pa_badge = "🔴 SYNTHETIC PROVENANCE ASSERTIONS"
                    st.markdown(f"**Video Provenance Status:** `{v_c2pa_badge}` — *{c2pa_vid['verdict']}*")

    # ------------------ AUDIO SCANNER ------------------
    elif option == "Audio":
        uploaded_aud = st.file_uploader(
            "Upload Audio (WAV / MP3 / FLAC)",
            type=["wav", "mp3", "flac"],
            key="single_aud"
        )

        if uploaded_aud:
            st.markdown("---")
            aud_bytes = uploaded_aud.read()
            uploaded_aud.seek(0)

            orig_name = uploaded_aud.name
            _, ext = os.path.splitext(orig_name)
            ext = ext.lower() if ext else ".wav"
            mime_type = "audio/mp3" if ext == ".mp3" else ("audio/flac" if ext == ".flac" else "audio/wav")
            st.audio(aud_bytes, format=mime_type)

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(aud_bytes)
                tmp_path = tmp.name

            with st.spinner("🎵 Running Acoustic Voice Classifier..."):
                aud_res = detect_audio(tmp_path, chunk_seconds=audio_chunk_sec)
                verdict, conf_str, segment_preds, mel_img, time_series_data = aud_res
                waveform_img = get_waveform_image(tmp_path)
                conf_val = float(conf_str)

                st.session_state["latest_analysis"] = {
                    "filename": uploaded_aud.name,
                    "media_type": "Audio",
                    "verdict": verdict,
                    "confidence": f"{conf_val * 100:.1f}%",
                    "summary_metrics": {
                        "Total Segments": len(segment_preds),
                        "Chunk Duration": f"{audio_chunk_sec}s",
                        "Average Confidence": f"{conf_val:.2f}"
                    },
                    "metadata": None
                }

            # Cleanup
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            # Verdict Display
            render_verdict_card(
                verdict, conf_val,
                details=f"Analyzed {len(segment_preds)} segments. Voice verdict: {verdict}"
            )

            # Plots Row
            st.markdown("### 🎵 Acoustic Signal Analysis")
            aud_col1, aud_col2 = st.columns(2)
            with aud_col1:
                st.markdown("##### 🌊 Amplitude Waveform")
                st.image(waveform_img, use_container_width=True)
            with aud_col2:
                st.markdown("##### 🌌 Mel-Frequency Spectrogram")
                st.image(mel_img, use_container_width=True)

            # Audio Time Series Plotly Chart
            if time_series_data:
                st.markdown("### 📊 Time-Series Audio Confidence Graph")
                df_aud = pd.DataFrame(time_series_data)
                fig_aud = px.line(
                    df_aud, x="timestamp", y="fake_probability",
                    labels={
                        "timestamp": "Time (Seconds)",
                        "fake_probability": "Synthetic Voice Score"
                    },
                    title="Segmented Acoustic Fake Probability Progression",
                    markers=True
                )
                fig_aud.add_hline(
                    y=fake_threshold, line_dash="dash", line_color="red"
                )
                fig_aud.update_traces(
                    line_color="#c084fc", marker_color="#818cf8"
                )
                fig_aud.update_layout(
                    paper_bgcolor="rgba(15,23,42,0.8)",
                    plot_bgcolor="rgba(15,23,42,0.8)",
                    font_color="#e2e8f0"
                )
                st.plotly_chart(fig_aud, use_container_width=True)

            # Segment Table
            st.markdown("### ⏱️ Timestamped Segment Breakdown")
            seg_df = pd.DataFrame(segment_preds)
            st.dataframe(seg_df, use_container_width=True)

# ==========================================
# TAB 2: LIVE WEBCAM SCANNER
# ==========================================
with tab_cam:
    st.markdown(
        '<div class="section-title">📷 Real-Time Live Webcam Inspector</div>',
        unsafe_allow_html=True
    )
    st.markdown("Capture a camera snapshot to run real-time neural face detection and spectral inspection.")

    cam_picture = st.camera_input("Take a snapshot for instant forensic scan")

    if cam_picture:
        st.markdown("---")
        with st.spinner("🔍 Analyzing live camera frame..."):
            cam_res = detect_image(cam_picture, fake_threshold=fake_threshold)
            verdict, conf_str, face_count, annotated_img, face_thumbs = cam_res[:5]
            ela_img, fft_img, raw_details = cam_res[5:]
            conf_val = float(conf_str)

        render_verdict_card(
            verdict, conf_val,
            details=f"Live frame scanned. {face_count} face(s) evaluated."
        )

        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            st.markdown("##### 📷 Live Annotated Frame")
            st.image(annotated_img, use_container_width=True)
        with c_col2:
            st.markdown("##### 🔍 Live ELA Compression Map")
            st.image(ela_img, use_container_width=True)
        with c_col3:
            st.markdown("##### ⚡ Live 2D FFT Spectrum")
            st.image(fft_img, use_container_width=True)

# ==========================================
# TAB 3: BATCH FORENSIC SUITE
# ==========================================
with tab_batch:
    st.markdown(
        '<div class="section-title">⚡ Batch Multi-File Forensic Scanning</div>',
        unsafe_allow_html=True
    )
    st.markdown("Upload multiple images or audio clips for automated queue processing.")

    batch_files = st.file_uploader(
        "Upload multiple files (Images / Audios)",
        type=["jpg", "jpeg", "png", "wav", "mp3"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_files:
        if st.button("🚀 Start Batch Analysis", type="primary"):
            results = []
            progress_bar = st.progress(0)

            for idx, b_file in enumerate(batch_files):
                fname = b_file.name
                ext = fname.split(".")[-1].lower()

                if ext in ["jpg", "jpeg", "png"]:
                    b_res = detect_image(b_file, fake_threshold=fake_threshold)
                    verdict, conf_str, face_count = b_res[:3]
                    raw_details = b_res[7]
                    results.append({
                        "Filename": fname,
                        "Type": "Image",
                        "Verdict": verdict,
                        "Confidence": conf_str,
                        "Fake Probability": f"{raw_details['fake_probability'] * 100:.1f}%",
                        "Faces Found": face_count
                    })
                elif ext in ["wav", "mp3", "flac"]:
                    b_bytes = b_file.read()
                    b_file.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                        tmp.write(b_bytes)
                        tmp_path = tmp.name
                    verdict, conf_str, segs, _, _ = detect_audio(tmp_path)
                    fake_p_calc = (
                        f"{float(conf_str) * 100:.1f}%" if verdict == "Fake"
                        else f"{(1.0 - float(conf_str)) * 100:.1f}%"
                    )
                    results.append({
                        "Filename": fname,
                        "Type": "Audio",
                        "Verdict": verdict,
                        "Confidence": conf_str,
                        "Fake Probability": fake_p_calc,
                        "Faces Found": "N/A"
                    })
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                progress_bar.progress((idx + 1) / len(batch_files))

            st.success(f"✅ Successfully processed {len(batch_files)} files!")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

# ==========================================
# TAB 4: DUAL FILE COMPARISON LABORATORY
# ==========================================
with tab_compare:
    st.markdown(
        '<div class="section-title">🆚 Side-by-Side Dual File Inspector</div>',
        unsafe_allow_html=True
    )
    st.markdown("Compare two images side-by-side to visually inspect ELA heatmaps and 2D FFT spectra.")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("#### 📁 File A (Reference)")
        file_a = st.file_uploader("Upload Image A", type=["jpg", "png", "jpeg"], key="file_a")
    with comp_col2:
        st.markdown("#### 📁 File B (Suspect)")
        file_b = st.file_uploader("Upload Image B", type=["jpg", "png", "jpeg"], key="file_b")

    if file_a and file_b:
        st.markdown("---")
        with st.spinner("🔬 Running dual forensic comparison..."):
            res_a = detect_image(file_a, fake_threshold=fake_threshold)
            v_a, c_a, f_a, ann_a, _, ela_a, fft_a, raw_a = res_a

            res_b = detect_image(file_b, fake_threshold=fake_threshold)
            v_b, c_b, f_b, ann_b, _, ela_b, fft_b, raw_b = res_b

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            render_verdict_card(v_a, float(c_a), details="File A Result")
            st.image(ann_a, caption="File A Annotated", use_container_width=True)
            st.image(ela_a, caption="File A ELA", use_container_width=True)
            st.image(fft_a, caption="File A 2D FFT Spectrum", use_container_width=True)
        with res_c2:
            render_verdict_card(v_b, float(c_b), details="File B Result")
            st.image(ann_b, caption="File B Annotated", use_container_width=True)
            st.image(ela_b, caption="File B ELA", use_container_width=True)
            st.image(fft_b, caption="File B 2D FFT Spectrum", use_container_width=True)

# ==========================================
# TAB 5: DEEP FORENSIC LABORATORY
# ==========================================
with tab_lab:
    st.markdown(
        '<div class="section-title">🔬 SOTA Deep Forensic Intelligence & Biometrics Laboratory</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Explore cutting-edge, state-of-the-art forensic methodologies addressing generative AI anomalies: "
        "Azimuthal Radial Fourier Decay, Hemodynamic rPPG Pulse Coherence, Corneal Specular Highlights, and C2PA Provenance."
    )

    lab_sub_rapsd, lab_sub_rppg, lab_sub_corneal, lab_sub_c2pa, lab_sub_ela = st.tabs([
        "📐 Fourier RAPSD & 1/f^α Law",
        "💓 Biometric rPPG Cardiac Coherence",
        "👁️ Corneal Specular Reflection",
        "📜 C2PA Provenance Manifest Auditor",
        "🔍 Interactive ELA & 2D FFT Lab"
    ])

    # ------------------ SUB-TAB 1: RAPSD & 1/f^α LAW ------------------
    with lab_sub_rapsd:
        st.markdown("### 📐 Radially Averaged Power Spectral Distribution (RAPSD)")
        st.caption(
            "Natural camera photographs strictly adhere to the physical 1/f^α power-law Fourier decay. "
            "Generative AI upsamplers (transposed convolutions, pixel-shufflers) induce distinct high-frequency periodic lattice spikes."
        )

        rapsd_upload = st.file_uploader("Upload Image for RAPSD Spectral Analysis", type=["jpg", "png", "jpeg"], key="rapsd_up")
        if rapsd_upload:
            img_r = Image.open(rapsd_upload)
            with st.spinner("🔬 Computing 2D FFT and azimuthal radial integration..."):
                r_metrics = compute_rapsd(img_r)
                fft_preview = generate_fft_spectrum(img_r)

            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                st.image(fft_preview, caption="2D FFT Magnitude Spectrum", use_container_width=True)
                st.markdown(f"**Diagnostic Verdict:** `{r_metrics['verdict']}`")
                st.metric("Spectral Decay Slope (α)", f"{r_metrics['spectral_slope_alpha']}")
                st.metric("High-Frequency Anomaly Energy", f"{r_metrics['high_freq_anomaly_energy']:.4f}")
                st.metric("Periodic Spike Roughness", f"{r_metrics['spike_roughness']:.4f}")

            with r_col2:
                fig_rapsd = plot_rapsd_curve(r_metrics)
                st.plotly_chart(fig_rapsd, use_container_width=True)

    # ------------------ SUB-TAB 2: rPPG BIOMETRICS ------------------
    with lab_sub_rppg:
        st.markdown("### 💓 Remote Photoplethysmography (rPPG) Hemodynamic Coherence")
        st.caption(
            "Natural human faces display microscopic capillary blood pulsations synchronized across forehead and cheeks. "
            "Deepfakes and neural avatars disrupt or desynchronize these physiological vascular phase rhythms."
        )

        rppg_vid = st.file_uploader("Upload Video Clip for Cardiac Pulse Extraction", type=["mp4", "mov", "avi"], key="rppg_up")
        if rppg_vid:
            with st.spinner("💓 Extracting facial vascular ROIs and running CHROM bandpass filtering..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t_vid:
                    t_vid.write(rppg_vid.read())
                    t_vid_path = t_vid.name

                cap_t = cv2.VideoCapture(t_vid_path)
                fps_t = cap_t.get(cv2.CAP_PROP_FPS) or 30.0
                frames_t = []
                while len(frames_t) < 90:
                    ret, fr = cap_t.read()
                    if not ret:
                        break
                    frames_t.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                cap_t.release()

                if os.path.exists(t_vid_path):
                    os.remove(t_vid_path)

                rppg_out = analyze_video_rppg(frames_t, fps=fps_t)

            if rppg_out.get("is_valid"):
                rppg_fig = plot_rppg_wave(rppg_out)
                st.plotly_chart(rppg_fig, use_container_width=True)

                rp_c1, rp_c2, rp_c3 = st.columns(3)
                rp_c1.metric("Estimated Heart Rate", f"{rppg_out['estimated_bpm']} BPM")
                rp_c2.metric("Cardiac Phase Coherence", f"{rppg_out['coherence_score'] * 100:.1f}%")
                rp_c3.metric("Biological Liveness Confidence", f"{rppg_out['liveness_confidence'] * 100:.1f}%")

                st.markdown(f"**Forensic Biometric Verdict:** `{rppg_out['verdict']}`")
                st.json(rppg_out["regions_correlation"])
            else:
                st.warning(rppg_out.get("error", "Failed to compute rPPG signal."))

    # ------------------ SUB-TAB 3: CORNEAL LIGHTING ------------------
    with lab_sub_corneal:
        st.markdown("### 👁️ Bi-Ocular Corneal Specular Reflection Consistency")
        st.caption(
            "Physical scene lighting must produce geometrically coherent reflection highlights relative to pupil centers in both eyes. "
            "Generative AI portraits often generate divergent specular highlights and incompatible corneal light angles."
        )

        eye_upload = st.file_uploader("Upload Portrait Photo for Corneal Highlight Analysis", type=["jpg", "png", "jpeg"], key="eye_up")
        if eye_upload:
            img_eye = Image.open(eye_upload)
            with st.spinner("👁️ Segmenting corneal highlights and calculating reflection vectors..."):
                c_out = analyze_corneal_lighting(img_eye)

            if c_out.get("is_valid"):
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    st.image(c_out["left_eye_annotated"], caption="Left Eye Specular Highlight Centroid", use_container_width=True)
                with e_col2:
                    st.image(c_out["right_eye_annotated"], caption="Right Eye Specular Highlight Centroid", use_container_width=True)

                ec_1, ec_2, ec_3 = st.columns(3)
                ec_1.metric("Lighting Consistency Score", f"{c_out['consistency_score'] * 100:.1f}%")
                ec_2.metric("Highlight Angular Disparity", f"{c_out['angle_disparity_deg']}°")
                ec_3.metric("Left vs Right Peak Intensity", f"{c_out['left_peak_intensity']} / {c_out['right_peak_intensity']}")

                st.markdown(f"**Geometric Optics Verdict:** `{c_out['verdict']}`")
            else:
                st.warning(c_out.get("error", "Could not localize eye regions."))

    # ------------------ SUB-TAB 4: C2PA PROVENANCE AUDITOR ------------------
    with lab_sub_c2pa:
        st.markdown("### 📜 C2PA Content Credentials & Cryptographic Provenance Auditor")
        st.caption(
            "Inspect Coalition for Content Provenance and Authenticity (C2PA) JUMBF boxes, "
            "cryptographic claims, digital certificates, and AI generator metadata signatures."
        )

        c2pa_file = st.file_uploader("Upload Any Media File (Image, Video, Audio)", key="c2pa_up")
        if c2pa_file:
            raw_c2pa_data = c2pa_file.read()
            c2pa_file.seek(0)
            with st.spinner("📜 Scanning byte superboxes for C2PA manifests and AI signatures..."):
                c2pa_rep = audit_c2pa_provenance(raw_c2pa_data, c2pa_file.name)

            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.markdown(f"**Trust Classification:** `{c2pa_rep['trust_level']}`")
                st.markdown(f"**Status Verdict:** *{c2pa_rep['verdict']}*")
                st.metric("C2PA Boxes Found", c2pa_rep["c2pa_box_count"])
                st.metric("Cryptographic Signature", "Present" if c2pa_rep["has_digital_signature"] else "None")

            with p_col2:
                if c2pa_rep["ai_generator_clues"]:
                    st.error(f"🤖 **AI Generator Footprints Detected:** {', '.join(c2pa_rep['ai_generator_clues'])}")
                else:
                    st.success("✅ No known generative model signatures found in metadata.")

                if c2pa_rep["actions_recorded"]:
                    st.markdown("**Recorded Provenance Lineage Actions:**")
                    for act in c2pa_rep["actions_recorded"]:
                        st.markdown(f"- `{act}`")
                else:
                    st.caption("No certified C2PA edit actions recorded in container.")

    # ------------------ SUB-TAB 5: CUSTOM ELA & FFT ------------------
    with lab_sub_ela:
        st.markdown("### 🔍 Custom Error Level Analysis (ELA) & 2D FFT Experimentation")
        st.caption("Fine-tune JPEG recompression quality and ELA scaling to isolate multi-stage compression tampering.")

        lab_img = st.file_uploader(
            "Upload an image for ELA/FFT parameter tuning",
            type=["jpg", "png", "jpeg"],
            key="lab_img"
        )
        if lab_img:
            pil_lab_img = Image.open(lab_img)
            quality_val = st.slider("JPEG Compression Quality (ELA)", 50, 98, 90)
            scale_val = st.slider("ELA Brightness Scale Factor", 5, 30, 15)

            ela_res = generate_ela_image(
                pil_lab_img, quality=quality_val, scale=scale_val
            )
            fft_res = generate_fft_spectrum(pil_lab_img)

            c1, c2, c3 = st.columns(3)
            c1.image(pil_lab_img, caption="Original Input Image", use_container_width=True)
            c2.image(ela_res, caption=f"Custom ELA (Q={quality_val}, S={scale_val})", use_container_width=True)
            c3.image(fft_res, caption="2D FFT Magnitude Spectrum", use_container_width=True)

# ==========================================
# TAB 6: FORENSIC REPORT GENERATOR
# ==========================================
with tab_report:
    st.markdown(
        '<div class="section-title">📄 Generate & Export Forensic Audit Reports</div>',
        unsafe_allow_html=True
    )

    analysis_data = st.session_state.get("latest_analysis")

    if not analysis_data:
        st.warning(
            "⚠️ No active analysis found. Please inspect a file in the "
            "**Single File Scanner** tab first!"
        )
    else:
        st.markdown(
            f"**Current Loaded Media**: `{analysis_data['filename']}` "
            f"({analysis_data['media_type']})"
        )
        st.markdown(
            f"**Verdict**: `{analysis_data['verdict']}` | "
            f"**Confidence**: `{analysis_data['confidence']}`"
        )

        st.markdown("---")
        rep_col1, rep_col2 = st.columns(2)

        with rep_col1:
            st.markdown("### 🌐 Standalone HTML Forensic Report")
            html_content = generate_html_report(
                filename=analysis_data["filename"],
                media_type=analysis_data["media_type"],
                verdict=analysis_data["verdict"],
                confidence=analysis_data["confidence"],
                summary_metrics=analysis_data["summary_metrics"],
                metadata=analysis_data.get("metadata")
            )

            st.download_button(
                label="📥 Download HTML Report",
                data=html_content,
                file_name=f"Forensic_Report_{analysis_data['filename']}.html",
                mime="text/html",
                type="primary"
            )

        with rep_col2:
            st.markdown("### ⚙️ Machine-Readable JSON Audit Log")
            json_content = generate_json_report(
                filename=analysis_data["filename"],
                media_type=analysis_data["media_type"],
                verdict=analysis_data["verdict"],
                confidence=analysis_data["confidence"],
                summary_metrics=analysis_data["summary_metrics"],
                metadata=analysis_data.get("metadata")
            )

            st.download_button(
                label="📥 Download JSON Log",
                data=json_content,
                file_name=f"Forensic_Report_{analysis_data['filename']}.json",
                mime="application/json"
            )

        st.markdown("---")
        st.markdown("##### Preview of HTML Report Output:")
        components.html(html_content, height=450, scrolling=True)
