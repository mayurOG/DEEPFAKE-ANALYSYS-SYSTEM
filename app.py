import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from PIL import Image
import tempfile
import base64
import os

from detector.image_detector import detect_image
from detector.video_detector import detect_video
from detector.audio_detector import detect_audio, get_waveform_image
from detector.metadata_detector import analyze_metadata
from detector.report_generator import generate_html_report, generate_json_report
from utils.ui_components import (
    apply_custom_theme, render_header, render_verdict_card
)
from utils.visualizers import generate_ela_image, generate_fft_spectrum

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
- **Spectral XAI**: ELA & 2D FFT Frequency Spectrum
- **Noise Forensics**: SRM Residual Filter & Laplacian Blur
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

    # ------------------ AUDIO SCANNER ------------------
    elif option == "Audio":
        uploaded_aud = st.file_uploader(
            "Upload Audio (WAV / MP3 / FLAC)",
            type=["wav", "mp3", "flac"],
            key="single_aud"
        )

        if uploaded_aud:
            st.markdown("---")
            st.audio(uploaded_aud, format="audio/wav")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_aud.read())
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
                elif ext in ["wav", "mp3"]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                        tmp.write(b_file.read())
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
        '<div class="section-title">🔬 Explainable AI (XAI) Forensic Methods</div>',
        unsafe_allow_html=True
    )

    col_lab1, col_lab2 = st.columns(2)
    with col_lab1:
        st.markdown("""
        ### 🔍 Error Level Analysis (ELA)
        - **How it works**: Resaves an image at a known JPEG compression quality (e.g. 90%) and calculates the pixel delta relative to the original image.
        - **Why it spots deepfakes**: Manipulated face regions undergo multi-stage re-compression, exhibiting higher contrast under ELA enhancement.
        """)

    with col_lab2:
        st.markdown("""
        ### ⚡ 2D Fast Fourier Transform (FFT)
        - **How it works**: Converts spatial image intensities into the frequency domain to inspect the 2D magnitude spectrum.
        - **Why it spots deepfakes**: Generative AI architectures (StyleGAN, Diffusion Models, Flux) leave distinct high-frequency periodic grid artifacts.
        """)

    st.markdown("---")
    st.markdown("### 🧪 Interactive ELA & FFT Experiment Lab")
    lab_img = st.file_uploader(
        "Upload an image for standalone XAI experimentation",
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
