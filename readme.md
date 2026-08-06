# 🛡️ Deepfake Forensic Intelligence Suite v2.0

An enterprise-grade, multi-modal **Deepfake Forensic Detection & Explainable AI (XAI) Suite** built with Python, PyTorch, Hugging Face Transformers, OpenCV, Torchaudio, and Streamlit.

---

## 🚀 Key Features

### 1. 🧬 Multi-Modal Audio-Visual Video Detection
- **Synchronized Inspection**: Processes MP4/AVI videos frame-by-frame while simultaneously extracting and analyzing the embedded audio track.
- **Plotly Interactive Timeline**: Visualizes frame-by-frame deepfake probability over time with adjustable detection sensitivity thresholds.

### 2. 🔬 Explainable AI (XAI) & Spectral Forensics
- **Error Level Analysis (ELA)**: Computes JPEG re-compression variance heatmaps to highlight visual splicing, face swapping, and manipulation boundaries.
- **2D Fast Fourier Transform (FFT)**: Analyzes 2D spatial frequency spectra to detect high-frequency grid artifacts typical of generative AI models (StyleGAN, Stable Diffusion, Midjourney, Flux).
- **Mel-Spectrogram & Acoustic Formants**: Visualizes audio frequency power distributions to spot vocoder anomalies in synthetic voice clones.

### 3. 📄 EXIF & File Header Metadata Forensics
- Inspects camera make/model, original timestamps, color space, and software tags.
- Automatically flags editing software signatures (Photoshop, Canva, GIMP) and AI generator signatures in metadata.

### 4. ⚡ Batch Multi-File Processing
- Drag-and-drop multiple images or audio clips for automated queue processing.
- Generates comparative tabular summaries with downloadable metrics.

### 5. 📑 Downloadable Forensic Inspection Reports
- Export audit findings as **standalone HTML reports** or **JSON log files**.
- Reports contain confidence scores, timestamped evidence, EXIF metadata tables, and forensic warnings.

### 6. 🎨 Cyberpunk Dark Glassmorphism UI
- Custom Streamlit CSS design system featuring dark glassmorphic cards, glowing status badges, and responsive multi-tab navigation.

---

## 🧠 Neural Models & Forensic Pipeline

| Domain | Algorithm / Model | Provider |
|--------|-------------------|----------|
| Vision Classifier | `prithivMLmods/deepfake-detector-model-v1` (SigLIP) | Hugging Face |
| Face Detection | `MTCNN` (FaceNet-PyTorch) | PyTorch |
| Audio Classification | `mo-thecreator/Deepfake-audio-detection` | Hugging Face |
| Spectral Forensics | 2D FFT Magnitude Spectrum & ELA | SciPy / NumPy / PIL |
| Interactive Charts | Plotly Express & Graph Objects | Plotly |

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YogeshSharma185/Deepfake-Detection-System.git
   cd Deepfake-Detection-System
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Application**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Architecture

```
Deepfake-Detection-System/
├── app.py                      # Main Cyberpunk Streamlit UI
├── requirements.txt            # Project dependencies
├── readme.md                   # Documentation
├── detector/
│   ├── image_detector.py       # Image model + ELA + 2D FFT integration
│   ├── video_detector.py       # Video timeline + Multimodal audio extraction
│   ├── audio_detector.py       # Audio model + Mel-spectrogram + Time series
│   ├── metadata_detector.py    # EXIF metadata & software tag analyzer
│   └── report_generator.py     # HTML & JSON forensic report generator
└── utils/
    ├── ui_components.py        # Glassmorphic CSS & verdict cards
    └── visualizers.py          # ELA, FFT, and Mel-spectrogram generators
```

---

Built with ❤️ by Yogesh & Antigravity