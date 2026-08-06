import json
import base64
from io import BytesIO
from datetime import datetime
from jinja2 import Template
from PIL import Image

HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Deepfake Forensic Report - {{ filename }}</title>
    <style>
        body { font-family: sans-serif; background: #0f172a; color: #e2e8f0; padding: 40px; }
        .container { max-width: 900px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 30px; }
        .header { text-align: center; border-bottom: 2px solid #334155; padding-bottom: 20px; }
        .header h1 { color: #38bdf8; margin: 0; }
        .verdict-box { text-align: center; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .verdict-fake { background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; color: #f87171; }
        .verdict-real { background: rgba(34, 197, 94, 0.15); border: 2px solid #22c55e; color: #4ade80; }
        .verdict-title { font-size: 24px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #0f172a; color: #94a3b8; }
        .badge { background: #7f1d1d; color: #fca5a5; padding: 4px 8px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Deepfake Forensic Intelligence Report</h1>
            <p>Generated on {{ timestamp }} | Media File: <strong>{{ filename }}</strong></p>
        </div>

        <div class="verdict-box {% if verdict == 'Fake' %}verdict-fake{% else %}verdict-real{% endif %}">
            <div class="verdict-title">Final Verdict: {{ verdict.upper() }}</div>
            <div>Confidence Score: {{ confidence }}</div>
        </div>

        <div>
            <h3>📊 Analysis Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {% for key, val in summary_metrics.items() %}
                <tr><td>{{ key }}</td><td>{{ val }}</td></tr>
                {% endfor %}
            </table>
        </div>

        {% if metadata %}
        <div style="margin-top: 20px;">
            <h3>🔍 EXIF & File Header Metadata</h3>
            <table>
                <tr><td>Dimensions</td><td>{{ metadata.dimensions }}</td></tr>
                <tr><td>Format / Mode</td><td>{{ metadata.format }} ({{ metadata.mode }})</td></tr>
                <tr><td>File Size</td><td>{{ metadata.file_size_kb }}</td></tr>
                <tr><td>Camera Make / Model</td><td>{{ metadata.camera_make }} / {{ metadata.camera_model }}</td></tr>
                <tr><td>Software Tag</td><td>{{ metadata.software }}</td></tr>
                <tr><td>Creation Date</td><td>{{ metadata.date_taken }}</td></tr>
            </table>

            {% if metadata.warnings %}
            <div style="margin-top: 15px;">
                <strong style="color: #f87171;">Forensic Warnings:</strong><br>
                {% for warn in metadata.warnings %}
                <span class="badge">⚠️ {{ warn }}</span>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_html_report(
    filename: str, media_type: str, verdict: str,
    confidence: str, summary_metrics: dict, metadata: dict = None
) -> str:
    template = Template(HTML_REPORT_TEMPLATE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    html_out = template.render(
        filename=filename,
        media_type=media_type,
        verdict=verdict,
        confidence=confidence,
        summary_metrics=summary_metrics,
        metadata=metadata,
        timestamp=timestamp
    )
    return html_out


def generate_json_report(
    filename: str, media_type: str, verdict: str,
    confidence: str, summary_metrics: dict, metadata: dict = None
) -> str:
    report_dict = {
        "system": "Deepfake Forensic Intelligence Suite v2.0",
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "media_type": media_type,
        "verdict": verdict,
        "confidence": confidence,
        "summary_metrics": summary_metrics,
        "metadata": metadata or {}
    }
    return json.dumps(report_dict, indent=2)
