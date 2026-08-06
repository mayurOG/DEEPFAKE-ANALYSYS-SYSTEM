import os
from PIL import Image, ExifTags

SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "canva", "stable diffusion", "midjourney",
    "dall-e", "novelai", "faceapp", "deepface", "faceswap", "reface"
]


def analyze_metadata(image_input) -> dict:
    """
    Extracts EXIF metadata and analyzes image header parameters
    for manipulation indicators.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
        f_path = image_input
        file_size = os.path.getsize(f_path) if os.path.exists(f_path) else None
    elif hasattr(image_input, "read"):
        img = Image.open(image_input)
        image_input.seek(0)
        file_size = len(image_input.read())
        image_input.seek(0)
    elif isinstance(image_input, Image.Image):
        img = image_input
        file_size = None
    else:
        return {"error": "Unsupported image format"}

    width, height = img.size
    format_name = img.format
    mode = img.mode

    exif_data = {}
    software_found = None
    camera_make = None
    camera_model = None
    date_taken = None
    warnings = []

    # Extract EXIF tags if present
    try:
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[str(tag_name)] = str(value)

                if tag_name == "Software":
                    software_found = str(value)
                elif tag_name == "Make":
                    camera_make = str(value)
                elif tag_name == "Model":
                    camera_model = str(value)
                elif tag_name in ["DateTimeOriginal", "DateTime"]:
                    date_taken = str(value)
    except Exception:
        pass

    # Check image info dictionary for metadata
    if not software_found and "software" in img.info:
        software_found = str(img.info["software"])

    if "comment" in img.info:
        comment_str = str(img.info["comment"]).lower()
        for sw in SUSPICIOUS_SOFTWARE:
            if sw in comment_str:
                msg = f"AI Generation signature found in metadata: '{sw}'"
                warnings.append(msg)

    # Flag suspicious editing software
    if software_found:
        sw_lower = software_found.lower()
        for sw in SUSPICIOUS_SOFTWARE:
            if sw in sw_lower:
                warnings.append(f"Edited with software: '{software_found}'")

    # Check for missing EXIF (common in AI-generated or web-scraped images)
    if not exif_data:
        msg = "No EXIF metadata present (Typical of AI generated images)"
        warnings.append(msg)

    dpi = img.info.get("dpi")

    return {
        "dimensions": f"{width} x {height}",
        "width": width,
        "height": height,
        "format": format_name,
        "mode": mode,
        "file_size_bytes": file_size,
        "file_size_kb": f"{file_size / 1024:.1f} KB" if file_size else "N/A",
        "camera_make": camera_make or "Unknown",
        "camera_model": camera_model or "Unknown",
        "software": software_found or "None Detected",
        "date_taken": date_taken or "Unknown",
        "dpi": str(dpi) if dpi else "Unknown",
        "has_exif": len(exif_data) > 0,
        "warnings": warnings,
        "exif_data": exif_data
    }
