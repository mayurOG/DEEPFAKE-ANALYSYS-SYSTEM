import os
import io
import torch
import numpy as np

# Set environment variable to prevent duplicate OpenMP library crashes on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_audio(audio_source):
    """
    Robust audio loader supporting multiple formats (WAV, MP3, FLAC, OGG, video tracks).
    Returns (waveform, sample_rate) where:
      - waveform is a torch.FloatTensor of shape (channels, samples)
      - sample_rate is an int
    """
    last_err = None

    # Handle bytes or bytearray
    if isinstance(audio_source, (bytes, bytearray)):
        audio_source = io.BytesIO(audio_source)

    # 1. Try soundfile (handles WAV, FLAC, OGG, and MP3 when libsndfile has mpg123 support)
    if sf is not None:
        try:
            if hasattr(audio_source, "seek"):
                audio_source.seek(0)
            data, sr = sf.read(audio_source, dtype="float32")
            if hasattr(audio_source, "seek"):
                audio_source.seek(0)
            data = np.asarray(data, dtype=np.float32)
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()  # convert (samples, channels) -> (channels, samples)
            return waveform, int(sr)
        except Exception as e:
            last_err = e

    # 2. Try pydub (backed by ffmpeg; universal support for MP3, MP4 video audio, AAC, etc.)
    try:
        from pydub import AudioSegment
        if hasattr(audio_source, "seek"):
            audio_source.seek(0)
        seg = AudioSegment.from_file(audio_source)
        if hasattr(audio_source, "seek"):
            audio_source.seek(0)
        sr = int(seg.frame_rate)
        channels = int(seg.channels)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        max_val = float(1 << (seg.sample_width * 8 - 1))
        if max_val > 0:
            samples = samples / max_val
        if channels > 1:
            samples = samples.reshape((-1, channels)).T
            waveform = torch.from_numpy(samples)
        else:
            waveform = torch.from_numpy(samples).unsqueeze(0)
        return waveform, sr
    except Exception as e:
        last_err = e

    # 3. Try scipy.io.wavfile (standard fallback for PCM WAV)
    try:
        from scipy.io import wavfile
        if hasattr(audio_source, "seek"):
            audio_source.seek(0)
        sr, data = wavfile.read(audio_source)
        if hasattr(audio_source, "seek"):
            audio_source.seek(0)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        return waveform, int(sr)
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"Unable to read audio data with soundfile, pydub/ffmpeg, or scipy. Error: {last_err}"
    )


def resample_waveform(waveform, orig_sr, target_sr=16000):
    """
    Resample a waveform tensor of shape (channels, samples) or (samples,)
    to the target sample rate.
    """
    if orig_sr == target_sr:
        return waveform

    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, orig_sr, target_sr)
    except Exception:
        import scipy.signal
        is_1d = (waveform.ndim == 1)
        arr = waveform.numpy()
        num_target_samples = int(arr.shape[-1] * target_sr / orig_sr)
        resampled = scipy.signal.resample(arr, num_target_samples, axis=-1)
        res_tensor = torch.from_numpy(resampled.astype(np.float32))
        return res_tensor if not is_1d else res_tensor.squeeze(0)
