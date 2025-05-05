import torch
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import numpy as np
import os
import json
import logging
import webrtcvad
import time

# Setup logging
logging.basicConfig(level=logging.INFO, filename="transcription.log")

# Check hardware
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load model
try:
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = asr_model.to(device).half()  # Mixed precision for RTX 2070
    print("Model device:", next(asr_model.parameters()).device)
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise e

# Audio file
audio_path = os.path.join("input", "Audio_16khz.wav")

# Initialize VAD
vad = webrtcvad.Vad(1)  # Least aggressive mode
FRAME_DURATION_MS = 30  # 30ms frames
MIN_FRAME_SAMPLES = 480  # 30ms at 16kHz = 480 samples

# Transcribe with segmentation and VAD
def transcribe_audio(audio_path, segment_length=10, output_dir="/home/raph/transcripts"):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    os.makedirs(output_dir, exist_ok=True)
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    transcripts = []
    total_segments = int(np.ceil(duration / segment_length))

    for i, start in enumerate(np.arange(0, duration, segment_length)):
        end = min(start + segment_length, duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]

        # Pad segment if too short
        if len(segment) < MIN_FRAME_SAMPLES:
            padding = np.zeros(MIN_FRAME_SAMPLES - len(segment))
            segment = np.concatenate([segment, padding])

        # Check for speech using VAD
        is_speech = False
        if len(segment) >= MIN_FRAME_SAMPLES:
            try:
                segment_int16 = (segment * 32768).astype(np.int16)
                is_speech = vad.is_speech(segment_int16.tobytes()[:MIN_FRAME_SAMPLES*2], sr)
            except webrtcvad.Error as e:
                logging.error(f"VAD error at {start:.1f}-{end:.1f}s: {e}")
                is_speech = True  # Fallback to transcribe if VAD fails

        segment_path = os.path.join(output_dir, f"segment_{start:.1f}.wav")
        transcription = ""

        if is_speech:
            try:
                with sf.SoundFile(segment_path, 'w', sr, 1, 'PCM_16') as f:
                    f.write(segment)
                hypothesis = asr_model.transcribe([segment_path])[0]
                transcription = hypothesis.text if hasattr(hypothesis, 'text') else ""
                logging.info(f"Transcribed {start:.1f}-{end:.1f}s: {transcription}")
                print(f"Transcribed {start:.1f}-{end:.1f}s: {transcription} ({i+1}/{total_segments})")
            except Exception as e:
                logging.error(f"Transcription error at {start:.1f}-{end:.1f}s: {e}")
                transcription = ""
        else:
            logging.info(f"Skipped {start:.1f}-{end:.1f}s: No speech detected")

        # Retry file deletion
        for attempt in range(5):
            try:
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                break
            except PermissionError:
                logging.warning(f"PermissionError on {segment_path}, retry {attempt+1}/5")
                time.sleep(0.2)
            except Exception as e:
                logging.error(f"Failed to remove {segment_path}: {e}")
                break

        transcripts.append({
            "start_time": start,
            "end_time": end,
            "text": transcription.strip().lower() if transcription else ""
        })

    return transcripts

# Run and save
try:
    transcripts = transcribe_audio(audio_path)
    with open("transcripts.json", "w") as f:
        json.dump(transcripts, f, indent=2)
    print("Transcription complete. Saved to transcripts.json")
except Exception as e:
    logging.error(f"Transcription failed: {e}")
    raise e