import cuda_bootstrap

from faster_whisper import WhisperModel
from pathlib import Path
import time

audio = Path("audio/cuda_smoke_10s.wav")

print("audio exists:", audio.exists(), audio)
print("loading model...")
t0 = time.time()

model = WhisperModel("small", device="cuda", compute_type="float16")

print("transcribing...")
segments, info = model.transcribe(
    str(audio),
    language="tr",
    beam_size=1,
    vad_filter=False,
)

for seg in segments:
    print(f"{seg.start:.2f} -> {seg.end:.2f} | {seg.text}")

print("CUDA inference OK")
print("time:", round(time.time() - t0, 2), "sec")
