from faster_whisper import WhisperModel
import time

print("loading small model on CUDA...")
t0 = time.time()

model = WhisperModel("small", device="cuda", compute_type="float16")

print("CUDA faster-whisper OK")
print("load time:", round(time.time() - t0, 2), "sec")
