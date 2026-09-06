from pathlib import Path
import wave
import statistics

ROOT = Path("final_dataset_rick_tr_v1")
WAVS = ROOT / "wavs"
META = ROOT / "metadata.csv"

lines = META.read_text(encoding="utf-8").splitlines()

missing = []
bad_duration = []
durations = []
sample_rates = {}
channels = {}
duplicate_texts = {}

for line in lines:
    if "|" not in line:
        continue

    stem, text = line.split("|", 1)
    stem = stem.strip()
    text = text.strip()

    duplicate_texts[text] = duplicate_texts.get(text, 0) + 1

    wav = WAVS / f"{stem}.wav"

    if not wav.exists():
        missing.append(stem)
        continue

    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sec = w.getnframes() / float(sr)

    sample_rates[sr] = sample_rates.get(sr, 0) + 1
    channels[ch] = channels.get(ch, 0) + 1
    durations.append(sec)

    if sec < 0.75 or sec > 12.0:
        bad_duration.append((stem, sec, text))

dupes = {k: v for k, v in duplicate_texts.items() if v > 1}

print("metadata lines:", len(lines))
print("wav files:", len(list(WAVS.glob('*.wav'))))
print("missing wavs:", len(missing))
print("bad duration:", len(bad_duration))
print("sample rates:", sample_rates)
print("channels:", channels)

print("total minutes:", round(sum(durations) / 60, 2))
print("min sec:", round(min(durations), 3))
print("max sec:", round(max(durations), 3))
print("avg sec:", round(statistics.mean(durations), 3))
print("median sec:", round(statistics.median(durations), 3))
print("duplicate texts:", len(dupes))

if missing:
    print("MISSING:", missing[:20])

if bad_duration:
    print("BAD DURATION:")
    for x in bad_duration[:20]:
        print(x)
