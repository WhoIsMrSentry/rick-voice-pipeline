from pathlib import Path
import shutil
import wave
import re

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "final_dataset_auto"
OUT_WAVS = OUT / "wavs"

if OUT.exists():
    shutil.rmtree(OUT)

OUT_WAVS.mkdir(parents=True, exist_ok=True)

meta_lines = []
report_lines = ["stem\tseconds\tsource\ttext"]

total_seconds = 0.0
count = 0

def wav_seconds(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())

def bad_text(text: str):
    t = text.lower().strip()

    bad_terms = [
        "www.",
        ".com",
        "rickvemorty",
        "rick ve morty macera",
        "netflix",
    ]

    if any(x in t for x in bad_terms):
        return True

    if len(t) < 2:
        return True

    return False

def add_dataset(source_name: str, wav_dir: Path, metadata_path: Path):
    global count, total_seconds

    if not wav_dir.exists() or not metadata_path.exists():
        print("Skip missing:", source_name)
        return

    lines = metadata_path.read_text(encoding="utf-8").splitlines()

    added = 0
    seconds_sum = 0.0

    for line in lines:
        line = line.strip()

        if not line or "|" not in line:
            continue

        stem, text = line.split("|", 1)
        stem = stem.strip()
        text = text.strip()

        if bad_text(text):
            continue

        src = wav_dir / f"{stem}.wav"

        if not src.exists():
            continue

        seconds = wav_seconds(src)

        if seconds < 0.75 or seconds > 12.0:
            continue

        dst = OUT_WAVS / f"{stem}.wav"
        shutil.copy2(src, dst)

        meta_lines.append(f"{stem}|{text}")
        report_lines.append(f"{stem}\t{seconds:.3f}\t{source_name}\t{text}")

        count += 1
        added += 1
        total_seconds += seconds
        seconds_sum += seconds

    print(source_name, "clips:", added, "minutes:", round(seconds_sum / 60, 2))

# 1) Manuel temizlenmiş s01e01'i tam ekle.
add_dataset(
    "manual_s01e01",
    ROOT / "rick_clean" / "s01e01" / "wavs",
    ROOT / "rick_clean" / "s01e01" / "metadata.csv",
)

# 2) Otomatik yüksek güvenli bölümleri ekle.
# s01e01'i atlıyoruz çünkü manuel temiz versiyonu zaten eklendi.
auto_root = ROOT / "rick_auto_clean"

for ep_dir in sorted(auto_root.glob("s*")):
    if not ep_dir.is_dir():
        continue

    episode = ep_dir.name

    if episode == "s01e01":
        continue

    add_dataset(
        f"auto_{episode}",
        ep_dir / "wavs",
        ep_dir / "metadata.csv",
    )

OUT.mkdir(parents=True, exist_ok=True)

(OUT / "metadata.csv").write_text(
    "\n".join(meta_lines) + "\n",
    encoding="utf-8",
)

(OUT / "report.tsv").write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("")
print("FINAL")
print("clips:", count)
print("seconds:", round(total_seconds, 2))
print("minutes:", round(total_seconds / 60, 2))
print("dataset:", OUT)
print("wavs:", OUT_WAVS)
print("metadata:", OUT / "metadata.csv")
print("report:", OUT / "report.tsv")
