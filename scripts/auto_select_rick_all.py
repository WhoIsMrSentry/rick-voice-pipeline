from pathlib import Path
import csv
import shutil
import sys
import wave

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

try:
    from speechbrain.inference.speaker import EncoderClassifier
except Exception:
    from speechbrain.pretrained import EncoderClassifier

from speechbrain.utils.fetching import LocalStrategy

ROOT = Path(__file__).resolve().parents[1]

# Öncelik sırası:
# 1) refined_parts varsa onu kullan
# 2) yoksa temiz s01e01 setini kullan
# 3) yoksa eski parts klasörünü kullan
REF_DIR = ROOT / "rick_reference" / "refined_parts"

if not REF_DIR.exists() or not list(REF_DIR.glob("*.wav")):
    REF_DIR = ROOT / "rick_clean" / "s01e01" / "wavs"

if not REF_DIR.exists() or not list(REF_DIR.glob("*.wav")):
    REF_DIR = ROOT / "rick_reference" / "parts"

AUTO_ROOT = ROOT / "rick_auto_clean"
BORDER_ROOT = ROOT / "rick_borderline"
REPORT_ROOT = ROOT / "rick_reports"

AUTO_ROOT.mkdir(exist_ok=True)
BORDER_ROOT.mkdir(exist_ok=True)
REPORT_ROOT.mkdir(exist_ok=True)

# Güvenli başlangıç eşikleri.
# Auto çok kirli çıkarsa AUTO_THRESHOLD = 0.80 yap.
# Veri az çıkarsa AUTO_THRESHOLD = 0.75 yap.
AUTO_THRESHOLD = 0.77
BORDER_THRESHOLD = 0.69

DEVICE = "cpu"

args = sys.argv[1:]

if args:
    episodes = args
else:
    episodes = sorted(
        p.name for p in (ROOT / "segments_all").glob("s*")
        if p.is_dir()
    )

print("Reference dir:", REF_DIR)
print("Episodes:", episodes)
print("Auto threshold:", AUTO_THRESHOLD)
print("Border threshold:", BORDER_THRESHOLD)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
    run_opts={"device": DEVICE},
    local_strategy=LocalStrategy.COPY,
)

def load_wav_16k(path: Path):
    wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)

    # samples x channels -> mono
    wav_np = wav_np.mean(axis=1)

    wav = torch.from_numpy(wav_np).float().unsqueeze(0)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav

def embed(path: Path):
    wav = load_wav_16k(path).to(DEVICE)

    with torch.no_grad():
        emb = classifier.encode_batch(wav)

    emb = emb.squeeze()
    emb = F.normalize(emb, dim=0)
    return emb.detach().cpu()

def wav_seconds(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())

def bad_text(text: str):
    t = text.lower()

    bad_terms = [
        "www.",
        ".com",
        "rickvemorty",
        "rick ve morty macera",
        "sonsuz",
        "100 yıl",
        "netflix",
    ]

    return any(x in t for x in bad_terms)

ref_files = sorted(REF_DIR.glob("*.wav"))

if not ref_files:
    raise RuntimeError("Referans WAV bulunamadı: " + str(REF_DIR))

print("Reference files:", len(ref_files))

ref_embs = []

for i, p in enumerate(ref_files, 1):
    if i % 25 == 0 or i == 1 or i == len(ref_files):
        print("ref:", i, "/", len(ref_files), p.name)

    ref_embs.append(embed(p))

ref_mean = torch.stack(ref_embs).mean(dim=0)
ref_mean = F.normalize(ref_mean, dim=0)

summary_lines = ["episode\tauto_clips\tauto_minutes\tborder_clips\tborder_minutes"]

for episode in episodes:
    print("")
    print("Episode:", episode)

    seg_dir = ROOT / "segments_all" / episode / "wavs"
    seg_tsv = ROOT / "segments_all" / episode / "segments.tsv"

    if not seg_dir.exists() or not seg_tsv.exists():
        print("Skip missing:", episode)
        continue

    auto_base = AUTO_ROOT / episode
    border_base = BORDER_ROOT / episode

    auto_dir = auto_base / "wavs"
    border_dir = border_base / "wavs"

    if auto_base.exists():
        shutil.rmtree(auto_base)

    if border_base.exists():
        shutil.rmtree(border_base)

    auto_dir.mkdir(parents=True, exist_ok=True)
    border_dir.mkdir(parents=True, exist_ok=True)

    texts = {}

    with seg_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            stem = Path(row["wav"]).stem
            texts[stem] = row["text"]

    rows = []
    seg_files = sorted(seg_dir.glob("*.wav"))

    print("Segment files:", len(seg_files))

    for i, p in enumerate(seg_files, 1):
        stem = p.stem
        text = texts.get(stem, "").strip()

        if not text:
            continue

        seconds = wav_seconds(p)

        if seconds < 0.75 or seconds > 12.0:
            continue

        if bad_text(text):
            continue

        emb = embed(p)
        sim = float(torch.dot(ref_mean, emb))

        rows.append({
            "stem": stem,
            "path": p,
            "similarity": sim,
            "seconds": seconds,
            "text": text,
        })

        if i % 100 == 0:
            print("processed:", i, "/", len(seg_files))

    rows.sort(key=lambda x: x["similarity"], reverse=True)

    auto_meta = []
    border_meta = []

    report_lines = ["rank\tbucket\tsimilarity\tseconds\tstem\ttext"]

    auto_count = 0
    border_count = 0
    auto_seconds = 0.0
    border_seconds = 0.0

    for rank, r in enumerate(rows, 1):
        sim = r["similarity"]

        if sim >= AUTO_THRESHOLD:
            bucket = "auto"
            dst = auto_dir / f'{r["stem"]}.wav'
            shutil.copy2(r["path"], dst)
            auto_meta.append(f'{r["stem"]}|{r["text"]}')
            auto_count += 1
            auto_seconds += r["seconds"]

        elif sim >= BORDER_THRESHOLD:
            bucket = "borderline"
            dst = border_dir / f'{rank:04d}_{sim:.3f}_{r["stem"]}.wav'
            shutil.copy2(r["path"], dst)
            border_meta.append(f'{r["stem"]}|{r["text"]}')
            border_count += 1
            border_seconds += r["seconds"]

        else:
            bucket = "reject"

        report_lines.append(
            f'{rank}\t{bucket}\t{sim:.6f}\t{r["seconds"]:.3f}\t{r["stem"]}\t{r["text"]}'
        )

    (auto_base / "metadata.csv").write_text(
        "\n".join(auto_meta) + "\n",
        encoding="utf-8",
    )

    (border_base / "metadata.csv").write_text(
        "\n".join(border_meta) + "\n",
        encoding="utf-8",
    )

    report_path = REPORT_ROOT / f"{episode}_report.tsv"

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    auto_min = auto_seconds / 60
    border_min = border_seconds / 60

    summary_lines.append(
        f"{episode}\t{auto_count}\t{auto_min:.2f}\t{border_count}\t{border_min:.2f}"
    )

    print("Auto:", auto_count, "clips", round(auto_min, 2), "min")
    print("Border:", border_count, "clips", round(border_min, 2), "min")
    print("Report:", report_path)

summary_path = REPORT_ROOT / "summary.tsv"

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

print("")
print("Done.")
print("Summary:", summary_path)
