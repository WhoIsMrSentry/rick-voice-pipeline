from pathlib import Path
import csv
import shutil
import sys
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

episode = sys.argv[1] if len(sys.argv) > 1 else "s01e01"
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 240

REF_DIR = ROOT / "rick_reference" / "refined_parts"
if not REF_DIR.exists():
    REF_DIR = ROOT / "rick_reference" / "parts"
SEG_DIR = ROOT / "segments_all" / episode / "wavs"
SEG_TSV = ROOT / "segments_all" / episode / "segments.tsv"

OUT_DIR = ROOT / "rick_candidates" / episode
OUT_WAVS = OUT_DIR / "wavs"
OUT_WAVS.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / f"{episode}_rick_similarity.tsv"
META = OUT_DIR / "metadata.csv"

DEVICE = "cpu"

print("DEVICE:", DEVICE)
print("Episode:", episode)
print("Top N:", top_n)

if not REF_DIR.exists():
    raise FileNotFoundError(REF_DIR)

if not SEG_DIR.exists():
    raise FileNotFoundError(SEG_DIR)

if not SEG_TSV.exists():
    raise FileNotFoundError(SEG_TSV)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
    run_opts={"device": DEVICE},
    local_strategy=LocalStrategy.COPY,
)

def load_wav_16k(path: Path):
    wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)

    # shape: samples x channels
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

texts = {}

with SEG_TSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        stem = Path(row["wav"]).stem
        texts[stem] = row["text"]

ref_files = sorted(REF_DIR.glob("*.wav"))

if not ref_files:
    raise RuntimeError("Referans WAV bulunamadı: " + str(REF_DIR))

print("Reference files:", len(ref_files))

ref_embs = []

for i, p in enumerate(ref_files, 1):
    print(f"ref {i}/{len(ref_files)}:", p.name)
    ref_embs.append(embed(p))

ref_mean = torch.stack(ref_embs).mean(dim=0)
ref_mean = F.normalize(ref_mean, dim=0)

seg_files = sorted(SEG_DIR.glob("*.wav"))

if not seg_files:
    raise RuntimeError("Segment WAV bulunamadı: " + str(SEG_DIR))

print("Segment files:", len(seg_files))

rows = []

for i, p in enumerate(seg_files, 1):
    emb = embed(p)
    sim = float(torch.dot(ref_mean, emb))

    stem = p.stem
    text = texts.get(stem, "")

    rows.append({
        "stem": stem,
        "wav": p.name,
        "similarity": sim,
        "text": text,
        "path": p,
    })

    if i % 50 == 0:
        print("processed:", i, "/", len(seg_files))

rows.sort(key=lambda x: x["similarity"], reverse=True)

selected = rows[:top_n]
selected_stems = {r["stem"] for r in selected}

for old in OUT_WAVS.glob("*.wav"):
    old.unlink()

report_lines = ["rank\tselected\tstem\tsimilarity\ttext"]
meta_lines = []

for rank, r in enumerate(rows, 1):
    is_selected = "1" if r["stem"] in selected_stems else "0"
    report_lines.append(
        f'{rank}\t{is_selected}\t{r["stem"]}\t{r["similarity"]:.6f}\t{r["text"]}'
    )

for r in selected:
    dst = OUT_WAVS / f'{r["stem"]}.wav'
    shutil.copy2(r["path"], dst)
    meta_lines.append(f'{r["stem"]}|{r["text"]}')

REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
META.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

print("")
print("Selected:", len(selected))
print("Report:", REPORT)
print("Metadata:", META)
print("Wavs:", OUT_WAVS)
print("")
print("Top 30:")

for i, r in enumerate(rows[:30], 1):
    print(f'{i:03d} {r["similarity"]:.4f} {r["stem"]} | {r["text"]}')
