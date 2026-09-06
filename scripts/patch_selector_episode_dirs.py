from pathlib import Path

p = Path("scripts/select_rick_by_voice.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    'REF_DIR = ROOT / "rick_reference" / "parts"',
    'REF_DIR = ROOT / "rick_reference" / "refined_parts"\nif not REF_DIR.exists():\n    REF_DIR = ROOT / "rick_reference" / "parts"'
)

s = s.replace(
    'OUT_DIR = ROOT / "rick_candidates"\nOUT_WAVS = OUT_DIR / "wavs"',
    'OUT_DIR = ROOT / "rick_candidates" / episode\nOUT_WAVS = OUT_DIR / "wavs"'
)

p.write_text(s, encoding="utf-8")
print("patched:", p)
