from pathlib import Path

p = Path("scripts/select_rick_by_voice.py")
s = p.read_text(encoding="utf-8")

if "from speechbrain.utils.fetching import LocalStrategy" not in s:
    marker = "except Exception:\n    from speechbrain.pretrained import EncoderClassifier\n"
    replacement = marker + "\nfrom speechbrain.utils.fetching import LocalStrategy\n"
    s = s.replace(marker, replacement)

s = s.replace(
    'run_opts={"device": DEVICE},\n)',
    'run_opts={"device": DEVICE},\n    local_strategy=LocalStrategy.COPY,\n)'
)

p.write_text(s, encoding="utf-8")
print("patched:", p)
