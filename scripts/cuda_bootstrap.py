from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
NVIDIA_ROOT = ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"

_DLL_HANDLES = []

if NVIDIA_ROOT.exists():
    for dll_dir in NVIDIA_ROOT.rglob("bin"):
        if not any(dll_dir.glob("*.dll")):
            continue

        dll_dir_str = str(dll_dir)

        os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")

        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(dll_dir_str))
            except OSError:
                pass
