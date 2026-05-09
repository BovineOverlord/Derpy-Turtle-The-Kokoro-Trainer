from pathlib import Path

# Resolve the project root from this file's location so paths work regardless of CWD.
ROOT_DIR = Path(__file__).resolve().parent.parent

# In Dir Structure
IN_DIR = ROOT_DIR / "in"

# Out Dir Structure
OUT_DIR = ROOT_DIR / "out"
CONVERTED_DIR = OUT_DIR / "converted_audio"

# Texts Dir Structure
INTERPOLATED_DIR = ROOT_DIR / "interpolated"

# Texts Dir Structure
TEXTS_DIR = ROOT_DIR / "texts"

# Voices Dir Structure
VOICES_DIR = ROOT_DIR / "voices"

# Example DIr Structure
EXAMPLE_DIR = ROOT_DIR / "examples"


# Ensure writable output directories exist on import; soundfile/libsndfile cannot
# create parent directories on its own and fails with "System error" if missing.
for _writable in (IN_DIR, OUT_DIR, CONVERTED_DIR, INTERPOLATED_DIR, TEXTS_DIR):
    _writable.mkdir(parents=True, exist_ok=True)
