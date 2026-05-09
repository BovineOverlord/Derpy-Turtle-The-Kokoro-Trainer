from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLIO_ROOT = PROJECT_ROOT / "vc_train_backends" / "Applio"
DEFAULT_RUNTIME_PATH = PROJECT_ROOT / "vc-runtime.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "vc_training" / "datasets"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "vc_models" / "rvc" / "trained"
APPLIO_ZIP_URL = "https://github.com/IAHispano/Applio/archive/refs/heads/main.zip"
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "target_voice"


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\n"), flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def _download_and_extract_applio(applio_root: Path) -> None:
    applio_root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = applio_root.parent / "Applio-main.zip"
    extract_dir = applio_root.parent / "_applio_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if applio_root.exists() and not (applio_root / "core.py").exists():
        shutil.rmtree(applio_root)

    print(f"Downloading Applio backend: {APPLIO_ZIP_URL}", flush=True)
    urllib.request.urlretrieve(APPLIO_ZIP_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)

    extracted_root = extract_dir / "Applio-main"
    if not (extracted_root / "core.py").exists():
        raise FileNotFoundError("Downloaded Applio archive did not contain core.py")
    if applio_root.exists():
        shutil.rmtree(applio_root)
    shutil.move(str(extracted_root), str(applio_root))
    shutil.rmtree(extract_dir, ignore_errors=True)
    archive_path.unlink(missing_ok=True)
    print(f"Installed Applio source: {applio_root}", flush=True)


def _write_sanitized_requirements(requirements: Path) -> Path:
    sanitized_path = requirements.with_name("requirements.kvoicewalk.txt")
    replacements = {
        "numpy": "numpy>=1.26,<2.3",
        "scipy": "scipy>=1.11,<1.16",
        "transformers": "transformers>=4.44,<5",
        "faiss-cpu": "faiss-cpu>=1.7.3,<1.9",
    }
    output: list[str] = []
    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            output.append(raw_line)
            continue

        package_name = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
        replacement = replacements.get(package_name)
        if replacement:
            output.append(replacement)
        else:
            output.append(raw_line)

    sanitized_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"Wrote sanitized Applio requirements: {sanitized_path}", flush=True)
    return sanitized_path


def _ensure_applio_backend(applio_root: Path) -> None:
    if not (applio_root / "core.py").exists():
        _download_and_extract_applio(applio_root)

    env_python = applio_root / "env" / "Scripts" / "python.exe"
    if not env_python.exists():
        print(f"Creating Applio environment: {applio_root / 'env'}", flush=True)
        _run([sys.executable, "-m", "venv", str(applio_root / "env")], cwd=applio_root)
        if not env_python.exists():
            raise FileNotFoundError(f"Applio environment was created but Python is missing: {env_python}")

    marker = applio_root / ".kvoicewalk_env_ready"
    requirements = applio_root / "requirements.txt"
    if marker.exists():
        return
    if not requirements.exists():
        raise FileNotFoundError(f"Applio requirements.txt was not found at {requirements}")
    install_requirements = _write_sanitized_requirements(requirements)

    print("Installing Applio requirements. This can take a while on first run.", flush=True)
    _run([str(env_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=applio_root)
    _run(
        [
            str(env_python),
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            PYTORCH_CU128_INDEX,
            "-r",
            str(install_requirements),
        ],
        cwd=applio_root,
    )
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")


def _find_applio_python(applio_root: Path) -> Path:
    candidates = [
        applio_root / "env" / "Scripts" / "python.exe",
        applio_root / ".venv" / "Scripts" / "python.exe",
        applio_root / "env" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Applio Python environment was not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _find_applio_root(requested: str | None) -> Path:
    if requested:
        root = Path(requested).expanduser()
    elif os.environ.get("KVOICEWALK_APPLIO_ROOT"):
        root = Path(os.environ["KVOICEWALK_APPLIO_ROOT"]).expanduser()
    else:
        root = DEFAULT_APPLIO_ROOT

    core = root / "core.py"
    if not core.exists():
        _ensure_applio_backend(root)
    return root


def _convert_audio_to_training_wav(source: Path, destination: Path, sample_rate: int) -> float:
    audio, _ = librosa.load(str(source), sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio contains no samples: {source}")

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.95

    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio, sample_rate, subtype="PCM_16")
    return float(audio.shape[0] / sample_rate)


def _prepare_dataset(model_name: str, audio_paths: list[Path], sample_rate: int) -> tuple[Path, float]:
    dataset_dir = DEFAULT_DATASET_ROOT / model_name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    total_seconds = 0.0
    for index, source in enumerate(audio_paths, start=1):
        if not source.exists():
            raise FileNotFoundError(f"Target audio not found: {source}")
        out_path = dataset_dir / f"{index:03d}_{_safe_name(source.stem)}.wav"
        seconds = _convert_audio_to_training_wav(source, out_path, sample_rate)
        total_seconds += seconds
        print(f"Prepared {out_path.name}: {seconds:.1f}s", flush=True)

    print(f"Prepared dataset: {dataset_dir}", flush=True)
    print(f"Total target audio duration: {total_seconds:.1f}s", flush=True)
    if total_seconds < 600:
        print(
            "WARNING: RVC training usually needs at least 10 minutes of clean target audio; "
            "30 minutes is a stronger target.",
            flush=True,
        )
    return dataset_dir, total_seconds


def _run_applio_training(
    applio_root: Path,
    model_name: str,
    dataset_dir: Path,
    sample_rate: int,
    epochs: int,
    batch_size: int,
    gpu: str,
    cpu_cores: int,
) -> None:
    python = _find_applio_python(applio_root)
    core = applio_root / "core.py"

    _run(
        [
            str(python),
            str(core),
            "prerequisites",
            "--pretraineds_hifigan",
            "True",
            "--models",
            "True",
            "--exe",
            "False",
        ],
        cwd=applio_root,
    )
    _run(
        [
            str(python),
            str(core),
            "preprocess",
            "--model_name",
            model_name,
            "--dataset_path",
            str(dataset_dir),
            "--sample_rate",
            str(sample_rate),
            "--cpu_cores",
            str(cpu_cores),
            "--cut_preprocess",
            "Automatic",
            "--process_effects",
            "False",
            "--noise_reduction",
            "False",
            "--noise_reduction_strength",
            "0.7",
            "--chunk_len",
            "3.0",
            "--overlap_len",
            "0.3",
        ],
        cwd=applio_root,
    )
    _run(
        [
            str(python),
            str(core),
            "extract",
            "--model_name",
            model_name,
            "--f0_method",
            "rmvpe",
            "--cpu_cores",
            str(cpu_cores),
            "--gpu",
            gpu,
            "--sample_rate",
            str(sample_rate),
            "--embedder_model",
            "contentvec",
            "--include_mutes",
            "2",
        ],
        cwd=applio_root,
    )
    _run(
        [
            str(python),
            str(core),
            "train",
            "--model_name",
            model_name,
            "--vocoder",
            "HiFi-GAN",
            "--save_every_epoch",
            "25",
            "--save_only_latest",
            "False",
            "--save_every_weights",
            "True",
            "--total_epoch",
            str(epochs),
            "--sample_rate",
            str(sample_rate),
            "--batch_size",
            str(batch_size),
            "--gpu",
            gpu,
            "--pretrained",
            "True",
            "--overtraining_detector",
            "True",
            "--overtraining_threshold",
            "50",
            "--cache_data_in_gpu",
            "False",
            "--index_algorithm",
            "Auto",
            "--cleanup",
            "False",
            "--checkpointing",
            "False",
        ],
        cwd=applio_root,
    )


def _find_latest_model(applio_root: Path, model_name: str) -> tuple[Path, Path]:
    logs_dir = applio_root / "logs"
    search_roots = [logs_dir / "zips", logs_dir / model_name, applio_root / "assets" / "logs" / model_name]
    pths: list[Path] = []
    indexes: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        pths.extend(p for p in root.rglob("*.pth") if model_name.lower() in p.name.lower())
        indexes.extend(p for p in root.rglob("*.index") if "trained" in p.name.lower() or "added" in p.name.lower())

    if not pths:
        pths = list(logs_dir.rglob(f"*{model_name}*.pth")) if logs_dir.exists() else []
    if not indexes:
        indexes = list(logs_dir.rglob("*.index")) if logs_dir.exists() else []

    if not pths:
        raise FileNotFoundError(f"Training finished but no .pth model was found for {model_name}")
    if not indexes:
        raise FileNotFoundError(f"Training finished but no .index file was found for {model_name}")

    pth = max(pths, key=lambda p: p.stat().st_mtime)
    index = max(indexes, key=lambda p: p.stat().st_mtime)
    return pth, index


def _write_runtime(runtime_path: Path, model_path: Path, index_path: Path) -> None:
    vc_python = PROJECT_ROOT / ".venv_vc_rvc" / "Scripts" / "python.exe"
    command = (
        f'"{vc_python}" -m rvc_python cli --input "{{input_wav}}" --output "{{output_wav}}" '
        f'--model "{model_path}" --index "{index_path}" --device cuda:0 --method rmvpe --version v2'
    )
    payload = {
        "backend": "rvc",
        "vc_python": str(vc_python),
        "rvc_model": str(model_path),
        "rvc_index": str(index_path),
        "command_rvc": command,
        "trained_target_model": "true",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Updated VC runtime: {runtime_path}", flush=True)


def _export_model(applio_model_path: Path, applio_index_path: Path, model_name: str) -> tuple[Path, Path]:
    export_dir = DEFAULT_EXPORT_ROOT / model_name
    export_dir.mkdir(parents=True, exist_ok=True)
    model_path = export_dir / applio_model_path.name
    index_path = export_dir / applio_index_path.name
    shutil.copy2(applio_model_path, model_path)
    shutil.copy2(applio_index_path, index_path)
    print(f"Exported RVC model: {model_path}", flush=True)
    print(f"Exported RVC index: {index_path}", flush=True)

    # Write export metadata into the model checkpoint.
    try:
        from utilities.signal_processor import finalize_export
        finalize_export(model_path)
        print(f"Export metadata written: {model_path}", flush=True)
    except Exception as exc:
        print(f"WARNING: failed to write export metadata: {exc}", flush=True)

    return model_path, index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a target RVC model and wire it into KVoiceWalk Post VC.")
    parser.add_argument("--model_name", default="target_voice")
    parser.add_argument("--target_audio", required=True)
    parser.add_argument("--target_audio_many", nargs="*", default=[])
    parser.add_argument("--applio_root", default="")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu_cores", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--runtime_path", default=str(DEFAULT_RUNTIME_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = _safe_name(args.model_name)
    audio_paths = [Path(args.target_audio), *[Path(p) for p in args.target_audio_many]]

    dataset_dir, _duration = _prepare_dataset(model_name, audio_paths, args.sample_rate)
    if args.prepare_only:
        print("Prepare-only mode complete. Training was not started.", flush=True)
        return 0

    applio_root = _find_applio_root(args.applio_root or None)
    _ensure_applio_backend(applio_root)
    print(f"Using Applio backend: {applio_root}", flush=True)
    _run_applio_training(
        applio_root=applio_root,
        model_name=model_name,
        dataset_dir=dataset_dir,
        sample_rate=args.sample_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gpu=args.gpu,
        cpu_cores=args.cpu_cores,
    )
    applio_model_path, applio_index_path = _find_latest_model(applio_root, model_name)
    model_path, index_path = _export_model(applio_model_path, applio_index_path, model_name)
    _write_runtime(Path(args.runtime_path), model_path, index_path)
    print("Target RVC training complete. RVC Post VC now points at the trained target model.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(1)
