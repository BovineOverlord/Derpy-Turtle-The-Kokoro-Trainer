import argparse
import difflib
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from utilities.audio_processor import Transcriber, convert_to_wav_mono_24k
from utilities.kvoicewalk import KVoiceWalk
from utilities.pytorch_sanitizer import load_multiple_voices
from utilities.speech_generator import SpeechGenerator


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return unique_paths


def _expand_target_audio_inputs(primary: str | None, many: list[str] | None) -> list[Path]:
    paths: list[Path] = []

    if primary:
        paths.append(Path(primary))

    if many:
        for item in many:
            if not item:
                continue
            if "," in item:
                split_items = [part.strip() for part in item.split(",") if part.strip()]
                paths.extend(Path(part) for part in split_items)
            else:
                paths.append(Path(item))

    return _dedupe_paths(paths)


def _auto_collect_target_audio_many(primary: Path, limit: int) -> list[Path]:
    if limit <= 0 or not primary.exists() or not primary.is_file():
        return []

    base_stem = primary.stem.lower()
    scored: list[tuple[float, Path]] = []

    for candidate in sorted(primary.parent.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if candidate.resolve() == primary.resolve():
            continue

        similarity = difflib.SequenceMatcher(a=base_stem, b=candidate.stem.lower()).ratio()
        if similarity < 0.45:
            continue

        scored.append((similarity, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:limit]]


def _prepare_target_audio_paths(raw_paths: list[Path], parser: argparse.ArgumentParser) -> list[Path]:
    prepared: list[Path] = []
    for raw_path in raw_paths:
        if not raw_path.is_file():
            parser.error(f"Target audio file not found: {raw_path}")

        try:
            converted = convert_to_wav_mono_24k(raw_path)
        except Exception as exc:
            parser.error(f"Error reading target audio {raw_path}: {exc}")

        prepared.append(Path(converted))

    return prepared


def _expand_target_text_inputs(many: list[str] | None) -> list[str]:
    texts: list[str] = []
    if not many:
        return texts

    for item in many:
        if not item:
            continue
        texts.append(item.strip())

    return texts


def _resolve_text_input(value: str, parser: argparse.ArgumentParser, arg_name: str) -> str:
    text = value.strip()
    if not text:
        return ""

    if text.endswith(".txt"):
        text_path = Path(text)
        if not text_path.is_file():
            parser.error(f"{arg_name} text file not found: {text_path}")
        try:
            return text_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            parser.error(f"Error reading text file {text_path}: {exc}")

    return text


def _resolve_target_texts(
    primary_text: str,
    extra_text_inputs: list[str],
    prepared_target_audio_paths: list[Path],
    parser: argparse.ArgumentParser,
) -> list[str]:
    resolved_primary = _resolve_text_input(primary_text, parser, "--target_text")
    if not resolved_primary:
        parser.error("--target_text is required for random walk mode")

    resolved_extra = [_resolve_text_input(value, parser, "--target_text_many") for value in extra_text_inputs]
    if any(not value for value in resolved_extra):
        parser.error("--target_text_many entries must be non-empty")

    extra_audio_count = max(0, len(prepared_target_audio_paths) - 1)
    if len(resolved_extra) > extra_audio_count:
        parser.error("--target_text_many count must be <= number of extra target audios")

    if len(resolved_extra) < extra_audio_count:
        missing = extra_audio_count - len(resolved_extra)
        print(f"No target text supplied for {missing} extra target audio file(s); defaulting to primary target text.")
        resolved_extra.extend([resolved_primary] * missing)

    return [resolved_primary, *resolved_extra]


def _validate_weights(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    weights = [
        args.score_weight_target,
        args.score_weight_self,
        args.score_weight_feature,
        args.score_weight_accent,
    ]
    if any(weight < 0.0 for weight in weights):
        parser.error("All score weights must be non-negative")
    if sum(weights) <= 0.0:
        parser.error("At least one score weight must be positive")


def main():
    parser = argparse.ArgumentParser(description="A random walk Kokoro voice cloner.")

    parser.add_argument(
        "--target_text",
        type=str,
        help="The words contained in the target audio file. Should be around 100-200 tokens (two sentences). Alternatively, can point to a txt file of the transcription.",
    )

    parser.add_argument(
        "--other_text",
        type=str,
        help="A segment of text used to compare self similarity. Should be around 100-200 tokens.",
        default="If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale.",
    )
    parser.add_argument(
        "--voice_folder",
        type=str,
        help="Path to the voices you want to use as part of the random walk.",
        default="./voices",
    )
    parser.add_argument(
        "--transcribe_start",
        help="Input: filepath to wav file\nOutput: Transcription .txt in ./texts\nTranscribes a target wav and replaces --target_text",
        action="store_true",
    )
    parser.add_argument(
        "--interpolate_start",
        help="Goes through an interpolation search step before random walking",
        action="store_true",
    )
    parser.add_argument(
        "--population_limit",
        type=int,
        help="Limits the amount of voices used as part of initial population selection",
        default=10,
    )
    parser.add_argument(
        "--step_limit",
        type=int,
        help="Limits the amount of steps in the random walk",
        default=10000,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="How often to emit progress logs for non-interactive runs",
        default=10,
    )
    parser.add_argument(
        "--elite_size",
        type=int,
        help="How many elite voices to keep during random walk",
        default=4,
    )
    parser.add_argument(
        "--stagnation_limit",
        type=int,
        help="Trigger a restart after this many non-improving steps",
        default=250,
    )
    parser.add_argument(
        "--restart_diversity",
        type=float,
        help="Mutation diversity used during restart injections",
        default=0.35,
    )
    parser.add_argument(
        "--candidates_per_step",
        type=int,
        help="How many candidate voices to evaluate per random-walk step (base beam width)",
        default=3,
    )
    parser.add_argument(
        "--max_candidates_per_step",
        type=int,
        help="Maximum adaptive beam width used during stagnation",
        default=8,
    )
    parser.add_argument(
        "--no_adaptive_beam",
        action="store_true",
        help="Disable adaptive beam sizing during training.",
    )
    parser.add_argument(
        "--no_dynamic_weight_schedule",
        action="store_true",
        help="Disable automatic score-weight scheduling during training.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["random_walk", "cma_es", "hybrid"],
        default="hybrid",
        help="Optimization strategy. hybrid runs random walk warmup then CMA-ES refinement.",
    )
    parser.add_argument(
        "--refine_top_k",
        type=int,
        default=3,
        help="How many top proxy candidates receive full expensive scoring per step.",
    )
    parser.add_argument(
        "--cma_sigma",
        type=float,
        default=0.35,
        help="Initial CMA-ES sampling scale in latent space.",
    )
    parser.add_argument(
        "--cma_latent_dim",
        type=int,
        default=12,
        help="Latent PCA dimensions used by CMA-ES search.",
    )
    parser.add_argument(
        "--pareto_archive_size",
        type=int,
        default=24,
        help="Archive size for Pareto-ranked multi-objective candidate retention.",
    )
    parser.add_argument(
        "--score_weight_target",
        type=float,
        help="Weight for speaker similarity score",
        default=0.45,
    )
    parser.add_argument(
        "--score_weight_self",
        type=float,
        help="Weight for self-consistency score",
        default=0.33,
    )
    parser.add_argument(
        "--score_weight_feature",
        type=float,
        help="Weight for broad acoustic feature score",
        default=0.10,
    )
    parser.add_argument(
        "--score_weight_accent",
        type=float,
        help="Weight for accent/prosody feature score",
        default=0.12,
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="Filename for the generated output audio",
        default="my_new_voice",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for Kokoro/Resemblyzer/Whisper. auto chooses CUDA when available.",
    )

    group_walk = parser.add_argument_group("Random Walk Mode")
    group_walk.add_argument(
        "--target_audio",
        type=str,
        help="Path to primary target audio file. Must be 24000 Hz mono wav or convertible audio.",
    )
    group_walk.add_argument(
        "--target_audio_many",
        nargs="+",
        help="Optional additional target audios (space-separated paths or comma-separated entries) for multi-clip scoring.",
    )
    group_walk.add_argument(
        "--target_text_many",
        nargs="+",
        help="Optional target texts aligned one-to-one with --target_audio_many (quote each text with spaces).",
    )
    group_walk.add_argument(
        "--no_auto_target_audio_many",
        action="store_true",
        help="Disable automatic pickup of nearby audio files as extra target references.",
    )
    group_walk.add_argument(
        "--auto_target_audio_many_limit",
        type=int,
        default=1,
        help="Maximum number of nearby sibling audio files auto-added to target_audio_many (default: 1).",
    )
    group_walk.add_argument("--starting_voice", type=str, help="Path to the starting voice tensor")

    group_test = parser.add_argument_group("Test Mode")
    group_test.add_argument("--test_voice", type=str, help="Path to the voice tensor you want to test")

    group_util = parser.add_argument_group("Utility Mode")
    group_util.add_argument(
        "--export_bin",
        help="Exports target voices in the --voice_folder directory",
        action="store_true",
    )
    group_util.add_argument(
        "--transcribe_many",
        help="Input: filepath to wav file or folder\nOutput: Individualized transcriptions in ./texts folder\nTranscribes a target wav or wav folder. Replaces --target_text",
    )
    args = parser.parse_args()

    _validate_weights(args, parser)
    if args.auto_target_audio_many_limit < 0:
        parser.error("--auto_target_audio_many_limit must be >= 0")
    if args.candidates_per_step < 1:
        parser.error("--candidates_per_step must be >= 1")
    if args.max_candidates_per_step < 1:
        parser.error("--max_candidates_per_step must be >= 1")
    if args.max_candidates_per_step < args.candidates_per_step:
        parser.error("--max_candidates_per_step must be >= --candidates_per_step")
    if args.refine_top_k < 1:
        parser.error("--refine_top_k must be >= 1")
    if args.cma_sigma <= 0:
        parser.error("--cma_sigma must be > 0")
    if args.cma_latent_dim < 2:
        parser.error("--cma_latent_dim must be >= 2")
    if args.pareto_archive_size < 1:
        parser.error("--pareto_archive_size must be >= 1")

    if args.export_bin:
        if not args.voice_folder:
            parser.error("--voice_folder is required to export a voices bin file")

        file_paths = [os.path.join(args.voice_folder, f) for f in os.listdir(args.voice_folder) if f.endswith(".pt")]
        voices = load_multiple_voices(
            file_paths,
            auto_allow_unsafe=False,
        )

        with open("voices.bin", "wb") as f:
            np.savez(f, **voices)

        return

    target_audio_inputs = _expand_target_audio_inputs(args.target_audio, args.target_audio_many)
    if args.target_audio and not args.no_auto_target_audio_many:
        primary = Path(args.target_audio)
        auto_many = _auto_collect_target_audio_many(primary, args.auto_target_audio_many_limit)
        if auto_many:
            print(f"Auto-added {len(auto_many)} nearby target audio file(s) for multi-target scoring.")
            target_audio_inputs = _dedupe_paths(target_audio_inputs + auto_many)

    prepared_target_audio_paths: list[Path] = []
    if target_audio_inputs:
        prepared_target_audio_paths = _prepare_target_audio_paths(target_audio_inputs, parser)
        args.target_audio = str(prepared_target_audio_paths[0])

    if args.transcribe_start:
        if not prepared_target_audio_paths:
            parser.error("--transcribe_start requires at least one target audio")

        try:
            target_path = prepared_target_audio_paths[0]
            print(f"Sending {target_path.name} for transcription")
            transcriber = Transcriber(device=args.device)
            args.target_text = transcriber.transcribe(audio_path=target_path)
        except Exception as e:
            print(f"Error during transcription: {e}")
            return

    if args.transcribe_many:
        try:
            input_path = Path(args.transcribe_many)

            if input_path.is_file():
                if input_path.suffix.lower() == ".wav":
                    print(f"Sending {input_path.name} for transcription")
                    transcriber = Transcriber(device=args.device)
                    transcriber.transcribe(audio_path=input_path)
                else:
                    print(f"File Format Error: {input_path.name} is not an audio file!")
                return

            if input_path.is_dir():
                wav_files = list(input_path.glob("*.wav"))
                if not wav_files:
                    print(f"No .wav files found in {input_path}")
                    return

                transcriber = Transcriber(device=args.device)
                for audio_file in wav_files:
                    print(f"Sending {audio_file.name} for transcription")
                    transcriber.transcribe(audio_path=audio_file)
                return

            print(f"Input Format Error: {input_path.name} must be a .wav file or a directory!")
            return

        except Exception as e:
            print(f"Error during transcription: {e}")
            return

    if args.target_text:
        args.target_text = _resolve_text_input(args.target_text, parser, "--target_text")

    if args.test_voice:
        if not args.target_text:
            parser.error("--target_text is required when using --test_voice")

        speech_generator = SpeechGenerator(device=args.device)
        audio = speech_generator.generate_audio(args.target_text, args.test_voice)
        output_path = Path(args.output_name)
        if not output_path.suffix:
            output_path = Path("out") / f"{args.output_name}_test.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, 24000)
        print(f"Test Voice output: {output_path.resolve()}")
        return

    if not prepared_target_audio_paths:
        parser.error("--target_audio is required for random walk mode")

    extra_text_inputs = _expand_target_text_inputs(args.target_text_many)
    target_texts = _resolve_target_texts(
        args.target_text or "",
        extra_text_inputs,
        prepared_target_audio_paths,
        parser,
    )
    args.target_text = target_texts[0]

    ktb = KVoiceWalk(
        prepared_target_audio_paths,
        args.target_text,
        target_texts[1:],
        args.other_text,
        args.voice_folder,
        args.interpolate_start,
        args.population_limit,
        args.starting_voice,
        args.output_name,
        args.device,
        elite_size=args.elite_size,
        stagnation_limit=args.stagnation_limit,
        restart_diversity=args.restart_diversity,
        target_weight=args.score_weight_target,
        self_weight=args.score_weight_self,
        feature_weight=args.score_weight_feature,
        accent_weight=args.score_weight_accent,
        candidates_per_step=args.candidates_per_step,
        max_candidates_per_step=args.max_candidates_per_step,
        adaptive_beam=not args.no_adaptive_beam,
        dynamic_weight_schedule=not args.no_dynamic_weight_schedule,
    )
    ktb.optimize(
        args.step_limit,
        args.log_interval,
        optimizer=args.optimizer,
        refine_top_k=args.refine_top_k,
        cma_sigma=args.cma_sigma,
        cma_latent_dim=args.cma_latent_dim,
        pareto_archive_size=args.pareto_archive_size,
    )


if __name__ == "__main__":
    main()
