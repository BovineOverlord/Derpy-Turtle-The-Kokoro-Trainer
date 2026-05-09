import datetime
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from utilities.device_utils import resolve_device
from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.path_router import OUT_DIR
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator
from utilities.signal_processor import normalize_output


class KVoiceWalk:
    def __init__(
        self,
        target_audio_paths: list[Path],
        target_text: str,
        target_texts: list[str] | None,
        other_text: str,
        voice_folder: str,
        interpolate_start: bool,
        population_limit: int,
        starting_voice: str,
        output_name: str,
        device: str = "auto",
        elite_size: int = 4,
        stagnation_limit: int = 250,
        restart_diversity: float = 0.35,
        target_weight: float = 0.45,
        self_weight: float = 0.33,
        feature_weight: float = 0.10,
        accent_weight: float = 0.12,
        candidates_per_step: int = 3,
        max_candidates_per_step: int = 8,
        adaptive_beam: bool = True,
        dynamic_weight_schedule: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        print(f"Using device: {self.device}")

        if not target_audio_paths:
            raise ValueError("At least one target audio path is required")

        self.target_audio_paths = [Path(path) for path in target_audio_paths]
        self.target_audio = self.target_audio_paths[0]
        self.target_text = target_text
        self.target_texts = self._normalize_target_texts(target_text, target_texts, len(self.target_audio_paths))
        self.other_text = other_text

        self.elite_size = max(1, int(elite_size))
        self.stagnation_limit = max(1, int(stagnation_limit))
        self.restart_diversity = max(0.01, float(restart_diversity))
        self.candidates_per_step = max(1, int(candidates_per_step))
        self.max_candidates_per_step = max(self.candidates_per_step, int(max_candidates_per_step))
        self.adaptive_beam = bool(adaptive_beam)
        self.dynamic_weight_schedule = bool(dynamic_weight_schedule)

        self.base_weights = self._normalize_weights(
            (
                float(target_weight),
                float(self_weight),
                float(feature_weight),
                float(accent_weight),
            )
        )

        target_paths = [str(path) for path in self.target_audio_paths]
        self.initial_selector = InitialSelector(
            target_paths,
            self.target_texts,
            other_text,
            voice_folder=voice_folder,
            device=self.device,
            target_weight=self.base_weights[0],
            self_weight=self.base_weights[1],
            feature_weight=self.base_weights[2],
            accent_weight=self.base_weights[3],
        )

        if interpolate_start:
            voices = self.initial_selector.interpolate_search(population_limit)
        else:
            voices = self.initial_selector.top_performer_start(population_limit)

        if not voices:
            raise ValueError(f"No .pt voices found in voice folder: {voice_folder}")

        self.speech_generator = SpeechGenerator(device=self.device)
        self.fitness_scorer = FitnessScorer(
            target_paths,
            device=self.device,
            target_weight=self.base_weights[0],
            self_weight=self.base_weights[1],
            feature_weight=self.base_weights[2],
            accent_weight=self.base_weights[3],
        )
        self.voice_generator = VoiceGenerator(voices, starting_voice, device=self.device)
        self.starting_voice = self.voice_generator.starting_voice
        self.output_name = output_name

    @staticmethod
    def _normalize_weights(weights: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        bounded = [max(0.0, float(value)) for value in weights]
        total = sum(bounded)
        if total <= 0.0:
            return (0.45, 0.33, 0.10, 0.12)
        normalized = [value / total for value in bounded]
        return (normalized[0], normalized[1], normalized[2], normalized[3])

    @staticmethod
    def _normalize_target_texts(primary_text: str, extra_texts: list[str] | None, target_count: int) -> list[str]:
        texts = [primary_text]
        if extra_texts:
            texts.extend(extra_texts)

        if len(texts) < target_count:
            texts.extend([primary_text] * (target_count - len(texts)))
        elif len(texts) > target_count:
            texts = texts[:target_count]

        normalized: list[str] = []
        for text in texts:
            value = str(text).strip()
            normalized.append(value if value else primary_text)

        return normalized

    @staticmethod
    def _lerp_weights(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
        t: float,
    ) -> tuple[float, float, float, float]:
        t = max(0.0, min(1.0, float(t)))
        return (
            left[0] + (right[0] - left[0]) * t,
            left[1] + (right[1] - left[1]) * t,
            left[2] + (right[2] - left[2]) * t,
            left[3] + (right[3] - left[3]) * t,
        )

    def _weights_for_state(self, progress: float, stagnation_steps: int) -> tuple[float, float, float, float]:
        if not self.dynamic_weight_schedule:
            return self.base_weights

        start_profile = (0.60, 0.24, 0.10, 0.06)
        mid_profile = (0.55, 0.22, 0.08, 0.15)
        end_profile = (0.52, 0.20, 0.07, 0.21)

        if progress <= 0.60:
            stage_weights = self._lerp_weights(start_profile, mid_profile, progress / 0.60)
        else:
            stage_weights = self._lerp_weights(mid_profile, end_profile, (progress - 0.60) / 0.40)

        # Blend scheduled profile with user baseline so explicit user choices still matter.
        blended = [
            (stage_weights[idx] * 0.65) + (self.base_weights[idx] * 0.35)
            for idx in range(4)
        ]

        # During stagnation, bias toward escaping by prioritizing target+accent signal.
        if stagnation_steps >= max(1, int(self.stagnation_limit * 0.50)):
            blended[0] += 0.03
            blended[3] += 0.02
            blended[1] -= 0.03
            blended[2] -= 0.02

        if stagnation_steps >= max(1, int(self.stagnation_limit * 0.85)):
            blended[0] += 0.02
            blended[3] += 0.02
            blended[1] -= 0.02
            blended[2] -= 0.02

        clamped = tuple(max(0.02, value) for value in blended)
        return self._normalize_weights(clamped)

    def _set_fitness_weights(self, weights: tuple[float, float, float, float]) -> None:
        self.fitness_scorer.target_weight = float(weights[0])
        self.fitness_scorer.self_weight = float(weights[1])
        self.fitness_scorer.feature_weight = float(weights[2])
        self.fitness_scorer.accent_weight = float(weights[3])

    @staticmethod
    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        results = item["results"]
        score = float(results.get("score", 0.0))
        target = float(results.get("target_similarity", 0.0))
        accent = float(results.get("accent_similarity", 0.0))
        self_similarity = float(results.get("self_similarity", 0.0))
        return (score, target + (accent * 0.6), self_similarity)

    @staticmethod
    def _blend_voices(voice_a: torch.Tensor, voice_b: torch.Tensor, alpha: float) -> torch.Tensor:
        return voice_a * (1.0 - alpha) + voice_b * alpha

    def _score_summary(self, results: dict[str, Any]) -> str:
        return (
            f'Target Sim:{results.get("target_similarity", 0.0):.3f} '
            f'Self Sim:{results.get("self_similarity", 0.0):.3f} '
            f'Feature Sim:{results.get("feature_similarity", 0.0):.3f} '
            f'Accent Sim:{results.get("accent_similarity", 0.0):.3f} '
            f'Score:{results.get("score", 0.0):.2f}'
        )

    def _save_best(self, voice: torch.Tensor, results: dict[str, Any], step: int, results_dir: Path) -> None:
        torch.save(
            voice,
            f'{results_dir}/{self.output_name}_{step}_{results["score"]:.2f}_{results["target_similarity"]:.2f}_{self.target_audio.stem}.pt',
        )
        sf.write(
            f'{results_dir}/{self.output_name}_{step}_{results["score"]:.2f}_{results["target_similarity"]:.2f}_{self.target_audio.stem}.wav',
            normalize_output(results["audio"], 24000),
            24000,
        )

    @staticmethod
    def _accept_tiebreak(
        candidate: dict[str, Any],
        reference: dict[str, Any],
        score_epsilon: float,
        target_margin: float,
        accent_margin: float,
        max_self_drop: float,
    ) -> bool:
        candidate_score = float(candidate.get("score", 0.0))
        reference_score = float(reference.get("score", 0.0))

        if candidate_score > reference_score + 1e-6:
            return True

        if abs(candidate_score - reference_score) > score_epsilon:
            return False

        candidate_self = float(candidate.get("self_similarity", 0.0))
        reference_self = float(reference.get("self_similarity", 0.0))
        if candidate_self < (reference_self - max_self_drop):
            return False

        target_improved = float(candidate.get("target_similarity", 0.0)) > (
            float(reference.get("target_similarity", 0.0)) + target_margin
        )
        accent_improved = float(candidate.get("accent_similarity", 0.0)) > (
            float(reference.get("accent_similarity", 0.0)) + accent_margin
        )
        return target_improved or accent_improved

    @staticmethod
    def _accept_frontier(
        candidate: dict[str, Any],
        reference: dict[str, Any],
        target_margin: float,
        accent_margin: float,
        max_score_drop: float,
        max_self_drop: float,
        max_feature_drop: float,
    ) -> bool:
        candidate_score = float(candidate.get("score", 0.0))
        reference_score = float(reference.get("score", 0.0))
        if candidate_score < (reference_score - max_score_drop):
            return False

        candidate_self = float(candidate.get("self_similarity", 0.0))
        reference_self = float(reference.get("self_similarity", 0.0))
        if candidate_self < (reference_self - max_self_drop):
            return False

        candidate_feature = float(candidate.get("feature_similarity", 0.0))
        reference_feature = float(reference.get("feature_similarity", 0.0))
        if candidate_feature < (reference_feature - max_feature_drop):
            return False

        target_improved = float(candidate.get("target_similarity", 0.0)) > (
            float(reference.get("target_similarity", 0.0)) + target_margin
        )
        accent_improved = float(candidate.get("accent_similarity", 0.0)) > (
            float(reference.get("accent_similarity", 0.0)) + accent_margin
        )
        return target_improved or accent_improved

    def _generate_candidate(
        self,
        elites: list[dict[str, Any]],
        global_seeds: list[torch.Tensor],
        stagnation_steps: int,
    ) -> tuple[str, float, torch.Tensor]:
        late_stagnation = stagnation_steps >= max(1, self.stagnation_limit // 2)

        if late_stagnation:
            diversity_low = 0.04
            diversity_high = max(0.30, self.restart_diversity)
        else:
            diversity_low = 0.01
            diversity_high = 0.15

        roll = random.random()
        if roll < 0.60:
            mode = "elite"
            seed = random.choice(elites)["voice"]
        elif roll < 0.85 and len(elites) > 1:
            mode = "crossover"
            left, right = random.sample(elites, 2)
            alpha = random.uniform(0.25, 0.75)
            seed = self._blend_voices(left["voice"], right["voice"], alpha)
        else:
            mode = "global"
            seed = random.choice(global_seeds)

        diversity = random.uniform(diversity_low, diversity_high)
        if mode == "crossover":
            diversity *= 0.70

        voice = self.voice_generator.generate_voice(seed, diversity, device=self.device)
        return mode, diversity, voice

    def _beam_for_state(self, progress: float, stagnation_steps: int) -> int:
        base_beam = self.candidates_per_step
        max_beam = self.max_candidates_per_step
        if not self.adaptive_beam or max_beam <= base_beam:
            return base_beam

        stagnation_ratio = float(stagnation_steps) / float(max(1, self.stagnation_limit))
        beam = base_beam

        if stagnation_ratio >= 0.35:
            beam = min(max_beam, beam + 1)
        if stagnation_ratio >= 0.60:
            beam = min(max_beam, beam + 1)
        if stagnation_ratio >= 0.85:
            beam = min(max_beam, beam + 1)

        # In late training, widen slightly earlier to avoid plateau lock-in.
        if progress >= 0.75 and stagnation_ratio >= 0.20:
            beam = min(max_beam, beam + 1)

        return max(1, beam)

    def random_walk(self, step_limit: int, log_interval: int = 10) -> dict[str, Any]:
        if log_interval < 1:
            log_interval = 1

        interactive = sys.stdout.isatty()
        started_at = time.time()

        if interactive:
            loop = tqdm(range(step_limit), desc="Random Walk", unit="step")
            progress_write = loop.write
        else:
            loop = range(step_limit)
            progress_write = print

        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR / f"{self.output_name}_{self.target_audio.stem}_{now.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)

        candidate_voices = list(self.voice_generator.voices)
        candidate_voices.append(self.starting_voice)
        candidate_voices.append(self.voice_generator.mean)

        initial_weights = self._weights_for_state(progress=0.0, stagnation_steps=0)
        self._set_fitness_weights(initial_weights)

        scored_candidates: list[dict[str, Any]] = []
        for voice in candidate_voices:
            results = self.score_voice(voice)
            scored_candidates.append({"voice": voice, "results": results})

        scored_candidates.sort(key=self._sort_key, reverse=True)
        elite_size = min(self.elite_size, len(scored_candidates))
        elites = scored_candidates[:elite_size]

        best_voice = elites[0]["voice"]
        best_results = elites[0]["results"]
        stagnation_steps = 0

        progress_write("Initial elite pool:")
        for idx, elite in enumerate(elites, start=1):
            progress_write(f"Elite {idx}: {self._score_summary(elite['results'])}")

        self._save_best(best_voice, best_results, 0, results_dir)

        global_seeds = list(self.voice_generator.voices)
        global_seeds.append(self.starting_voice)
        global_seeds.append(self.voice_generator.mean)

        for i in loop:
            progress = (i + 1) / max(1, step_limit)
            active_weights = self._weights_for_state(progress=progress, stagnation_steps=stagnation_steps)
            self._set_fitness_weights(active_weights)
            active_beam = self._beam_for_state(progress=progress, stagnation_steps=stagnation_steps)

            min_similarity = max(best_results["target_similarity"] * 0.98, 0.0)
            if stagnation_steps >= max(1, self.stagnation_limit // 2):
                min_similarity *= 0.94
            if stagnation_steps >= int(self.stagnation_limit * 0.9):
                min_similarity *= 0.90

            step_candidates: list[dict[str, Any]] = []
            for _ in range(active_beam):
                mode, diversity, voice = self._generate_candidate(elites, global_seeds, stagnation_steps)
                voice_results = self.score_voice(voice, min_similarity)
                step_candidates.append(
                    {
                        "voice": voice,
                        "results": voice_results,
                        "mode": mode,
                        "diversity": diversity,
                    }
                )

            step_candidates.sort(key=self._sort_key, reverse=True)

            for candidate in step_candidates:
                worst_results = elites[-1]["results"]
                if self._accept_tiebreak(
                    candidate["results"],
                    worst_results,
                    score_epsilon=0.03,
                    target_margin=0.004,
                    accent_margin=0.006,
                    max_self_drop=0.03,
                ):
                    elites.append({"voice": candidate["voice"], "results": candidate["results"]})
                    elites.sort(key=self._sort_key, reverse=True)
                    elites = elites[:elite_size]

            best_candidate = step_candidates[0]
            improved = self._accept_tiebreak(
                best_candidate["results"],
                best_results,
                score_epsilon=0.05,
                target_margin=0.002,
                accent_margin=0.003,
                max_self_drop=0.02,
            )

            if improved:
                best_results = best_candidate["results"]
                best_voice = best_candidate["voice"]
                stagnation_steps = 0
                if interactive:
                    progress_write(
                        f"Step:{i:<6} {self._score_summary(best_results)} "
                        f"Diversity:{best_candidate['diversity']:.2f} Mode:{best_candidate['mode']}"
                    )
                self._save_best(best_voice, best_results, i + 1, results_dir)
            else:
                stagnation_steps += 1

            if stagnation_steps >= self.stagnation_limit:
                restart_candidates: list[dict[str, Any]] = list(elites)
                restart_trials = max(6, elite_size * max(2, active_beam))
                restart_min_similarity = max(best_results["target_similarity"] * 0.85, 0.0)

                for _ in range(restart_trials):
                    roll = random.random()
                    if roll < 0.35:
                        seed = random.choice(elites)["voice"]
                    elif roll < 0.70:
                        seed = random.choice(global_seeds)
                    else:
                        elite_seed = random.choice(elites)["voice"]
                        global_seed = random.choice(global_seeds)
                        seed = self._blend_voices(elite_seed, global_seed, random.uniform(0.2, 0.8))

                    restart_voice = self.voice_generator.generate_voice(
                        seed,
                        diversity=random.uniform(max(0.18, self.restart_diversity * 0.6), max(0.45, self.restart_diversity)),
                        device=self.device,
                    )
                    restart_results = self.score_voice(restart_voice, min_similarity=restart_min_similarity)
                    restart_candidates.append({"voice": restart_voice, "results": restart_results})

                restart_candidates.sort(key=self._sort_key, reverse=True)
                elites = restart_candidates[:elite_size]

                accent_best = max(restart_candidates, key=lambda item: float(item["results"].get("accent_similarity", 0.0)))
                if not any(candidate is accent_best for candidate in elites):
                    if float(accent_best["results"].get("accent_similarity", 0.0)) > float(
                        elites[-1]["results"].get("accent_similarity", 0.0)
                    ) + 0.03:
                        elites[-1] = accent_best
                        elites.sort(key=self._sort_key, reverse=True)

                elite_head = elites[0]["results"]
                if self._accept_tiebreak(
                    elite_head,
                    best_results,
                    score_epsilon=0.05,
                    target_margin=0.001,
                    accent_margin=0.002,
                    max_self_drop=0.03,
                ):
                    best_voice = elites[0]["voice"]
                    best_results = elite_head
                    self._save_best(best_voice, best_results, i + 1, results_dir)

                progress_write(
                    f"Restart triggered at step {i + 1}: mode mix injected -> "
                    f"elite head {self._score_summary(elites[0]['results'])}"
                )
                stagnation_steps = 0

            if not interactive and ((i + 1) % log_interval == 0 or i == step_limit - 1):
                completed = i + 1
                elapsed = time.time() - started_at
                eta = (elapsed / completed) * (step_limit - completed) if completed > 0 else 0.0
                beam_text = (
                    f"{active_beam}/{self.max_candidates_per_step}"
                    if self.adaptive_beam
                    else str(active_beam)
                )
                print(
                    f"Progress: step {completed}/{step_limit} | "
                    f"best_score={best_results.get('score', 0.0):.2f} | "
                    f"best_target_sim={best_results.get('target_similarity', 0.0):.3f} | "
                    f"best_accent_sim={best_results.get('accent_similarity', 0.0):.3f} | "
                    f"stagnation={stagnation_steps}/{self.stagnation_limit} | "
                    f"beam={beam_text} | "
                    f"w=t:{active_weights[0]:.2f}/s:{active_weights[1]:.2f}/f:{active_weights[2]:.2f}/a:{active_weights[3]:.2f} | "
                    f"elapsed={elapsed:.1f}s | "
                    f"eta={eta:.1f}s"
                )

        elapsed = time.time() - started_at
        print(f"Random Walk Final Results for {self.output_name}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Best Score: {best_results['score']:.2f}_")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}_")
        print(f"Best Accent Similarity: {best_results.get('accent_similarity', 0.0):.2f}_")
        print(f"Random Walk pt and wav files ---> {results_dir}")
        return {"voice": best_voice, "results": best_results, "results_dir": results_dir}

    def score_voice(self, voice: torch.Tensor, min_similarity: float = 0.0) -> dict[str, Any]:
        """Using a weighted harmonic mean to score voice similarity and accent consistency."""
        target_audios = [self.speech_generator.generate_audio(text, voice) for text in self.target_texts]
        primary_audio = target_audios[0] if target_audios else self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity_pairwise(target_audios)
        results: dict[str, Any] = {"audio": primary_audio}

        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            results.update(self.fitness_scorer.hybrid_similarity(target_audios, audio2, target_similarity))
        else:
            results["score"] = 0.0
            results["target_similarity"] = float(target_similarity)
            results["self_similarity"] = 0.0
            results["feature_similarity"] = 0.0
            results["accent_similarity"] = 0.0

        return results




    def optimize(
        self,
        step_limit: int,
        log_interval: int = 10,
        optimizer: str = "hybrid",
        refine_top_k: int = 3,
        cma_sigma: float = 0.35,
        cma_latent_dim: int = 12,
        pareto_archive_size: int = 24,
    ) -> None:
        optimizer_name = str(optimizer).strip().lower()
        if optimizer_name not in {"random_walk", "cma_es", "hybrid"}:
            optimizer_name = "hybrid"

        if optimizer_name == "random_walk":
            self.random_walk(step_limit, log_interval)
            return

        if optimizer_name == "hybrid":
            if step_limit < 4:
                self.cma_es_walk(
                    step_limit,
                    log_interval,
                    refine_top_k=refine_top_k,
                    cma_sigma=cma_sigma,
                    cma_latent_dim=cma_latent_dim,
                    pareto_archive_size=pareto_archive_size,
                )
                return

            rw_steps = max(1, int(step_limit * 0.35))
            cma_steps = max(1, step_limit - rw_steps)
            print(
                f"Hybrid optimizer selected: random_walk warmup={rw_steps} steps -> "
                f"cma_es refine={cma_steps} steps"
            )
            warmup = self.random_walk(rw_steps, log_interval)
            self.cma_es_walk(
                cma_steps,
                log_interval,
                refine_top_k=refine_top_k,
                cma_sigma=cma_sigma,
                cma_latent_dim=cma_latent_dim,
                pareto_archive_size=pareto_archive_size,
                seed_voice=warmup.get("voice"),
                seed_results=warmup.get("results"),
            )
            return

        self.cma_es_walk(
            step_limit,
            log_interval,
            refine_top_k=refine_top_k,
            cma_sigma=cma_sigma,
            cma_latent_dim=cma_latent_dim,
            pareto_archive_size=pareto_archive_size,
        )

    def _score_voice_proxy(self, voice: torch.Tensor, min_similarity: float = 0.0) -> dict[str, Any]:
        target_audios = [self.speech_generator.generate_audio(text, voice) for text in self.target_texts]
        primary_audio = target_audios[0] if target_audios else self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = float(self.fitness_scorer.target_similarity_pairwise(target_audios))

        results: dict[str, Any] = {
            "audio": primary_audio,
            "target_audios": target_audios,
            "target_similarity": target_similarity,
            "self_similarity": 0.0,
            "feature_similarity": 0.0,
            "accent_similarity": 0.0,
            "score": 0.0,
            "proxy_score": 0.0,
        }

        if target_similarity <= min_similarity:
            return results

        features = self.fitness_scorer._aggregate_features(target_audios)
        feature_penalty = float(self.fitness_scorer.target_feature_penalty(features))
        accent_penalty = float(self.fitness_scorer.accent_penalty(features))

        feature_similarity = max((100.0 - feature_penalty) / 100.0, 0.01)
        accent_similarity = max((100.0 - accent_penalty) / 100.0, 0.01)

        proxy_weights = self._normalize_weights(
            (
                self.fitness_scorer.target_weight,
                0.0,
                self.fitness_scorer.feature_weight * 0.70,
                self.fitness_scorer.accent_weight + (self.fitness_scorer.self_weight * 0.65),
            )
        )
        weighted_values = [target_similarity, feature_similarity, accent_similarity]
        weighted_weights = [proxy_weights[0], proxy_weights[2], proxy_weights[3]]
        proxy_score = self._weighted_harmonic(weighted_values, weighted_weights) * 100.0

        results["feature_similarity"] = float(feature_similarity)
        results["accent_similarity"] = float(accent_similarity)
        results["score"] = float(proxy_score)
        results["proxy_score"] = float(proxy_score)
        return results

    def _score_voice_from_proxy(
        self,
        voice: torch.Tensor,
        proxy_results: dict[str, Any],
        min_similarity: float = 0.0,
    ) -> dict[str, Any]:
        target_similarity = float(proxy_results.get("target_similarity", 0.0))
        if target_similarity <= min_similarity:
            return {
                "audio": proxy_results.get("audio"),
                "score": 0.0,
                "target_similarity": target_similarity,
                "self_similarity": 0.0,
                "feature_similarity": 0.0,
                "accent_similarity": 0.0,
                "proxy_score": float(proxy_results.get("proxy_score", 0.0)),
            }

        target_audios = proxy_results.get("target_audios")
        if not target_audios:
            target_audios = [self.speech_generator.generate_audio(text, voice) for text in self.target_texts]

        audio2 = self.speech_generator.generate_audio(self.other_text, voice)
        full_results = self.fitness_scorer.hybrid_similarity(target_audios, audio2, target_similarity)
        full_results["audio"] = proxy_results.get("audio", target_audios[0] if target_audios else None)
        full_results["proxy_score"] = float(proxy_results.get("proxy_score", 0.0))
        return full_results

    def _score_candidates_two_stage(
        self,
        candidates: list[dict[str, Any]],
        min_similarity: float,
        full_eval_budget: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        full_eval_budget = max(1, min(int(full_eval_budget), len(candidates)))

        for candidate in candidates:
            candidate["results"] = self._score_voice_proxy(candidate["voice"], min_similarity=min_similarity)

        candidates.sort(
            key=lambda item: (
                float(item["results"].get("proxy_score", item["results"].get("score", 0.0))),
                float(item["results"].get("target_similarity", 0.0)),
                float(item["results"].get("accent_similarity", 0.0)),
            ),
            reverse=True,
        )

        for idx, candidate in enumerate(candidates):
            if idx < full_eval_budget:
                candidate["results"] = self._score_voice_from_proxy(
                    candidate["voice"],
                    candidate["results"],
                    min_similarity=min_similarity,
                )
                candidate["full_eval"] = True
            else:
                candidate["full_eval"] = False
                candidate["results"]["score"] = float(candidate["results"].get("score", 0.0)) * 0.97

        candidates.sort(key=self._sort_key, reverse=True)
        return candidates

    @staticmethod
    def _dominates_results(left: dict[str, Any], right: dict[str, Any]) -> bool:
        keys = ("target_similarity", "accent_similarity", "self_similarity")
        at_least_equal = True
        strictly_better = False

        for key in keys:
            lv = float(left.get(key, 0.0))
            rv = float(right.get(key, 0.0))
            if lv < rv - 1e-6:
                at_least_equal = False
                break
            if lv > rv + 1e-6:
                strictly_better = True

        return at_least_equal and strictly_better

    def _pareto_ranks(self, results: list[dict[str, Any]]) -> list[int]:
        n = len(results)
        if n == 0:
            return []

        dominates: list[set[int]] = [set() for _ in range(n)]
        dominated_counts = [0] * n
        ranks = [0] * n

        first_front: list[int] = []
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates_results(results[i], results[j]):
                    dominates[i].add(j)
                    dominated_counts[j] += 1
                elif self._dominates_results(results[j], results[i]):
                    dominates[j].add(i)
                    dominated_counts[i] += 1

            if dominated_counts[i] == 0:
                first_front.append(i)

        current_front = first_front
        current_rank = 0
        while current_front:
            next_front: list[int] = []
            for idx in current_front:
                ranks[idx] = current_rank
                for dominated_idx in dominates[idx]:
                    dominated_counts[dominated_idx] -= 1
                    if dominated_counts[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            current_rank += 1
            current_front = next_front

        return ranks

    def _select_pareto_elites(self, pool: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
        if not pool:
            return []

        keep = max(1, min(int(keep), len(pool)))
        ranks = self._pareto_ranks([item["results"] for item in pool])

        indexed = list(enumerate(pool))
        indexed.sort(
            key=lambda item: (
                ranks[item[0]],
                -self._sort_key(item[1])[0],
                -self._sort_key(item[1])[1],
                -self._sort_key(item[1])[2],
            )
        )

        return [item for _, item in indexed[:keep]]

    def _build_latent_basis(self, seeds: list[torch.Tensor], cma_latent_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = torch.stack([seed.detach().reshape(-1).to(self.voice_generator.device) for seed in seeds], dim=0)
        center = flat.mean(dim=0)
        demeaned = flat - center

        max_rank = min(
            max(2, int(cma_latent_dim)),
            max(1, int(demeaned.shape[0]) - 1),
            int(demeaned.shape[1]),
        )

        if max_rank <= 1:
            basis = torch.zeros((1, center.numel()), device=center.device, dtype=center.dtype)
            basis[0, 0] = 1.0
            return center, basis

        try:
            _, _, v = torch.pca_lowrank(demeaned, q=max_rank, center=False)
            basis = v[:, :max_rank].T.contiguous()
        except Exception:
            _, _, vh = np.linalg.svd(demeaned.detach().cpu().numpy(), full_matrices=False)
            basis_np = vh[:max_rank, :]
            basis = torch.tensor(basis_np, device=center.device, dtype=center.dtype)

        if not torch.isfinite(basis).all() or basis.numel() == 0:
            basis = torch.zeros((1, center.numel()), device=center.device, dtype=center.dtype)
            basis[0, 0] = 1.0

        return center, basis

    def _project_voice_to_latent(self, voice: torch.Tensor, center: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
        flat = voice.detach().reshape(-1).to(center.device)
        latent = torch.matmul(basis, flat - center)
        return latent.detach().cpu().numpy().astype(np.float64)

    def _decode_latent_voice(
        self,
        latent: np.ndarray,
        center: torch.Tensor,
        basis: torch.Tensor,
        flat_min: torch.Tensor,
        flat_max: torch.Tensor,
        voice_shape: tuple[int, ...],
    ) -> torch.Tensor:
        latent_tensor = torch.tensor(latent, dtype=center.dtype, device=center.device)
        flat = center + torch.matmul(latent_tensor, basis)
        flat = torch.max(torch.min(flat, flat_max), flat_min)
        return flat.reshape(voice_shape).detach()

    def cma_es_walk(
        self,
        step_limit: int,
        log_interval: int = 10,
        refine_top_k: int = 3,
        cma_sigma: float = 0.35,
        cma_latent_dim: int = 12,
        pareto_archive_size: int = 24,
        seed_voice: torch.Tensor | None = None,
        seed_results: dict[str, Any] | None = None,
    ) -> None:
        if log_interval < 1:
            log_interval = 1

        refine_top_k = max(1, int(refine_top_k))
        cma_sigma = max(0.05, float(cma_sigma))
        cma_latent_dim = max(2, int(cma_latent_dim))
        pareto_archive_size = max(self.elite_size, int(pareto_archive_size))

        interactive = sys.stdout.isatty()
        started_at = time.time()

        if interactive:
            loop = tqdm(range(step_limit), desc="CMA-ES", unit="step")
            progress_write = loop.write
        else:
            loop = range(step_limit)
            progress_write = print

        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR / f"{self.output_name}_{self.target_audio.stem}_{now.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)

        candidate_voices = list(self.voice_generator.voices)
        candidate_voices.append(self.starting_voice)
        candidate_voices.append(self.voice_generator.mean)
        if seed_voice is not None:
            candidate_voices.append(seed_voice.detach().to(self.voice_generator.device))

        initial_weights = self._weights_for_state(progress=0.0, stagnation_steps=0)
        self._set_fitness_weights(initial_weights)

        initial_scored: list[dict[str, Any]] = []
        if seed_voice is not None and isinstance(seed_results, dict):
            initial_scored.append(
                {
                    "voice": seed_voice.detach().to(self.voice_generator.device),
                    "results": seed_results,
                }
            )
        for voice in candidate_voices:
            initial_scored.append({"voice": voice, "results": self.score_voice(voice)})

        elite_size = min(self.elite_size, len(initial_scored))
        elites = self._select_pareto_elites(initial_scored, elite_size)
        archive = self._select_pareto_elites(initial_scored, pareto_archive_size)

        best_item = max(elites, key=self._sort_key)
        best_voice = best_item["voice"]
        best_results = best_item["results"]
        best_target_seen = float(best_results.get("target_similarity", 0.0))
        best_accent_seen = float(best_results.get("accent_similarity", 0.0))
        self._save_best(best_voice, best_results, 0, results_dir)

        basis_seeds = [item["voice"] for item in archive]
        center, basis = self._build_latent_basis(basis_seeds, cma_latent_dim)

        mean_latent = self._project_voice_to_latent(best_voice, center, basis)
        latent_dim = int(mean_latent.shape[0])
        sigma = float(cma_sigma)
        diag_scale = np.ones(latent_dim, dtype=np.float64)

        flat_min = self.voice_generator.min.reshape(-1).to(self.voice_generator.device)
        flat_max = self.voice_generator.max.reshape(-1).to(self.voice_generator.device)
        voice_shape = tuple(best_voice.shape)

        stagnation_steps = 0

        progress_write("Initial CMA-ES elite pool:")
        for idx, elite in enumerate(elites, start=1):
            progress_write(f"Elite {idx}: {self._score_summary(elite['results'])}")

        for i in loop:
            progress = (i + 1) / max(1, step_limit)
            active_weights = self._weights_for_state(progress=progress, stagnation_steps=stagnation_steps)
            self._set_fitness_weights(active_weights)

            stagnation_ratio = float(stagnation_steps) / float(max(1, self.stagnation_limit))
            pop_size = max(
                6,
                self._beam_for_state(progress=progress, stagnation_steps=stagnation_steps)
                + self.candidates_per_step
                + 2,
            )
            best_target = float(best_results.get("target_similarity", 0.0))
            gate_scale = 0.92 - (0.16 * min(1.0, stagnation_ratio))
            min_similarity = max(best_target * gate_scale, best_target - 0.22, 0.45)

            samples: list[dict[str, Any]] = []
            for _ in range(pop_size):
                noise = np.random.randn(latent_dim)
                latent = mean_latent + (sigma * diag_scale * noise)
                voice = self._decode_latent_voice(latent, center, basis, flat_min, flat_max, voice_shape)
                samples.append({"voice": voice, "latent": latent, "mode": "cma_es", "diversity": float(sigma)})

            exploration_count = max(1, int(np.ceil(pop_size * 0.25)))
            for _ in range(exploration_count):
                if random.random() < 0.55 and elites:
                    seed = random.choice(elites)["voice"]
                elif archive:
                    seed = random.choice(archive)["voice"]
                else:
                    seed = random.choice(candidate_voices)

                diversity = random.uniform(
                    max(0.04, sigma * 0.08),
                    min(0.55, 0.14 + (0.25 * stagnation_ratio)),
                )
                explore_voice = self.voice_generator.generate_voice(
                    seed,
                    diversity=diversity,
                    device=self.device,
                    clip=True,
                )
                explore_latent = self._project_voice_to_latent(explore_voice, center, basis)
                samples.append(
                    {
                        "voice": explore_voice,
                        "latent": explore_latent,
                        "mode": "explore",
                        "diversity": float(diversity),
                    }
                )

            eval_fraction = 0.45 + (0.40 * min(1.0, stagnation_ratio))
            full_eval_budget = min(
                len(samples),
                max(refine_top_k, int(np.ceil(len(samples) * eval_fraction))),
            )
            scored = self._score_candidates_two_stage(samples, min_similarity=min_similarity, full_eval_budget=full_eval_budget)

            pareto_ranks = self._pareto_ranks([item["results"] for item in scored])
            for idx, candidate in enumerate(scored):
                candidate["pareto_rank"] = pareto_ranks[idx]

            scored.sort(
                key=lambda item: (
                    int(item.get("pareto_rank", 0)),
                    0 if item.get("full_eval") else 1,
                    -self._sort_key(item)[0],
                    -self._sort_key(item)[1],
                    -self._sort_key(item)[2],
                )
            )

            mu = max(2, len(scored) // 2)
            full_eval_parents = [item for item in scored if item.get("full_eval")]
            if len(full_eval_parents) >= 2:
                parents: list[dict[str, Any]] = []
                for item in full_eval_parents:
                    parents.append(item)
                    if len(parents) >= mu:
                        break
                if len(parents) < mu:
                    for item in scored:
                        if item in parents:
                            continue
                        parents.append(item)
                        if len(parents) >= mu:
                            break
            else:
                parents = scored[:mu]

            mu = max(2, len(parents))
            parent_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            parent_weights = parent_weights / np.sum(parent_weights)

            old_mean = mean_latent.copy()
            parent_latents = np.stack([parent["latent"] for parent in parents], axis=0)
            mean_latent = np.sum(parent_latents * parent_weights[:, None], axis=0)

            spread = np.sqrt(np.mean((parent_latents - old_mean) ** 2, axis=0))
            diag_scale = (0.85 * diag_scale) + (0.15 * np.clip(spread / max(sigma, 1e-6), 0.35, 3.0))
            diag_scale = np.clip(diag_scale, 0.30, 4.0)

            candidate_pool = [{"voice": item["voice"], "results": item["results"]} for item in parents]
            elites = self._select_pareto_elites([*elites, *candidate_pool], elite_size)
            archive = self._select_pareto_elites(
                [
                    *archive,
                    *[{"voice": item["voice"], "results": item["results"]} for item in scored[:full_eval_budget]],
                ],
                pareto_archive_size,
            )

            full_eval_scored = [item for item in scored if item.get("full_eval")]
            best_candidate = max(full_eval_scored, key=self._sort_key) if full_eval_scored else max(scored, key=self._sort_key)
            improved_by_score = self._accept_tiebreak(
                best_candidate["results"],
                best_results,
                score_epsilon=0.04,
                target_margin=0.001,
                accent_margin=0.002,
                max_self_drop=0.03,
            )
            improved_by_frontier = self._accept_frontier(
                best_candidate["results"],
                best_results,
                target_margin=0.003,
                accent_margin=0.006,
                max_score_drop=2.50,
                max_self_drop=0.04,
                max_feature_drop=0.18,
            )
            improved = improved_by_score or improved_by_frontier

            if improved:
                best_results = best_candidate["results"]
                best_voice = best_candidate["voice"]
                stagnation_steps = 0
                sigma = max(0.05, sigma * 0.96)
                self._save_best(best_voice, best_results, i + 1, results_dir)
                best_target_seen = max(best_target_seen, float(best_results.get("target_similarity", 0.0)))
                best_accent_seen = max(best_accent_seen, float(best_results.get("accent_similarity", 0.0)))
                if interactive:
                    reason = "score" if improved_by_score else "frontier"
                    progress_write(
                        f"Step:{i:<6} {self._score_summary(best_results)} "
                        f"Sigma:{sigma:.4f} ParetoRank:{best_candidate.get('pareto_rank', 0)} "
                        f"Accept:{reason}"
                    )
            else:
                stagnation_steps += 1
                sigma = min(2.0, sigma * 1.01)

            # Persist useful tradeoff checkpoints even when global score does not improve.
            frontier_head = max(elites, key=self._sort_key)
            frontier_results = frontier_head["results"]
            frontier_target = float(frontier_results.get("target_similarity", 0.0))
            frontier_accent = float(frontier_results.get("accent_similarity", 0.0))
            if (
                frontier_target > (best_target_seen + 0.002)
                or frontier_accent > (best_accent_seen + 0.004)
            ) and self._accept_frontier(
                frontier_results,
                best_results,
                target_margin=0.002,
                accent_margin=0.004,
                max_score_drop=3.00,
                max_self_drop=0.05,
                max_feature_drop=0.22,
            ):
                self._save_best(frontier_head["voice"], frontier_results, i + 1, results_dir)
                best_target_seen = max(best_target_seen, frontier_target)
                best_accent_seen = max(best_accent_seen, frontier_accent)
                progress_write(
                    f"Frontier snapshot at step {i + 1}: {self._score_summary(frontier_results)}"
                )

            if stagnation_steps >= self.stagnation_limit:
                restart_seed = random.choice(archive)["voice"] if archive else random.choice(candidate_voices)
                mean_latent = self._project_voice_to_latent(restart_seed, center, basis)
                diag_scale = np.clip(diag_scale * 1.15, 0.35, 4.0)
                sigma = min(2.5, max(0.10, sigma * 1.35))
                restart_head = max(elites, key=self._sort_key)
                restart_results = restart_head["results"]
                if self._accept_frontier(
                    restart_results,
                    best_results,
                    target_margin=0.002,
                    accent_margin=0.004,
                    max_score_drop=3.00,
                    max_self_drop=0.05,
                    max_feature_drop=0.22,
                ):
                    best_voice = restart_head["voice"]
                    best_results = restart_results
                    self._save_best(best_voice, best_results, i + 1, results_dir)
                    best_target_seen = max(best_target_seen, float(best_results.get("target_similarity", 0.0)))
                    best_accent_seen = max(best_accent_seen, float(best_results.get("accent_similarity", 0.0)))
                progress_write(
                    f"Restart triggered at step {i + 1}: sigma={sigma:.4f}, "
                    f"elite head {self._score_summary(max(elites, key=self._sort_key)['results'])}"
                )
                stagnation_steps = 0

            if not interactive and ((i + 1) % log_interval == 0 or i == step_limit - 1):
                completed = i + 1
                elapsed = time.time() - started_at
                eta = (elapsed / completed) * (step_limit - completed) if completed > 0 else 0.0
                print(
                    f"Progress: step {completed}/{step_limit} | "
                    f"best_score={best_results.get('score', 0.0):.2f} | "
                    f"best_target_sim={best_results.get('target_similarity', 0.0):.3f} | "
                    f"best_accent_sim={best_results.get('accent_similarity', 0.0):.3f} | "
                    f"stagnation={stagnation_steps}/{self.stagnation_limit} | "
                    f"beam={len(samples)} | eval={full_eval_budget} | "
                    f"w=t:{active_weights[0]:.2f}/s:{active_weights[1]:.2f}/f:{active_weights[2]:.2f}/a:{active_weights[3]:.2f} | "
                    f"elapsed={elapsed:.1f}s | "
                    f"eta={eta:.1f}s"
                )

        elapsed = time.time() - started_at
        print(f"CMA-ES Final Results for {self.output_name}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Best Score: {best_results['score']:.2f}_")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}_")
        print(f"Best Accent Similarity: {best_results.get('accent_similarity', 0.0):.2f}_")
        print(f"CMA-ES pt and wav files ---> {results_dir}")

    @staticmethod
    def _weighted_harmonic(values: list[float], weights: list[float]) -> float:
        weighted_values = []
        weighted_weights = []
        for value, weight in zip(values, weights):
            if weight <= 0.0:
                continue
            weighted_values.append(max(float(value), 1e-6))
            weighted_weights.append(float(weight))

        if not weighted_values:
            return 0.0

        w = np.array(weighted_weights, dtype=np.float64)
        v = np.array(weighted_values, dtype=np.float64)
        return float(np.sum(w) / np.sum(w / v))



