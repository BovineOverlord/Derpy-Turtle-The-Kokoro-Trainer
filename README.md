# Derpy Turtle: The Kokoro Trainer

Derpy Turtle is a Windows-first GUI for building better local Kokoro voice outputs. It combines Kokoro voice search, target-audio scoring, RVC model training, and post-generation voice conversion into one queue-based workflow.

The practical goal is simple: generate clear Kokoro speech, then use a target-trained RVC model to move the final audio closer to the desired voice.

## This project wouldn't be possible without…

- **[Kokoro](https://github.com/hexgrad/kokoro)** — the TTS model and voice tensor system that everything here is built around.
- **[kvoicewalk](https://github.com/RobViren/kvoicewalk)** — the original Kokoro random-walk voice search that this project grew out of.
- **[RVC — Retrieval-based Voice Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** — the voice conversion technique used to push generated speech toward a target voice identity.
- **[rvc-python](https://github.com/daswer123/rvc-python)** — the Python wrapper that makes RVC straightforward to integrate.
- **[Applio](https://github.com/IAHispano/Applio)** — the RVC training backend used by `Train Target RVC Model`.
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — the transcription engine powering `Transcribe Many` (built on OpenAI's Whisper).
- **[resemblyzer](https://github.com/resemble-ai/resemblyzer)** — speaker embedding and voice similarity scoring used to evaluate candidate voices against your target.
- **[librosa](https://librosa.org)** — audio analysis and resampling used throughout the audio processing pipeline.
- **[so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork)** — the alternative voice conversion backend (`sovits` mode).
- **[PyTorch](https://pytorch.org)** — the ML backbone under all of the above.

## What It Does

- Runs Kokoro random-walk and hybrid voice searches against one or more target clips.
- Trains a target RVC model from your own clean reference audio.
- Applies RVC automatically after generation and writes a final `_rvc.wav`.
- Provides a queue GUI with presets, ETA, progress logging, text mapping for extra audio clips, and playback for generated WAV files.
- Bootstraps its own Python environment from a single launcher executable.

## Recommended Workflow

1. Train an RVC model from the target voice.
2. Run a short Kokoro search or refinement to get stable source speech.
3. Enable `Use Latest RVC`.
4. Generate and listen to the `_rvc.wav` output.

The optimizer score is measured before RVC conversion. If the `_rvc.wav` sounds better, trust the audio over the pre-RVC score.

## First Run

Run:

```powershell
.\derpy-turtle-kokoro-trainer.exe
```

On first launch, the executable creates `.venv`, installs the Python dependencies, prepares the selected voice-conversion backend, and opens the GUI.

Setup logs are written to:

```text
derpy-turtle-launcher.log
```

## Requirements

- Windows 10/11.
- Python 3.10, 3.11, or 3.12 available through `py` or `python`.
- NVIDIA GPU recommended.
- CUDA-capable PyTorch is installed for the RVC backend when available.
- CPU mode works, but long searches and RVC conversion are much slower.

Observed local performance: CUDA mode used about 4 GB VRAM and reduced a run from roughly 26 hours on CPU to about 4 hours on an RTX 3060.

## GUI Modes

`Random Walk`

Searches Kokoro voice tensors and writes `.pt` plus `.wav` candidates.

`Train Target RVC Model`

Builds an RVC model from the primary target audio plus any extra target clips. The result is exported under `vc_models/rvc/trained/<model_name>/` and becomes available through `Use Latest RVC`.

`Test Voice`

Generates a quick WAV from a selected `.pt` voice and the current target text. Output is written to `out/<output_name>_test.wav`.

`Transcribe Many`

Transcribes a file or folder of audio clips to text files.

`Export Voices Bin`

Exports source voice data for faster startup paths.

## RVC Training

Use `Train Target RVC Model` when random walk has plateaued or when voice identity matters more than the Kokoro similarity score.

Recommended starting settings:

```text
RVC Epochs: 250-350
RVC Batch: 4
RVC Sample Rate: 48000
Prepare Dataset Only: off
```

Data quality matters more than huge step counts:

- Use 10-30 minutes of clean target speech as a minimum.
- 45-90 minutes is better when the style is consistent.
- Remove music, heavy reverb, clipping, background speakers, and noisy sections.
- Include the emotional range, pitch range, pace, and accent you want the model to reproduce.

After training, click `Use Latest RVC` before adding the generation task.

## Kokoro Search Settings

For RVC-based output, do not spend days chasing a higher random-walk score. The Kokoro voice is now the source performance, not the final identity.

Good defaults:

```text
Preset: Fast Iterate or Similarity Recovery
Steps: 500-1500 for source checks
Device: cuda
Post VC: enabled
VC: Use Latest RVC
```

Run longer searches only when the pre-RVC source has pronunciation, pacing, stability, or accent problems.

## Outputs

Random-walk and hybrid runs write result folders under:

```text
out/
```

Typical files:

```text
*.pt          Kokoro voice tensor candidate
*.wav         pre-RVC Kokoro output
*_rvc.wav     final RVC-converted output
```

Use the `_rvc.wav` as the final audio when RVC is enabled.

## Extra Target Audio

Extra target clips can improve scoring and RVC training. Each extra clip can have its own transcript through `Map Texts`.

Different text is useful. Multiple clips with different words, pacing, and emotion give the trainer more information than repeating the same phrase.

## Playback

Use `Play Latest WAV` in the GUI to open the newest generated WAV under `out` with the Windows default audio player.

## Building The Launcher

To rebuild the launcher executable:

```powershell
.\build-launcher.cmd
```

The output is:

```text
derpy-turtle-kokoro-trainer.exe
```

## Safety

Only train on and clone voices you have permission to use. Do not use this project to impersonate people without consent.
