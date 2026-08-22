from __future__ import annotations
import json
import os
import re


import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import scrolledtext, ttk


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MASCOT_PATH = ASSETS_DIR / "derpyturtle.jpg"
VC_BACKEND_CONFIG_PATH = PROJECT_ROOT / "vc-backend.txt"
VC_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "vc-runtime.json"
DEFAULT_RVC_PYTHON_PATH = PROJECT_ROOT / ".venv_vc_rvc" / "Scripts" / "python.exe"
DEFAULT_RVC_MODEL_PATH = PROJECT_ROOT / "vc_models" / "rvc" / "default_abe_shinzo" / "AbeShinzo2.pth"
DEFAULT_RVC_INDEX_PATH = PROJECT_ROOT / "vc_models" / "rvc" / "default_abe_shinzo" / "added_IVF429_Flat_nprobe_6.index"
TRAINED_RVC_MODEL_ROOT = PROJECT_ROOT / "vc_models" / "rvc" / "trained"

# -- colour palette --
CLR_BG = "#1e1e2e"
CLR_BG_LIGHT = "#2a2a3d"
CLR_FG = "#cdd6f4"
CLR_FG_DIM = "#a6adc8"
CLR_ACCENT = "#89b4fa"
CLR_GREEN = "#a6e3a1"
CLR_GREEN_DARK = "#40a02b"
CLR_RED = "#f38ba8"
CLR_SURFACE = "#313244"
CLR_SURFACE2 = "#45475a"
CLR_YELLOW = "#f9e2af"
CLR_OVERLAY = "#585b70"
CLR_BORDER = "#3a3a52"
CLR_ACCENT_HOVER = "#a6c8ff"
CLR_GREEN_HOVER = "#94e2a3"
DEFAULT_OTHER_TEXT = (
    "If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, "
    "the resulting eruption is both a visual and tactile delight, often used in "
    "classrooms to simulate volcanic activity on a miniature scale."
)

MODE_RANDOM_WALK = "Random Walk"
MODE_TRAIN_RVC = "Train Target RVC Model"
MODE_TEST_VOICE = "Test Voice"
MODE_TRANSCRIBE_MANY = "Transcribe Many"
MODE_EXPORT_BIN = "Export Voices Bin"
MODES = [MODE_RANDOM_WALK, MODE_TRAIN_RVC, MODE_TEST_VOICE, MODE_TRANSCRIBE_MANY, MODE_EXPORT_BIN]

SETTING_PRESETS: dict[str, dict[str, str | bool]] = {
    "Balanced (Default)": {
        "population_limit": "10",
        "step_limit": "10000",
        "log_interval": "10",
        "elite_size": "4",
        "stagnation_limit": "250",
        "restart_diversity": "0.35",
        "candidates_per_step": "3",
        "max_candidates_per_step": "8",
        "weight_target": "0.45",
        "weight_self": "0.33",
        "weight_feature": "0.10",
        "weight_accent": "0.12",
        "dynamic_schedule": True,
        "adaptive_beam": True,
        "auto_extra": True,
        "auto_extra_limit": "1",
        "optimizer": "hybrid",
        "refine_top_k": "4",
        "cma_sigma": "0.30",
        "cma_latent_dim": "16",
        "pareto_archive_size": "32",
    },
    "Fast Iterate": {
        "population_limit": "8",
        "step_limit": "3000",
        "log_interval": "10",
        "elite_size": "3",
        "stagnation_limit": "140",
        "restart_diversity": "0.30",
        "candidates_per_step": "2",
        "max_candidates_per_step": "4",
        "weight_target": "0.48",
        "weight_self": "0.30",
        "weight_feature": "0.12",
        "weight_accent": "0.10",
        "dynamic_schedule": False,
        "adaptive_beam": True,
        "auto_extra": False,
        "auto_extra_limit": "0",
        "optimizer": "hybrid",
        "refine_top_k": "3",
        "cma_sigma": "0.38",
        "cma_latent_dim": "10",
        "pareto_archive_size": "20",
    },
    "Accent Focus": {
        "population_limit": "10",
        "step_limit": "8000",
        "log_interval": "10",
        "elite_size": "4",
        "stagnation_limit": "220",
        "restart_diversity": "0.22",
        "candidates_per_step": "2",
        "max_candidates_per_step": "5",
        "weight_target": "0.52",
        "weight_self": "0.18",
        "weight_feature": "0.08",
        "weight_accent": "0.22",
        "dynamic_schedule": False,
        "adaptive_beam": True,
        "auto_extra": False,
        "auto_extra_limit": "0",
        "optimizer": "hybrid",
        "refine_top_k": "4",
        "cma_sigma": "0.30",
        "cma_latent_dim": "16",
        "pareto_archive_size": "28",
    },
    "Refine From Best": {
        "population_limit": "6",
        "step_limit": "4000",
        "log_interval": "10",
        "elite_size": "3",
        "stagnation_limit": "300",
        "restart_diversity": "0.15",
        "candidates_per_step": "2",
        "max_candidates_per_step": "4",
        "weight_target": "0.45",
        "weight_self": "0.15",
        "weight_feature": "0.07",
        "weight_accent": "0.33",
        "dynamic_schedule": False,
        "adaptive_beam": True,
        "auto_extra": False,
        "auto_extra_limit": "0",
        "optimizer": "cma_es",
        "refine_top_k": "5",
        "cma_sigma": "0.20",
        "cma_latent_dim": "20",
        "pareto_archive_size": "36",
    },
    "Similarity Recovery": {
        "population_limit": "6",
        "step_limit": "2500",
        "log_interval": "10",
        "elite_size": "3",
        "stagnation_limit": "120",
        "restart_diversity": "0.15",
        "candidates_per_step": "3",
        "max_candidates_per_step": "7",
        "weight_target": "0.62",
        "weight_self": "0.18",
        "weight_feature": "0.10",
        "weight_accent": "0.10",
        "dynamic_schedule": False,
        "adaptive_beam": True,
        "auto_extra": False,
        "auto_extra_limit": "0",
        "optimizer": "hybrid",
        "refine_top_k": "8",
        "cma_sigma": "0.14",
        "cma_latent_dim": "12",
        "pareto_archive_size": "28",
    },
}

AUTO_RVC_TEMPLATE_TOKEN = "__AUTO_RVC__"
AUTO_SOVITS_TEMPLATE_TOKEN = "__AUTO_SOVITS__"

VC_TEMPLATE_PRESETS: dict[str, tuple[str, str]] = {
    "Custom (Keep Current)": ("", "_vc"),
    "RVC (Auto from Launcher)": (
        AUTO_RVC_TEMPLATE_TOKEN,
        "_rvc",
    ),
    "SoVITS (Auto from Launcher)": (
        AUTO_SOVITS_TEMPLATE_TOKEN,
        "_sovits",
    ),
    "RVC CLI (Manual Template)": (
        'python infer.py --input "{input_wav}" --output "{output_wav}" --model "path/to/model.pth" --index "path/to/model.index"',
        "_rvc",
    ),
    "SoVITS CLI (Manual Template)": (
        'python inference_main.py --input "{input_wav}" --output "{output_wav}" --model "path/to/sovits_model.pth"',
        "_sovits",
    ),
}

PROGRESS_RE = re.compile(
    r"Progress:\s*step\s+(?P<step>\d+)/(?P<total>\d+)\s*\|.*?elapsed=(?P<elapsed>[0-9.]+)s(?:\s*\|\s*eta=(?P<eta>[0-9.]+)s)?"
)
RESULT_DIR_RE = re.compile(
    r"(?:Random Walk|CMA-ES)\s+pt and wav files\s*--->\s*(?P<path>.+)$"
)


@dataclass
class Task:
    task_id: int
    mode: str
    args: list[str]
    summary: str
    script: str = "main.py"
    status: str = "Queued"
    step_limit: int | None = None
    post_vc_enabled: bool = False
    post_vc_command: str = ""
    post_vc_suffix: str = "_vc"


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, wrap_length: int = 420):
        self.widget = widget
        self.text = text
        self.wrap_length = wrap_length
        self.tip_window: tk.Toplevel | None = None

        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tip_window or not self.text:
            return

        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background=CLR_YELLOW,
            foreground=CLR_BG,
            padx=8,
            pady=4,
            wraplength=self.wrap_length,
            font=("Segoe UI", 9),
        )
        label.pack()
        self.tip_window = tip

    def _hide(self, _event=None) -> None:
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class KVoiceWalkGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Derpy Turtle: The Kokoro Trainer")
        self.root.geometry("1320x920")          # restored-down size
        self.root.minsize(1100, 720)
        try:
            self.root.state("zoomed")           # open maximised on Windows
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)  # fallback for some X11/Linux WMs
            except tk.TclError:
                pass
        self.root.configure(bg=CLR_BG)
        self.root.option_add("*Font", "TkDefaultFont")

        self._setup_theme()

        self.tasks: list[Task] = []
        self.next_task_id = 1
        self.runner_thread: threading.Thread | None = None
        self.current_process: subprocess.Popen | None = None

        self.stop_requested = False
        self.events: queue.Queue[tuple] = queue.Queue()

        self.mode_var = tk.StringVar(value=MODE_RANDOM_WALK)
        self.device_var = tk.StringVar(value="cuda")
        self.preset_var = tk.StringVar(value="Balanced (Default)")
        self.target_audio_var = tk.StringVar(value="")
        self.target_audio_many_var = tk.StringVar(value="")
        self.target_text_many_map: dict[str, str] = {}
        self.target_text_many_window: tk.Toplevel | None = None
        self.target_text_many_widgets: dict[str, scrolledtext.ScrolledText] = {}
        self.target_text_many_button_label = tk.StringVar(value="Map Texts (0/0)")
        self.auto_target_audio_many_var = tk.BooleanVar(value=True)
        self.auto_target_audio_many_limit_var = tk.StringVar(value="1")

        self.voice_folder_var = tk.StringVar(value=str((PROJECT_ROOT / "voices").resolve()))
        self.starting_voice_var = tk.StringVar(value="")
        self.test_voice_var = tk.StringVar(value="")
        self.transcribe_many_var = tk.StringVar(value="")
        self.output_name_var = tk.StringVar(value="my_new_voice")
        self.post_vc_enabled_var = tk.BooleanVar(value=False)
        self.post_vc_preset_var = tk.StringVar(value="Custom (Keep Current)")
        self.post_vc_command_var = tk.StringVar(value="")
        self.post_vc_suffix_var = tk.StringVar(value="_vc")
        self.vc_train_epochs_var = tk.StringVar(value="250")
        self.vc_train_batch_size_var = tk.StringVar(value="4")
        self.vc_train_sample_rate_var = tk.StringVar(value="48000")
        self.vc_train_applio_root_var = tk.StringVar(value=str((PROJECT_ROOT / "vc_train_backends" / "Applio").resolve()))
        self.vc_train_prepare_only_var = tk.BooleanVar(value=False)
        self.population_limit_var = tk.StringVar(value="10")
        self.step_limit_var = tk.StringVar(value="10000")
        self.log_interval_var = tk.StringVar(value="10")

        self.elite_size_var = tk.StringVar(value="4")
        self.stagnation_limit_var = tk.StringVar(value="250")
        self.restart_diversity_var = tk.StringVar(value="0.35")
        self.candidates_per_step_var = tk.StringVar(value="3")
        self.max_candidates_per_step_var = tk.StringVar(value="8")

        self.score_weight_target_var = tk.StringVar(value="0.45")
        self.score_weight_self_var = tk.StringVar(value="0.33")
        self.score_weight_feature_var = tk.StringVar(value="0.10")
        self.score_weight_accent_var = tk.StringVar(value="0.12")
        self.dynamic_weight_schedule_var = tk.BooleanVar(value=True)
        self.adaptive_beam_var = tk.BooleanVar(value=True)
        self.optimizer_var = tk.StringVar(value="hybrid")
        self.refine_top_k_var = tk.StringVar(value="4")
        self.cma_sigma_var = tk.StringVar(value="0.30")
        self.cma_latent_dim_var = tk.StringVar(value="16")
        self.pareto_archive_size_var = tk.StringVar(value="32")

        self.interpolate_start_var = tk.BooleanVar(value=False)
        self.transcribe_start_var = tk.BooleanVar(value=False)

        self._tooltips: list[ToolTip] = []
        self.target_audio_many_var.trace_add("write", self._on_target_audio_many_changed)

        self._build_ui()
        self._apply_launcher_vc_backend_default(auto_apply=True)
        self._on_target_audio_many_changed()
        self._append_log(f"Project root: {PROJECT_ROOT}")
        self._append_log(f"Python: {sys.executable}")
        self.root.after(100, self._poll_events)

    def _bind_tooltip(self, widget: tk.Widget, text: str) -> None:
        self._tooltips.append(ToolTip(widget, text))

    def _setup_theme(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        base_font = ("Segoe UI", 10)

        # Global defaults. focuscolor matches the background so the default
        # dotted focus rectangle disappears -- focus is shown via accent borders.
        style.configure(".", background=CLR_BG, foreground=CLR_FG, fieldbackground=CLR_SURFACE,
                        bordercolor=CLR_BORDER, troughcolor=CLR_SURFACE, selectbackground=CLR_ACCENT,
                        selectforeground=CLR_BG, font=base_font, focuscolor=CLR_BG)

        style.configure("TFrame", background=CLR_BG)
        style.configure("TLabel", background=CLR_BG, foreground=CLR_FG)

        # Section panels: flat with a thin border and a bold accent title.
        style.configure("TLabelframe", background=CLR_BG, foreground=CLR_ACCENT,
                        bordercolor=CLR_BORDER, relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=CLR_BG, foreground=CLR_ACCENT,
                        font=("Segoe UI", 11, "bold"))

        # Inputs: flattened bevels + an accent focus ring.
        for field in ("TEntry", "TCombobox", "TSpinbox"):
            style.configure(field, fieldbackground=CLR_SURFACE, foreground=CLR_FG,
                            insertcolor=CLR_FG, bordercolor=CLR_BORDER, lightcolor=CLR_BORDER,
                            darkcolor=CLR_BORDER, arrowcolor=CLR_ACCENT, padding=5)
            style.map(field,
                      bordercolor=[("focus", CLR_ACCENT), ("hover", CLR_OVERLAY)],
                      lightcolor=[("focus", CLR_ACCENT)],
                      darkcolor=[("focus", CLR_ACCENT)])

        style.configure("TCombobox", background=CLR_SURFACE2)
        style.map("TCombobox",
                  fieldbackground=[("readonly", CLR_SURFACE)],
                  foreground=[("readonly", CLR_FG)],
                  selectbackground=[("readonly", CLR_SURFACE)],
                  selectforeground=[("readonly", CLR_FG)])

        # Buttons: flat, generously padded. Secondary (default) style.
        style.configure("TButton", background=CLR_SURFACE2, foreground=CLR_FG,
                        bordercolor=CLR_SURFACE2, lightcolor=CLR_SURFACE2, darkcolor=CLR_SURFACE2,
                        relief="flat", padding=(12, 7), font=("Segoe UI", 9))
        style.map("TButton",
                  background=[("active", CLR_OVERLAY), ("pressed", CLR_OVERLAY)],
                  foreground=[("active", CLR_FG), ("pressed", CLR_FG)])

        # Primary (accent) button style.
        style.configure("Accent.TButton", background=CLR_ACCENT, foreground=CLR_BG,
                        bordercolor=CLR_ACCENT, lightcolor=CLR_ACCENT, darkcolor=CLR_ACCENT,
                        font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton",
                  background=[("active", CLR_ACCENT_HOVER), ("pressed", CLR_ACCENT_HOVER)],
                  foreground=[("active", CLR_BG), ("pressed", CLR_BG)])

        # Destructive (stop) button style.
        style.configure("Danger.TButton", background=CLR_SURFACE2, foreground=CLR_RED,
                        bordercolor=CLR_SURFACE2, lightcolor=CLR_SURFACE2, darkcolor=CLR_SURFACE2,
                        font=("Segoe UI", 9, "bold"))
        style.map("Danger.TButton",
                  background=[("active", CLR_RED), ("pressed", CLR_RED)],
                  foreground=[("active", CLR_BG), ("pressed", CLR_BG)])

        style.configure("TCheckbutton", background=CLR_BG, foreground=CLR_FG,
                        indicatorcolor=CLR_SURFACE, indicatorrelief="flat", focuscolor=CLR_BG)
        style.map("TCheckbutton",
                  indicatorcolor=[("selected", CLR_ACCENT), ("active", CLR_SURFACE2)],
                  foreground=[("active", CLR_ACCENT)],
                  background=[("active", CLR_BG)])

        style.configure("Treeview", background=CLR_SURFACE, foreground=CLR_FG,
                        fieldbackground=CLR_SURFACE, bordercolor=CLR_BORDER,
                        font=("Segoe UI", 9), rowheight=28, relief="flat")
        style.configure("Treeview.Heading", background=CLR_BG_LIGHT, foreground=CLR_ACCENT,
                        font=("Segoe UI", 9, "bold"), bordercolor=CLR_BORDER,
                        relief="flat", padding=(6, 6))
        style.map("Treeview", background=[("selected", CLR_ACCENT)],
                  foreground=[("selected", CLR_BG)])
        style.map("Treeview.Heading", background=[("active", CLR_SURFACE2)])

        style.configure("Vertical.TScrollbar", background=CLR_SURFACE2, troughcolor=CLR_BG,
                        bordercolor=CLR_BG, arrowcolor=CLR_FG_DIM, relief="flat", arrowsize=13)
        style.configure("Horizontal.TScrollbar", background=CLR_SURFACE2, troughcolor=CLR_BG,
                        bordercolor=CLR_BG, arrowcolor=CLR_FG_DIM, relief="flat", arrowsize=13)
        style.map("Vertical.TScrollbar", background=[("active", CLR_OVERLAY)])
        style.map("Horizontal.TScrollbar", background=[("active", CLR_OVERLAY)])

    def _apply_preset(self) -> None:
        name = self.preset_var.get().strip()
        preset = SETTING_PRESETS.get(name)
        if not preset:
            return

        self.population_limit_var.set(str(preset["population_limit"]))
        self.step_limit_var.set(str(preset["step_limit"]))
        self.log_interval_var.set(str(preset["log_interval"]))
        self.elite_size_var.set(str(preset["elite_size"]))
        self.stagnation_limit_var.set(str(preset["stagnation_limit"]))
        self.restart_diversity_var.set(str(preset["restart_diversity"]))
        self.candidates_per_step_var.set(str(preset["candidates_per_step"]))
        self.max_candidates_per_step_var.set(str(preset["max_candidates_per_step"]))

        self.score_weight_target_var.set(str(preset["weight_target"]))
        self.score_weight_self_var.set(str(preset["weight_self"]))
        self.score_weight_feature_var.set(str(preset["weight_feature"]))
        self.score_weight_accent_var.set(str(preset["weight_accent"]))

        self.dynamic_weight_schedule_var.set(bool(preset["dynamic_schedule"]))
        self.adaptive_beam_var.set(bool(preset["adaptive_beam"]))
        self.auto_target_audio_many_var.set(bool(preset["auto_extra"]))
        self.auto_target_audio_many_limit_var.set(str(preset["auto_extra_limit"]))
        self.optimizer_var.set(str(preset["optimizer"]))
        self.refine_top_k_var.set(str(preset["refine_top_k"]))
        self.cma_sigma_var.set(str(preset["cma_sigma"]))
        self.cma_latent_dim_var.set(str(preset["cma_latent_dim"]))
        self.pareto_archive_size_var.set(str(preset["pareto_archive_size"]))

        self._append_log(f"Applied preset: {name}")

    def _install_tooltips(self, config_frame: ttk.LabelFrame) -> None:
        tooltip_cells: list[tuple[tuple[int, int], str]] = [
            ((0, 0), "Mode selects which operation this queued task will run."),
            ((0, 1), "Mode selects which operation this queued task will run."),
            ((0, 2), "Device for synthesis and scoring. Use cuda for NVIDIA GPU acceleration."),
            ((0, 3), "Device for synthesis and scoring. Use cuda for NVIDIA GPU acceleration."),
            ((0, 4), "Output base name used for saved pt/wav results."),
            ((0, 5), "Output base name used for saved pt/wav results."),
            ((1, 0), "Primary reference audio clip used as the main target."),
            ((1, 1), "Primary reference audio clip used as the main target."),
            ((2, 0), "Additional reference clips for multi-target scoring."),
            ((2, 1), "Additional reference clips for multi-target scoring."),
            ((2, 4), "Browse and append more extra target clips."),
            ((2, 5), "Open one-to-one transcript mapping for each extra target clip."),
            ((3, 0), "Transcript for the primary target audio."),
            ((3, 1), "Transcript for the primary target audio."),
            ((4, 0), "Self-consistency prompt to keep generated voice stable across text."),
            ((4, 1), "Self-consistency prompt to keep generated voice stable across text."),
            ((5, 0), "Folder containing source .pt voices used as search seeds."),
            ((6, 0), "Optional starting .pt voice for refinement runs."),
            ((9, 0), "How many initial voices are evaluated for the starting pool."),
            ((9, 2), "Maximum random-walk steps for this task."),
            ((10, 0), "Emit one progress log line every N steps."),
            ((10, 2), "How many best candidates are kept as elites."),
            ((10, 4), "Restart trigger after N non-improving steps."),
            ((11, 0), "Mutation strength used during restart injections."),
            ((11, 2), "Base beam width: candidates scored per step."),
            ((11, 4), "If enabled, score weights shift over training progress/stagnation."),
            ((12, 0), "Weight for target speaker similarity in final score."),
            ((12, 2), "Weight for self-consistency/stability."),
            ((13, 0), "Weight for broad acoustic feature similarity."),
            ((13, 2), "Weight for accent/prosody similarity."),
            ((13, 4), "Auto-add nearby similar files as extra targets when no manual extras are set."),
            ((13, 5), "Maximum number of auto-added nearby extra targets."),
            ((14, 0), "Adaptive beam increases candidate count during stagnation."),
            ((14, 2), "Upper bound used by Adaptive Beam."),
            ((14, 3), "Upper bound used by Adaptive Beam."),
            ((15, 0), "Optimizer mode: random_walk, cma_es, or hybrid warmup+refine."),
            ((15, 1), "Optimizer mode: random_walk, cma_es, or hybrid warmup+refine."),
            ((15, 2), "Top proxy candidates receiving full expensive scoring per step."),
            ((15, 3), "Top proxy candidates receiving full expensive scoring per step."),
            ((15, 4), "Initial CMA-ES latent sampling scale."),
            ((15, 5), "Initial CMA-ES latent sampling scale."),
            ((16, 0), "PCA latent dimensions used for CMA-ES search."),
            ((16, 1), "PCA latent dimensions used for CMA-ES search."),
            ((16, 2), "Pareto archive size used for multi-objective retention."),
            ((16, 3), "Pareto archive size used for multi-objective retention."),
            ((17, 0), "Preset selector for common tuning profiles."),
            ((17, 1), "Preset selector for common tuning profiles."),
            ((17, 3), "Apply selected preset values to the configuration fields."),
            ((18, 0), "Run voice conversion after successful Random Walk output."),
            ((18, 2), "Command template for VC stage. Use {input_wav} and {output_wav}."),
            ((18, 3), "Command template for VC stage. Use {input_wav} and {output_wav}."),
            ((18, 5), "Show VC template placeholders and usage examples."),
            ((19, 0), "Suffix appended to VC output filename (for example _rvc)."),
            ((19, 1), "Suffix appended to VC output filename (for example _rvc)."),
            ((19, 2), "Select a VC preset (RVC auto is one-click when launcher setup is complete)."),
            ((19, 3), "Select a VC preset (RVC auto is one-click when launcher setup is complete)."),
            ((19, 5), "Apply selected VC preset into command and suffix fields."),
            ((20, 0), "Epochs for target RVC training. Start around 200-400 with clean data."),
            ((20, 2), "Training batch size. Use 3-4 on lower VRAM, 6-8 on larger GPUs."),
            ((20, 4), "Target sample rate for the trained RVC model."),
            ((21, 0), "Applio backend folder containing core.py and env\\python.exe."),
            ((21, 2), "Only prepare the target dataset; do not start training."),
            ((22, 0), "Add current configuration as a queued task."),
            ((22, 3), "Run queued tasks in order."),
        ]

        seen: set[str] = set()
        for (row, col), text in tooltip_cells:
            for widget in config_frame.grid_slaves(row=row, column=col):
                key = str(widget)
                if key in seen:
                    continue
                self._bind_tooltip(widget, text)
                seen.add(key)
    def _style_text_widget(self, widget: tk.Text | scrolledtext.ScrolledText) -> None:
        widget.configure(bg=CLR_SURFACE, fg=CLR_FG, insertbackground=CLR_FG,
                         selectbackground=CLR_ACCENT, selectforeground=CLR_BG,
                         relief="flat", borderwidth=2, highlightthickness=1,
                         highlightcolor=CLR_ACCENT, highlightbackground=CLR_SURFACE2,
                         font=("Segoe UI", 10))

    def _load_mascot(self) -> tk.PhotoImage | None:
        if not MASCOT_PATH.exists():
            return None
        try:
            from PIL import Image, ImageTk
            img = Image.open(MASCOT_PATH)
            target_h = 44
            ratio = target_h / img.height
            new_size = (int(img.width * ratio), target_h)
            img = img.resize(new_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            return photo  # type: ignore[return-value]
        except Exception:
            return None

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.rowconfigure(4, weight=1)

        # -- header bar --
        self._mascot_photo = self._load_mascot()
        header_h = 72 if not self._mascot_photo else max(72, self._mascot_photo.height() + 24)
        header = tk.Frame(self.root, bg=CLR_BG_LIGHT, height=header_h)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        header.columnconfigure(0, weight=1)

        title_box = tk.Frame(header, bg=CLR_BG_LIGHT)
        title_box.grid(row=0, column=0, sticky="w", padx=18, pady=12)

        title_lbl = tk.Label(title_box, text="Derpy Turtle: The Kokoro Trainer", font=("Segoe UI", 17, "bold"),
                             bg=CLR_BG_LIGHT, fg=CLR_ACCENT)
        title_lbl.grid(row=0, column=0, sticky="w")

        subtitle_lbl = tk.Label(title_box, text="Kokoro Search  ·  RVC Training  ·  Voice Conversion",
                                font=("Segoe UI", 10), bg=CLR_BG_LIGHT, fg=CLR_FG_DIM)
        subtitle_lbl.grid(row=1, column=0, sticky="w", pady=(3, 0))

        if self._mascot_photo:
            mascot_lbl = tk.Label(header, image=self._mascot_photo, bg=CLR_BG_LIGHT)
            mascot_lbl.grid(row=0, column=1, sticky="e", padx=(0, 18), pady=8)

        # accent underline pinned to the bottom of the header
        tk.Frame(header, bg=CLR_ACCENT, height=2).place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)

        config_frame = ttk.LabelFrame(self.root, text="Task Configuration", padding=10)
        config_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(6, 10))
        for col in range(6):
            config_frame.columnconfigure(col, weight=1)

        ttk.Label(config_frame, text="Mode").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(config_frame, textvariable=self.mode_var, values=MODES, state="readonly")
        mode_combo.grid(row=0, column=1, sticky="ew", padx=(4, 8))

        ttk.Label(config_frame, text="Device").grid(row=0, column=2, sticky="w")
        device_combo = ttk.Combobox(
            config_frame,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
        )
        device_combo.grid(row=0, column=3, sticky="ew", padx=(4, 8))

        ttk.Label(config_frame, text="Output Name").grid(row=0, column=4, sticky="w")
        ttk.Entry(config_frame, textvariable=self.output_name_var).grid(row=0, column=5, sticky="ew", padx=(4, 0))

        ttk.Label(config_frame, text="Target Audio").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.target_audio_var).grid(row=1, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_target_audio).grid(row=1, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Extra Target Audios").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.target_audio_many_var).grid(row=2, column=1, columnspan=3, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse Many", command=self._browse_target_audio_many).grid(row=2, column=4, sticky="ew", pady=(8, 0))
        ttk.Button(config_frame, textvariable=self.target_text_many_button_label, command=self._open_target_text_many_editor).grid(row=2, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Target Text (Primary)").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        self.target_text_widget = scrolledtext.ScrolledText(config_frame, height=4, wrap=tk.WORD)
        self.target_text_widget.grid(row=3, column=1, columnspan=5, sticky="ew", padx=(4, 0), pady=(8, 0))
        self._style_text_widget(self.target_text_widget)

        ttk.Label(config_frame, text="Other Text").grid(row=4, column=0, sticky="nw", pady=(8, 0))
        self.other_text_widget = scrolledtext.ScrolledText(config_frame, height=3, wrap=tk.WORD)
        self.other_text_widget.grid(row=4, column=1, columnspan=5, sticky="ew", padx=(4, 0), pady=(8, 0))
        self._style_text_widget(self.other_text_widget)
        self.other_text_widget.insert("1.0", DEFAULT_OTHER_TEXT)

        ttk.Label(config_frame, text="Voice Folder").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.voice_folder_var).grid(row=5, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_voice_folder).grid(row=5, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Starting Voice (.pt)").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.starting_voice_var).grid(row=6, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_starting_voice).grid(row=6, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Test Voice (.pt)").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.test_voice_var).grid(row=7, column=1, columnspan=3, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_test_voice).grid(row=7, column=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Play Latest WAV", command=self._play_latest_wav).grid(row=7, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Transcribe Many Path").grid(row=8, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.transcribe_many_var).grid(row=8, column=1, columnspan=4, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_transcribe_many).grid(row=8, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(config_frame, text="Population Limit").grid(row=9, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=1000, textvariable=self.population_limit_var).grid(row=9, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Step Limit").grid(row=9, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=10000000, textvariable=self.step_limit_var).grid(row=9, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Checkbutton(config_frame, text="Interpolate Start", variable=self.interpolate_start_var).grid(row=9, column=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(config_frame, text="Transcribe Start", variable=self.transcribe_start_var).grid(row=9, column=5, sticky="w", pady=(8, 0))

        ttk.Label(config_frame, text="Log Every N Steps").grid(row=10, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=100000, textvariable=self.log_interval_var).grid(row=10, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Elite Size").grid(row=10, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=100, textvariable=self.elite_size_var).grid(row=10, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Stagnation Limit").grid(row=10, column=4, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=100000, textvariable=self.stagnation_limit_var).grid(row=10, column=5, sticky="ew", padx=(4, 0), pady=(8, 0))

        ttk.Label(config_frame, text="Restart Diversity").grid(row=11, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.restart_diversity_var).grid(row=11, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Candidates / Step").grid(row=11, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=32, textvariable=self.candidates_per_step_var).grid(row=11, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Checkbutton(
            config_frame,
            text="Dynamic Weight Schedule",
            variable=self.dynamic_weight_schedule_var,
        ).grid(row=11, column=4, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(config_frame, text="Weight Target").grid(row=12, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.score_weight_target_var).grid(row=12, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Weight Self").grid(row=12, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.score_weight_self_var).grid(row=12, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Weight Feature").grid(row=13, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.score_weight_feature_var).grid(row=13, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Weight Accent").grid(row=13, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.score_weight_accent_var).grid(row=13, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Checkbutton(
            config_frame,
            text="Auto add nearby target audios",
            variable=self.auto_target_audio_many_var,
        ).grid(row=13, column=4, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=0, to=20, textvariable=self.auto_target_audio_many_limit_var).grid(row=13, column=5, sticky="ew", padx=(4, 0), pady=(8, 0))

        ttk.Checkbutton(
            config_frame,
            text="Adaptive Beam",
            variable=self.adaptive_beam_var,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(config_frame, text="Max Beam").grid(row=14, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=64, textvariable=self.max_candidates_per_step_var).grid(row=14, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Optimizer").grid(row=15, column=0, sticky="w", pady=(8, 0))
        optimizer_combo = ttk.Combobox(
            config_frame,
            textvariable=self.optimizer_var,
            values=["hybrid", "cma_es", "random_walk"],
            state="readonly",
        )
        optimizer_combo.grid(row=15, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Refine Top-K").grid(row=15, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=64, textvariable=self.refine_top_k_var).grid(row=15, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="CMA Sigma").grid(row=15, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.cma_sigma_var).grid(row=15, column=5, sticky="ew", padx=(4, 0), pady=(8, 0))

        ttk.Label(config_frame, text="CMA Latent Dim").grid(row=16, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=2, to=128, textvariable=self.cma_latent_dim_var).grid(row=16, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Pareto Archive").grid(row=16, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=512, textvariable=self.pareto_archive_size_var).grid(row=16, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(config_frame, text="Preset").grid(row=17, column=0, sticky="w", pady=(8, 0))
        preset_combo = ttk.Combobox(
            config_frame,
            textvariable=self.preset_var,
            values=list(SETTING_PRESETS.keys()),
            state="readonly",
        )
        preset_combo.grid(row=17, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=(8, 0))
        preset_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_preset())
        ttk.Button(config_frame, text="Apply Preset", command=self._apply_preset).grid(
            row=17, column=3, sticky="ew", padx=(4, 8), pady=(8, 0)
        )

        ttk.Checkbutton(
            config_frame,
            text="Post VC (Command)",
            variable=self.post_vc_enabled_var,
        ).grid(row=18, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(config_frame, text="VC Command Template").grid(row=18, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.post_vc_command_var).grid(row=18, column=3, columnspan=2, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Button(config_frame, text="VC Help", command=self._show_post_vc_help).grid(row=18, column=5, sticky="ew", padx=(4, 0), pady=(8, 0))

        ttk.Label(config_frame, text="VC Output Suffix").grid(row=19, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.post_vc_suffix_var).grid(row=19, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Label(config_frame, text="VC Preset").grid(row=19, column=2, sticky="w", pady=(8, 0))
        vc_preset_combo = ttk.Combobox(
            config_frame,
            textvariable=self.post_vc_preset_var,
            values=list(VC_TEMPLATE_PRESETS.keys()),
            state="readonly",
        )
        vc_preset_combo.grid(row=19, column=3, columnspan=2, sticky="ew", padx=(4, 8), pady=(8, 0))
        vc_preset_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_post_vc_template_preset())
        ttk.Button(config_frame, text="Use Latest RVC", command=self._use_latest_trained_rvc).grid(
            row=19, column=5, sticky="ew", padx=(4, 0), pady=(8, 0)
        )

        ttk.Label(config_frame, text="RVC Epochs").grid(row=20, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=2000, textvariable=self.vc_train_epochs_var).grid(row=20, column=1, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Label(config_frame, text="RVC Batch").grid(row=20, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(config_frame, from_=1, to=32, textvariable=self.vc_train_batch_size_var).grid(row=20, column=3, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Label(config_frame, text="RVC Sample Rate").grid(row=20, column=4, sticky="w", pady=(8, 0))
        ttk.Combobox(
            config_frame,
            textvariable=self.vc_train_sample_rate_var,
            values=["32000", "40000", "48000"],
            state="readonly",
        ).grid(row=20, column=5, sticky="ew", padx=(4, 0), pady=(8, 0))

        ttk.Label(config_frame, text="Applio Root").grid(row=21, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.vc_train_applio_root_var).grid(row=21, column=1, columnspan=3, sticky="ew", padx=(4, 8), pady=(8, 0))
        ttk.Checkbutton(config_frame, text="Prepare Dataset Only", variable=self.vc_train_prepare_only_var).grid(row=21, column=4, sticky="w", pady=(8, 0))
        ttk.Button(config_frame, text="Browse", command=self._browse_applio_root).grid(row=21, column=5, sticky="ew", padx=(4, 0), pady=(8, 0))

        controls = ttk.Frame(config_frame)
        controls.grid(row=22, column=0, columnspan=6, sticky="ew", pady=(12, 0))
        controls.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        ttk.Button(controls, text="Add Task", command=self._add_task, style="Accent.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(controls, text="Remove Selected", command=self._remove_selected).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear Queue", command=self._clear_queue).grid(row=0, column=2, sticky="ew", padx=6)
        self.start_button = tk.Button(
            controls,
            text="▶  Start Queue",
            command=self._start_queue,
            bg=CLR_GREEN_DARK,
            fg="#ffffff",
            activebackground=CLR_GREEN,
            activeforeground=CLR_BG,
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
        )
        self.start_button.grid(row=0, column=3, sticky="ew", padx=6)
        self.start_button.bind("<Enter>", lambda _e: self.start_button.configure(bg=CLR_GREEN_HOVER, fg=CLR_BG))
        self.start_button.bind("<Leave>", lambda _e: self.start_button.configure(bg=CLR_GREEN_DARK, fg="#ffffff"))
        ttk.Button(controls, text="Stop Current", command=self._stop_queue, style="Danger.TButton").grid(row=0, column=4, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear Log", command=self._clear_log).grid(row=0, column=5, sticky="ew", padx=(6, 0))

        queue_frame = ttk.LabelFrame(self.root, text="Queued Tasks", padding=10)
        queue_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        self.task_tree = ttk.Treeview(
            queue_frame,
            columns=("id", "mode", "status", "eta", "summary"),
            show="headings",
            height=8,
        )
        self.task_tree.heading("id", text="ID")
        self.task_tree.heading("mode", text="Mode")
        self.task_tree.heading("status", text="Status")
        self.task_tree.heading("eta", text="ETA")
        self.task_tree.heading("summary", text="Summary")
        self.task_tree.column("id", width=60, anchor="center")
        self.task_tree.column("mode", width=140, anchor="center")
        self.task_tree.column("status", width=110, anchor="center")
        self.task_tree.column("eta", width=120, anchor="center")
        self.task_tree.column("summary", width=730, anchor="w")
        self.task_tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky="ns")

        # status-based row colours
        self.task_tree.tag_configure("row_queued", foreground=CLR_FG)
        self.task_tree.tag_configure("row_running", foreground=CLR_YELLOW)
        self.task_tree.tag_configure("row_done", foreground=CLR_GREEN)
        self.task_tree.tag_configure("row_failed", foreground=CLR_RED)
        self.task_tree.tag_configure("row_stopped", foreground=CLR_FG_DIM)

        log_frame = ttk.LabelFrame(self.root, text="Execution Log", padding=10)
        log_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_widget = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_widget.grid(row=0, column=0, sticky="nsew")
        self._style_text_widget(self.log_widget)
        self.log_widget.configure(font=("Consolas", 9), padx=8, pady=6)
        self.log_widget.tag_configure("log_error", foreground=CLR_RED)
        self.log_widget.tag_configure("log_success", foreground=CLR_GREEN)
        self.log_widget.tag_configure("log_warn", foreground=CLR_YELLOW)
        self._install_tooltips(config_frame)

    def _browse_target_audio(self) -> None:
        path = filedialog.askopenfilename(title="Select Target Audio", filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"), ("All Files", "*.*")])
        if path:
            self.target_audio_var.set(path)

    def _browse_target_audio_many(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select Additional Target Audios",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"), ("All Files", "*.*")],
        )
        if paths:
            existing = self._split_paths(self.target_audio_many_var.get())
            merged: list[str] = []
            seen: set[str] = set()
            for path in [*existing, *paths]:
                if path not in seen:
                    merged.append(path)
                    seen.add(path)

            self.target_audio_many_var.set(" | ".join(merged))
            self._on_target_audio_many_changed()

    def _browse_voice_folder(self) -> None:
        path = filedialog.askdirectory(title="Select Voice Folder")
        if path:
            self.voice_folder_var.set(path)

    def _browse_starting_voice(self) -> None:
        path = filedialog.askopenfilename(title="Select Starting Voice", filetypes=[("PyTorch", "*.pt"), ("All Files", "*.*")])
        if path:
            self.starting_voice_var.set(path)

    def _browse_test_voice(self) -> None:
        path = filedialog.askopenfilename(title="Select Test Voice", filetypes=[("PyTorch", "*.pt"), ("All Files", "*.*")])
        if path:
            self.test_voice_var.set(path)

    def _play_audio_file(self, path: Path) -> None:
        if not path.exists():
            messagebox.showwarning("Audio Not Found", f"Audio file not found:\n{path}")
            return

        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
            self._append_log(f"Opened audio: {path}")
        except Exception as exc:
            messagebox.showerror("Play Failed", f"Could not open audio file:\n{path}\n\n{exc}")

    def _play_latest_wav(self) -> None:
        out_dir = PROJECT_ROOT / "out"
        if not out_dir.exists():
            messagebox.showwarning("No Output", "No out folder exists yet.")
            return

        wav_files = [p for p in out_dir.rglob("*.wav") if p.is_file()]
        if not wav_files:
            messagebox.showwarning("No WAV Output", "No generated WAV files were found in the out folder.")
            return

        self._play_audio_file(max(wav_files, key=lambda p: p.stat().st_mtime))

    def _browse_transcribe_many(self) -> None:
        file_path = filedialog.askopenfilename(title="Select Audio File for --transcribe_many", filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")])
        if file_path:
            self.transcribe_many_var.set(file_path)
            return

        dir_path = filedialog.askdirectory(title="Or Select Folder for --transcribe_many")
        if dir_path:
            self.transcribe_many_var.set(dir_path)

    def _browse_applio_root(self) -> None:
        path = filedialog.askdirectory(title="Select Applio Folder (contains core.py)")
        if path:
            self.vc_train_applio_root_var.set(path)

    def _read_text(self, widget: scrolledtext.ScrolledText) -> str:
        return widget.get("1.0", tk.END).strip()

    def _load_vc_runtime_config(self) -> dict[str, str]:
        if not VC_RUNTIME_CONFIG_PATH.exists():
            return {}

        try:
            raw = json.loads(VC_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                return {}
            return {str(k): "" if v is None else str(v) for k, v in raw.items()}
        except Exception:
            return {}

    @staticmethod
    def _build_rvc_cli_command(vc_python: str, model_path: str, index_path: str) -> str:
        vc_python = vc_python.strip()
        model_path = model_path.strip()
        index_path = index_path.strip()
        if not vc_python or not model_path or not index_path:
            return ""
        return (
            f'"{vc_python}" -m rvc_python cli --input "{{input_wav}}" --output "{{output_wav}}" '
            f'--model "{model_path}" --index "{index_path}" --device cuda:0 --method rmvpe --version v2'
        )

    @staticmethod
    def _find_latest_trained_rvc_pair() -> tuple[str, str] | None:
        if not TRAINED_RVC_MODEL_ROOT.exists():
            return None

        pth_files = [p for p in TRAINED_RVC_MODEL_ROOT.rglob("*.pth") if p.is_file()]
        if not pth_files:
            return None

        latest_pth = max(pth_files, key=lambda p: p.stat().st_mtime)
        index_files = [p for p in latest_pth.parent.glob("*.index") if p.is_file()]
        if not index_files:
            index_files = [p for p in TRAINED_RVC_MODEL_ROOT.rglob("*.index") if p.is_file()]
        if not index_files:
            return None

        latest_index = max(index_files, key=lambda p: p.stat().st_mtime)
        return str(latest_pth), str(latest_index)

    @staticmethod
    def _resolve_auto_rvc_command(runtime: dict[str, str]) -> str:
        command = runtime.get("command_rvc", "").strip()
        vc_python = runtime.get("vc_python", "").strip() or str(DEFAULT_RVC_PYTHON_PATH)
        model_path = runtime.get("rvc_model", "").strip() or str(DEFAULT_RVC_MODEL_PATH)
        index_path = runtime.get("rvc_index", "").strip() or str(DEFAULT_RVC_INDEX_PATH)
        trained_pair = KVoiceWalkGui._find_latest_trained_rvc_pair()
        default_model = str(DEFAULT_RVC_MODEL_PATH)
        command_uses_default_model = bool(command and default_model in command)

        if trained_pair and (not command or command_uses_default_model or model_path == default_model):
            model_path, index_path = trained_pair
            if Path(vc_python).exists():
                return KVoiceWalkGui._build_rvc_cli_command(vc_python, model_path, index_path)

        if command:
            return command

        required_paths = (vc_python, model_path, index_path)
        if all(Path(p).exists() for p in required_paths):
            return KVoiceWalkGui._build_rvc_cli_command(vc_python, model_path, index_path)
        return ""

    def _apply_launcher_vc_backend_default(self, auto_apply: bool = False) -> None:
        if not VC_BACKEND_CONFIG_PATH.exists():
            return

        backend = ""
        try:
            for raw_line in VC_BACKEND_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                backend = line.lower()
                break
        except Exception:
            return

        if backend == "sovits":
            runtime = self._load_vc_runtime_config()
            if not runtime.get("command_sovits", "").strip():
                return

        preset_name = {
            "rvc": "RVC (Auto from Launcher)",
            "sovits": "SoVITS (Auto from Launcher)",
        }.get(backend)

        if not preset_name:
            return

        current = self.post_vc_preset_var.get().strip()
        if current == "Custom (Keep Current)" or auto_apply:
            self.post_vc_preset_var.set(preset_name)
        if auto_apply:
            self._apply_post_vc_template_preset(quiet=True)

    def _apply_post_vc_template_preset(self, quiet: bool = False) -> None:
        name = self.post_vc_preset_var.get().strip()
        preset = VC_TEMPLATE_PRESETS.get(name)
        if not preset:
            return

        command_template, suffix = preset
        runtime = self._load_vc_runtime_config()

        if command_template == AUTO_RVC_TEMPLATE_TOKEN:
            command_template = self._resolve_auto_rvc_command(runtime)
            if not command_template:
                msg = (
                    "RVC auto runtime is not ready yet. Run derpy-turtle-kokoro-trainer.exe once to install RVC "
                    "and download the default model."
                )
                self._append_log(msg)
                if not quiet:
                    messagebox.showwarning("RVC Not Ready", msg)
                return

        if command_template == AUTO_SOVITS_TEMPLATE_TOKEN:
            command_template = runtime.get("command_sovits", "").strip()
            if not command_template:
                msg = (
                    "SoVITS auto command is not configured in vc-runtime.json. Use RVC auto mode for full one-click "
                    "setup, or set a custom SoVITS command."
                )
                self._append_log(msg)
                if not quiet:
                    messagebox.showwarning("SoVITS Not Ready", msg)
                return

        if not command_template:
            return

        self.post_vc_command_var.set(command_template)
        self.post_vc_suffix_var.set(suffix)
        self.post_vc_enabled_var.set(True)
        self._append_log(f"Applied VC preset: {name}")

    def _use_latest_trained_rvc(self) -> None:
        trained_pair = self._find_latest_trained_rvc_pair()
        if not trained_pair:
            msg = f"No trained RVC .pth/.index pair found under {TRAINED_RVC_MODEL_ROOT}"
            self._append_log(msg)
            messagebox.showwarning("RVC Model Not Found", msg)
            return

        runtime = self._load_vc_runtime_config()
        vc_python = runtime.get("vc_python", "").strip() or str(DEFAULT_RVC_PYTHON_PATH)
        if not Path(vc_python).exists():
            msg = f"RVC Python was not found: {vc_python}"
            self._append_log(msg)
            messagebox.showwarning("RVC Not Ready", msg)
            return

        model_path, index_path = trained_pair
        self.post_vc_preset_var.set("RVC (Auto from Launcher)")
        self.post_vc_command_var.set(self._build_rvc_cli_command(vc_python, model_path, index_path))
        self.post_vc_suffix_var.set("_rvc")
        self.post_vc_enabled_var.set(True)
        self._append_log(f"Using latest trained RVC model: {model_path}")
        self._append_log(f"Using latest trained RVC index: {index_path}")

    def _show_post_vc_help(self) -> None:
        messagebox.showinfo(
            "Post VC Command Template",
            "Enable Post VC to run an external voice-conversion command after each successful Random Walk task.\n\n"
            "Required placeholders in the command:\n"
            "- {input_wav}: latest generated WAV from the run\n"
            "- {output_wav}: where the converted WAV should be written\n\n"
            "Optional placeholders:\n"
            "- {result_dir}: output folder for this run\n"
            "- {project_root}: Derpy Turtle project root\n\n"
            "Quick setup:\n"
            "- Run derpy-turtle-kokoro-trainer.exe (it installs VC backend and writes vc-runtime.json)\n"
            "- Pick VC Preset (RVC/SoVITS auto)\n"
            "- Click Apply VC Preset\n"
            "- Start queue (no command editing needed for RVC auto)\n\n"
            "Manual mode is optional if you want custom commands/models.\n\n"
            "Example:\n"
            "python infer.py --input \"{input_wav}\" --output \"{output_wav}\" --model \"path/to/model.pth\""
        )

    @staticmethod
    def _split_paths(raw_value: str) -> list[str]:
        if not raw_value.strip():
            return []
        normalized = raw_value.replace("|", "\n").replace(";", "\n").replace(",", "\n")
        parts = [part.strip() for part in normalized.splitlines() if part.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for part in parts:
            if part not in seen:
                deduped.append(part)
                seen.add(part)
        return deduped

    @staticmethod
    def _normalize_audio_key(path: str) -> str:
        candidate = Path(path).expanduser()
        try:
            return str(candidate.resolve())
        except Exception:
            return str(candidate)

    def _on_target_audio_many_changed(self, *_args) -> None:
        self._sync_target_text_many_map()
        self._update_target_text_many_button_label()
        if self.target_text_many_window and self.target_text_many_window.winfo_exists():
            self._render_target_text_many_editor()

    def _sync_target_text_many_map(self) -> None:
        extras = self._split_paths(self.target_audio_many_var.get())
        current = dict(self.target_text_many_map)
        remapped: dict[str, str] = {}
        for extra in extras:
            key = self._normalize_audio_key(extra)
            remapped[key] = current.get(key, "")
        self.target_text_many_map = remapped

    def _update_target_text_many_button_label(self) -> None:
        total = len(self.target_text_many_map)
        filled = sum(1 for value in self.target_text_many_map.values() if value.strip())
        self.target_text_many_button_label.set(f"Map Texts ({filled}/{total})")

    def _open_target_text_many_editor(self) -> None:
        self._sync_target_text_many_map()

        if self.target_text_many_window and self.target_text_many_window.winfo_exists():
            self.target_text_many_window.deiconify()
            self.target_text_many_window.lift()
            self.target_text_many_window.focus_force()
            self._render_target_text_many_editor()
            return

        window = tk.Toplevel(self.root)
        window.title("Extra Target Text Mapping")
        window.geometry("1000x700")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_target_text_many_editor)

        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)

        ttk.Label(
            window,
            text="Provide one transcription/text per extra target audio. Each text is linked to exactly that audio.",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 8))

        body = ttk.Frame(window)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        body.columnconfigure(0, weight=1)

        controls = ttk.Frame(window)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        ttk.Button(controls, text="Save Mapping", command=self._save_target_text_many_editor).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(controls, text="Close", command=self._close_target_text_many_editor).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.target_text_many_window = window
        self.target_text_many_body = body
        self.target_text_many_widgets = {}
        self._render_target_text_many_editor()
    def _close_target_text_many_editor(self) -> None:
        self._save_target_text_many_editor()
        if self.target_text_many_window and self.target_text_many_window.winfo_exists():
            self.target_text_many_window.destroy()
        self.target_text_many_window = None
        self.target_text_many_widgets = {}
    def _save_target_text_many_editor(self) -> None:
        if not self.target_text_many_widgets:
            self._update_target_text_many_button_label()
            return

        for key, widget in self.target_text_many_widgets.items():
            try:
                if widget.winfo_exists():
                    self.target_text_many_map[key] = self._read_text(widget)
            except tk.TclError:
                continue

        self._update_target_text_many_button_label()

    def _render_target_text_many_editor(self) -> None:
        if not hasattr(self, "target_text_many_body"):
            return

        self._sync_target_text_many_map()

        for child in self.target_text_many_body.winfo_children():
            child.destroy()

        self.target_text_many_widgets = {}
        extras = self._split_paths(self.target_audio_many_var.get())
        if not extras:
            ttk.Label(self.target_text_many_body, text="No extra target audios selected.").grid(row=0, column=0, sticky="w")
            return

        for index, audio_path in enumerate(extras, start=1):
            key = self._normalize_audio_key(audio_path)
            card = ttk.LabelFrame(self.target_text_many_body, text=f"Extra Audio #{index}: {Path(audio_path).name}", padding=8)
            card.grid(row=index - 1, column=0, sticky="ew", pady=(0, 8))
            card.columnconfigure(0, weight=1)

            ttk.Label(card, text=audio_path).grid(row=0, column=0, sticky="w")
            text_widget = scrolledtext.ScrolledText(card, height=4, wrap=tk.WORD)
            text_widget.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            text_widget.insert("1.0", self.target_text_many_map.get(key, ""))
            self.target_text_many_widgets[key] = text_widget

    def _build_task(self) -> Task:
        mode = self.mode_var.get().strip()
        if mode not in MODES:
            raise ValueError("Select a valid mode")

        args: list[str] = []
        script = "main.py"
        device = self.device_var.get().strip() or "auto"

        step_limit: int | None = None

        post_vc_enabled = bool(self.post_vc_enabled_var.get())
        post_vc_command = self.post_vc_command_var.get().strip()
        post_vc_suffix = self.post_vc_suffix_var.get().strip() or "_vc"

        if mode != MODE_TRAIN_RVC:
            args += ["--device", device]

        if post_vc_enabled and mode != MODE_RANDOM_WALK:
            post_vc_enabled = False

        if post_vc_enabled:
            if not post_vc_command:
                raise ValueError("Post VC is enabled but VC Command Template is empty")
            if "{input_wav}" not in post_vc_command or "{output_wav}" not in post_vc_command:
                raise ValueError("VC Command Template must include {input_wav} and {output_wav}")

        output_name = self.output_name_var.get().strip()
        if output_name and mode != MODE_TRAIN_RVC:
            args += ["--output_name", output_name]

        if mode == MODE_RANDOM_WALK:
            target_audio = self.target_audio_var.get().strip()
            target_text = self._read_text(self.target_text_widget)
            if not target_audio:
                raise ValueError("Random Walk requires Target Audio")
            if not Path(target_audio).exists():
                raise ValueError(f"Target Audio not found: {target_audio}")
            if not target_text and not self.transcribe_start_var.get():
                raise ValueError("Random Walk requires Target Text unless Transcribe Start is enabled")

            self._save_target_text_many_editor()

            args += ["--target_audio", target_audio, "--target_text", target_text]

            extra_targets = self._split_paths(self.target_audio_many_var.get())
            mapped_extra_texts: list[str] = []
            for extra in extra_targets:
                if not Path(extra).exists():
                    raise ValueError(f"Extra target audio not found: {extra}")

                key = self._normalize_audio_key(extra)
                mapped_text = self.target_text_many_map.get(key, "").strip()
                if not mapped_text:
                    raise ValueError(
                        f"Missing target text for extra audio: {Path(extra).name}. "
                        "Click 'Map Texts' and add one text per extra audio."
                    )
                mapped_extra_texts.append(mapped_text)

            if extra_targets:
                args += ["--target_audio_many", *extra_targets]
                args += ["--target_text_many", *mapped_extra_texts]

            auto_extra_enabled = self.auto_target_audio_many_var.get() and not extra_targets
            if not auto_extra_enabled:
                args.append("--no_auto_target_audio_many")

            auto_limit = int(self.auto_target_audio_many_limit_var.get().strip() or "0")
            if auto_limit < 0:
                raise ValueError("Auto target audio limit must be >= 0")
            args += ["--auto_target_audio_many_limit", str(auto_limit)]

            other_text = self._read_text(self.other_text_widget)
            if other_text:
                args += ["--other_text", other_text]

            voice_folder = self.voice_folder_var.get().strip()
            if voice_folder:
                args += ["--voice_folder", voice_folder]

            starting_voice = self.starting_voice_var.get().strip()
            if starting_voice:
                args += ["--starting_voice", starting_voice]

            args += ["--population_limit", self.population_limit_var.get().strip() or "10"]
            step_limit = int(self.step_limit_var.get().strip() or "10000")
            if step_limit < 1:
                raise ValueError("Step Limit must be at least 1")
            log_interval = int(self.log_interval_var.get().strip() or "10")
            if log_interval < 1:
                raise ValueError("Log Every N Steps must be at least 1")

            elite_size = int(self.elite_size_var.get().strip() or "4")
            stagnation_limit = int(self.stagnation_limit_var.get().strip() or "250")
            restart_diversity = float(self.restart_diversity_var.get().strip() or "0.35")
            candidates_per_step = int(self.candidates_per_step_var.get().strip() or "3")
            max_candidates_per_step = int(self.max_candidates_per_step_var.get().strip() or "8")
            if elite_size < 1:
                raise ValueError("Elite Size must be at least 1")
            if stagnation_limit < 1:
                raise ValueError("Stagnation Limit must be at least 1")
            if restart_diversity <= 0:
                raise ValueError("Restart Diversity must be > 0")
            if candidates_per_step < 1:
                raise ValueError("Candidates / Step must be at least 1")
            if max_candidates_per_step < 1:
                raise ValueError("Max Beam must be at least 1")
            if max_candidates_per_step < candidates_per_step:
                raise ValueError("Max Beam must be >= Candidates / Step")

            optimizer = self.optimizer_var.get().strip().lower() or "hybrid"
            if optimizer not in {"random_walk", "cma_es", "hybrid"}:
                raise ValueError("Optimizer must be one of: hybrid, cma_es, random_walk")
            refine_top_k = int(self.refine_top_k_var.get().strip() or "4")
            cma_sigma = float(self.cma_sigma_var.get().strip() or "0.30")
            cma_latent_dim = int(self.cma_latent_dim_var.get().strip() or "16")
            pareto_archive_size = int(self.pareto_archive_size_var.get().strip() or "32")
            if refine_top_k < 1:
                raise ValueError("Refine Top-K must be at least 1")
            if cma_sigma <= 0:
                raise ValueError("CMA Sigma must be > 0")
            if cma_latent_dim < 2:
                raise ValueError("CMA Latent Dim must be at least 2")
            if pareto_archive_size < 1:
                raise ValueError("Pareto Archive must be at least 1")

            weight_target = float(self.score_weight_target_var.get().strip() or "0.45")
            weight_self = float(self.score_weight_self_var.get().strip() or "0.33")
            weight_feature = float(self.score_weight_feature_var.get().strip() or "0.10")
            weight_accent = float(self.score_weight_accent_var.get().strip() or "0.12")
            weights = [weight_target, weight_self, weight_feature, weight_accent]
            if any(w < 0 for w in weights):
                raise ValueError("All score weights must be non-negative")
            if sum(weights) <= 0:
                raise ValueError("At least one score weight must be positive")

            args += ["--step_limit", str(step_limit)]
            args += ["--log_interval", str(log_interval)]
            args += ["--elite_size", str(elite_size)]
            args += ["--stagnation_limit", str(stagnation_limit)]
            args += ["--restart_diversity", str(restart_diversity)]
            args += ["--candidates_per_step", str(candidates_per_step)]
            args += ["--max_candidates_per_step", str(max_candidates_per_step)]
            args += ["--score_weight_target", str(weight_target)]
            args += ["--score_weight_self", str(weight_self)]
            args += ["--score_weight_feature", str(weight_feature)]
            args += ["--score_weight_accent", str(weight_accent)]
            args += ["--optimizer", optimizer]
            args += ["--refine_top_k", str(refine_top_k)]
            args += ["--cma_sigma", str(cma_sigma)]
            args += ["--cma_latent_dim", str(cma_latent_dim)]
            args += ["--pareto_archive_size", str(pareto_archive_size)]

            if not self.adaptive_beam_var.get():
                args.append("--no_adaptive_beam")
            if not self.dynamic_weight_schedule_var.get():
                args.append("--no_dynamic_weight_schedule")

            if self.interpolate_start_var.get():
                args.append("--interpolate_start")
            if self.transcribe_start_var.get():
                args.append("--transcribe_start")

            auto_mode = "on" if auto_extra_enabled else "off"
            adaptive_mode = "on" if self.adaptive_beam_var.get() else "off"
            dynamic_mode = "on" if self.dynamic_weight_schedule_var.get() else "off"
            post_vc_summary = f"on({post_vc_suffix})" if post_vc_enabled else "off"
            beam_summary = f"{candidates_per_step}->{max_candidates_per_step}" if self.adaptive_beam_var.get() else str(candidates_per_step)
            summary = (
                f"audio={Path(target_audio).name}, output={output_name or 'my_new_voice'}, device={device}, "
                f"log={log_interval}, beam={beam_summary}, adaptive={adaptive_mode}, dyn_sched={dynamic_mode}, "
                f"optimizer={optimizer}, topk={refine_top_k}, sigma={cma_sigma:.2f}, latent={cma_latent_dim}, "
                f"accent_w={weight_accent:.2f}, extra={len(extra_targets)}, mapped_extra={len(mapped_extra_texts)}, auto_extra={auto_mode}, post_vc={post_vc_summary}"
            )

        elif mode == MODE_TRAIN_RVC:
            script = str(Path("utilities") / "vc_trainer.py")
            target_audio = self.target_audio_var.get().strip()
            if not target_audio:
                raise ValueError("Train Target RVC Model requires Target Audio")
            if not Path(target_audio).exists():
                raise ValueError(f"Target Audio not found: {target_audio}")

            extra_targets = self._split_paths(self.target_audio_many_var.get())
            for extra in extra_targets:
                if not Path(extra).exists():
                    raise ValueError(f"Extra target audio not found: {extra}")

            model_name = output_name or "target_voice"
            sample_rate = int(self.vc_train_sample_rate_var.get().strip() or "48000")
            epochs = int(self.vc_train_epochs_var.get().strip() or "250")
            batch_size = int(self.vc_train_batch_size_var.get().strip() or "4")
            if sample_rate not in {32000, 40000, 48000}:
                raise ValueError("RVC Sample Rate must be 32000, 40000, or 48000")
            if epochs < 1:
                raise ValueError("RVC Epochs must be at least 1")
            if batch_size < 1:
                raise ValueError("RVC Batch must be at least 1")

            applio_root = self.vc_train_applio_root_var.get().strip()
            args += [
                "--model_name", model_name,
                "--target_audio", target_audio,
                "--sample_rate", str(sample_rate),
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--gpu", "0",
            ]
            if extra_targets:
                args += ["--target_audio_many", *extra_targets]
            if applio_root:
                args += ["--applio_root", applio_root]
            if self.vc_train_prepare_only_var.get():
                args.append("--prepare_only")

            summary = (
                f"model={model_name}, primary={Path(target_audio).name}, extra={len(extra_targets)}, "
                f"epochs={epochs}, batch={batch_size}, sr={sample_rate}, applio={Path(applio_root).name if applio_root else 'auto'}"
            )

        elif mode == MODE_TEST_VOICE:
            test_voice = self.test_voice_var.get().strip()
            target_text = self._read_text(self.target_text_widget)
            if not test_voice:
                raise ValueError("Test Voice mode requires Test Voice (.pt)")
            if not Path(test_voice).exists():
                raise ValueError(f"Test Voice not found: {test_voice}")
            if not target_text:
                raise ValueError("Test Voice mode requires Target Text")

            args += ["--test_voice", test_voice, "--target_text", target_text]
            summary = f"test={Path(test_voice).name}, output={output_name or 'my_new_voice'}, device={device}"

        elif mode == MODE_TRANSCRIBE_MANY:
            transcribe_many = self.transcribe_many_var.get().strip()
            if not transcribe_many:
                raise ValueError("Transcribe Many mode requires a file or folder path")
            if not Path(transcribe_many).exists():
                raise ValueError(f"Transcribe Many path not found: {transcribe_many}")

            args += ["--transcribe_many", transcribe_many]
            summary = f"transcribe_many={Path(transcribe_many).name}, device={device}"

        elif mode == MODE_EXPORT_BIN:
            voice_folder = self.voice_folder_var.get().strip()
            if not voice_folder:
                raise ValueError("Export Voices Bin mode requires Voice Folder")
            if not Path(voice_folder).exists():
                raise ValueError(f"Voice Folder not found: {voice_folder}")

            args += ["--voice_folder", voice_folder, "--export_bin"]
            summary = f"export_bin from {Path(voice_folder).name}"

        else:
            raise ValueError("Unsupported mode")

        task = Task(
            task_id=self.next_task_id,
            mode=mode,
            args=args,
            summary=summary,
            script=script,
            step_limit=step_limit,
            post_vc_enabled=post_vc_enabled,
            post_vc_command=post_vc_command,
            post_vc_suffix=post_vc_suffix,
        )
        self.next_task_id += 1
        return task

    def _add_task(self) -> None:
        try:
            task = self._build_task()
        except ValueError as e:
            messagebox.showerror("Invalid Task", str(e))
            self._append_log(f"Add Task validation failed: {e}")
            return
        except Exception as e:
            messagebox.showerror("Task Error", f"Unexpected error while adding task:\n{e}")
            self._append_log(f"Add Task unexpected error: {e}")
            self._append_log(traceback.format_exc().strip())
            return

        self.tasks.append(task)
        self.task_tree.insert("", tk.END, iid=str(task.task_id),
                              values=(task.task_id, task.mode, task.status, "-", task.summary),
                              tags=(self._status_tag(task.status),))
        self._append_log(f"Queued task #{task.task_id}: {task.mode} ({task.summary})")

    @staticmethod
    def _status_tag(status: str) -> str:
        return {
            "Running": "row_running",
            "Done": "row_done",
            "Failed": "row_failed",
            "Stopped": "row_stopped",
        }.get(status, "row_queued")

    def _remove_selected(self) -> None:
        selected = self.task_tree.selection()
        if not selected:
            return

        selected_ids = {int(iid) for iid in selected}
        self.tasks = [task for task in self.tasks if task.task_id not in selected_ids or task.status == "Running"]

        for iid in selected:
            values = self.task_tree.item(iid, "values")
            if len(values) >= 3 and values[2] == "Running":
                continue
            self.task_tree.delete(iid)

    def _clear_queue(self) -> None:
        running_ids = {task.task_id for task in self.tasks if task.status == "Running"}
        self.tasks = [task for task in self.tasks if task.task_id in running_ids]

        for iid in self.task_tree.get_children():
            values = self.task_tree.item(iid, "values")
            if len(values) >= 3 and values[2] == "Running":
                continue
            self.task_tree.delete(iid)

    def _clear_log(self) -> None:
        self.log_widget.delete("1.0", tk.END)

    def _append_log(self, text: str) -> None:
        lowered = text.lower()
        tag = ""
        if any(word in lowered for word in ("error", "traceback", "failed", "exception")):
            tag = "log_error"
        elif any(word in lowered for word in ("done", "complete", "finished", "queued", "success")):
            tag = "log_success"
        elif any(word in lowered for word in ("warn", "stopp")):
            tag = "log_warn"
        self.log_widget.insert(tk.END, text + "\n", tag)
        self.log_widget.see(tk.END)

    def _set_task_status(self, task_id: int, status: str) -> None:
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = status
                break

        iid = str(task_id)
        if self.task_tree.exists(iid):
            values = list(self.task_tree.item(iid, "values"))
            if len(values) >= 5:
                values[2] = status
                self.task_tree.item(iid, values=values, tags=(self._status_tag(status),))
        if status in {"Done", "Failed", "Stopped"}:
            self._set_task_eta(task_id, "-")

    def _set_task_eta(self, task_id: int, eta_text: str) -> None:
        iid = str(task_id)
        if self.task_tree.exists(iid):
            values = list(self.task_tree.item(iid, "values"))
            if len(values) >= 5:
                values[3] = eta_text
                self.task_tree.item(iid, values=values)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _extract_arg_value(args: list[str], flag: str) -> str | None:
        for index, token in enumerate(args):
            if token == flag and index + 1 < len(args):
                return args[index + 1]
        return None

    @staticmethod
    def _resolve_result_dir(raw_path: str) -> Path:
        cleaned = raw_path.strip().strip('\"').strip("'")
        candidate = Path(cleaned)
        if not candidate.is_absolute():
            return (PROJECT_ROOT / candidate).resolve()
        return candidate.resolve()

    def _find_latest_result_dir(self, output_name: str) -> Path | None:
        out_dir = PROJECT_ROOT / "out"
        if not out_dir.exists():
            return None

        candidates = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(f"{output_name}_")]
        if not candidates:
            candidates = [p for p in out_dir.iterdir() if p.is_dir()]
        if not candidates:
            return None

        return max(candidates, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _find_latest_wav(result_dir: Path, excluded_suffix: str) -> Path | None:
        wav_files = [p for p in result_dir.glob("*.wav") if p.is_file()]
        if not wav_files:
            return None

        normalized_suffix = excluded_suffix.lower()
        if normalized_suffix:
            filtered = [p for p in wav_files if not p.stem.lower().endswith(normalized_suffix)]
            if filtered:
                wav_files = filtered

        return max(wav_files, key=lambda p: p.stat().st_mtime)

    def _run_post_vc(self, task: Task, discovered_result_dirs: list[Path], output_name: str) -> bool:
        result_dir: Path | None = None
        for candidate in reversed(discovered_result_dirs):
            if candidate.exists():
                result_dir = candidate
                break

        if result_dir is None:
            result_dir = self._find_latest_result_dir(output_name)

        if result_dir is None or not result_dir.exists():
            self.events.put(("log", f"Task #{task.task_id} post-VC failed: result folder not found"))
            return False

        input_wav = self._find_latest_wav(result_dir, task.post_vc_suffix)
        if input_wav is None:
            self.events.put(("log", f"Task #{task.task_id} post-VC failed: no input WAV found in {result_dir}"))
            return False

        suffix = task.post_vc_suffix or "_vc"
        if any(sep in suffix for sep in ("/", "\\")):
            self.events.put(("log", f"Task #{task.task_id} post-VC failed: invalid suffix '{suffix}'"))
            return False

        output_wav = result_dir / f"{input_wav.stem}{suffix}.wav"

        vc_cmd = task.post_vc_command
        replacements = {
            "{input_wav}": str(input_wav),
            "{output_wav}": str(output_wav),
            "{result_dir}": str(result_dir),
            "{project_root}": str(PROJECT_ROOT),
        }
        for key, value in replacements.items():
            vc_cmd = vc_cmd.replace(key, value)

        self.events.put(("log", f"Task #{task.task_id} post-VC command: {vc_cmd}"))

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        process: subprocess.Popen[str] | None = None
        return_code = -1
        try:
            process = subprocess.Popen(
                vc_cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=env,
                shell=True,
            )

            assert process.stdout is not None
            for line in process.stdout:
                line_text = line.rstrip("\n")
                self.events.put(("log", f"[VC] {line_text}"))
                if self.stop_requested and process.poll() is None:
                    try:
                        process.terminate()
                    except Exception:
                        pass

            return_code = process.wait()
        except Exception as exc:
            self.events.put(("log", f"Task #{task.task_id} post-VC launch failed: {exc}"))
            return False
        finally:
            if process and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass

        if self.stop_requested:
            self.events.put(("log", f"Task #{task.task_id} post-VC stopped"))
            return False

        if return_code != 0:
            self.events.put(("log", f"Task #{task.task_id} post-VC failed with exit code {return_code}"))
            return False

        if not output_wav.exists():
            self.events.put(("log", f"Task #{task.task_id} post-VC failed: expected output missing -> {output_wav}"))
            return False

        # Post-process the VC output.
        try:
            from utilities.signal_processor import normalize_output
            import soundfile as _sf

            vc_audio, vc_sr = _sf.read(str(output_wav))
            _sf.write(str(output_wav), normalize_output(vc_audio, vc_sr), vc_sr)
        except Exception as _sp_err:
            self.events.put(("log", f"Task #{task.task_id} post-process note: {_sp_err}"))

        self.events.put(("log", f"Task #{task.task_id} post-VC output: {output_wav}"))
        return True

    def _start_queue(self) -> None:
        if self.runner_thread and self.runner_thread.is_alive():
            messagebox.showinfo("Queue Running", "Queue is already running")
            return

        has_queued = any(task.status == "Queued" for task in self.tasks)
        if not has_queued:
            messagebox.showinfo("No Tasks", "No queued tasks to run")
            return

        self.stop_requested = False
        self.runner_thread = threading.Thread(target=self._runner_loop, daemon=True)
        self.runner_thread.start()
        self._append_log("Queue started")

    def _stop_queue(self) -> None:
        self.stop_requested = True
        if self.current_process and self.current_process.poll() is None:
            try:
                self.current_process.terminate()
            except Exception as e:
                self._append_log(f"Failed to terminate current task: {e}")
        self._append_log("Stop requested")

    def _runner_loop(self) -> None:
        while True:
            if self.stop_requested:
                self.events.put(("queue_state", "stopped"))
                break

            next_task = next((task for task in self.tasks if task.status == "Queued"), None)
            if next_task is None:
                self.events.put(("queue_state", "completed"))
                break

            task = next_task
            task_output_name = self._extract_arg_value(task.args, "--output_name") or "my_new_voice"
            discovered_result_dirs: list[Path] = []
            self.events.put(("task_status", task.task_id, "Running"))
            self.events.put(("task_eta", task.task_id, "Estimating..."))
            cmd = [sys.executable, task.script, *task.args]
            self.events.put(("log", f"\n=== Running task #{task.task_id}: {' '.join(cmd)}"))

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            env = os.environ.copy()
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            return_code = -1

            try:
                self.current_process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                    env=env,
                )

                assert self.current_process.stdout is not None
                for line in self.current_process.stdout:
                    line_text = line.rstrip("\n")
                    self.events.put(("log", line_text))

                    result_match = RESULT_DIR_RE.search(line_text)
                    if result_match:
                        try:
                            result_dir = self._resolve_result_dir(result_match.group("path"))
                            if result_dir not in discovered_result_dirs:
                                discovered_result_dirs.append(result_dir)
                                self.events.put(("log", f"Detected result folder: {result_dir}"))
                        except Exception as parse_exc:
                            self.events.put(("log", f"Failed to parse result folder line: {parse_exc}"))

                    if task.mode == MODE_RANDOM_WALK:
                        match = PROGRESS_RE.search(line_text)
                        if match:
                            completed = int(match.group("step"))
                            total = int(match.group("total"))
                            elapsed = float(match.group("elapsed"))
                            eta_group = match.group("eta")
                            if eta_group is not None:
                                eta_seconds = float(eta_group)
                            elif completed > 0 and total >= completed:
                                eta_seconds = (elapsed / completed) * (total - completed)
                            else:
                                eta_seconds = 0.0
                            self.events.put(("task_eta", task.task_id, self._format_eta(eta_seconds)))
                    if self.stop_requested and self.current_process.poll() is None:
                        try:
                            self.current_process.terminate()
                        except Exception:
                            pass

                return_code = self.current_process.wait()

                if return_code == 0 and task.post_vc_enabled and task.mode == MODE_RANDOM_WALK and not self.stop_requested:
                    vc_ok = self._run_post_vc(task, discovered_result_dirs, task_output_name)
                    if not vc_ok:
                        return_code = 2

            except Exception as e:
                self.events.put(("log", f"Task #{task.task_id} failed to start: {e}"))
                return_code = -1

            finally:
                self.current_process = None

            if self.stop_requested:
                self.events.put(("task_status", task.task_id, "Stopped"))
                self.events.put(("task_eta", task.task_id, "-"))
                self.events.put(("log", f"Task #{task.task_id} stopped"))
                self.events.put(("queue_state", "stopped"))
                break

            if return_code == 0:
                self.events.put(("task_status", task.task_id, "Done"))
                self.events.put(("task_eta", task.task_id, "-"))
                self.events.put(("log", f"Task #{task.task_id} finished successfully"))
            else:
                self.events.put(("task_status", task.task_id, "Failed"))
                self.events.put(("task_eta", task.task_id, "-"))
                self.events.put(("log", f"Task #{task.task_id} failed with exit code {return_code}"))

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "log":
                self._append_log(event[1])
            elif kind == "task_status":
                _, task_id, status = event
                self._set_task_status(task_id, status)
            elif kind == "task_eta":
                _, task_id, eta_text = event
                self._set_task_eta(task_id, eta_text)
            elif kind == "queue_state":
                state = event[1]
                if state == "completed":
                    self._append_log("Queue completed")
                elif state == "stopped":
                    self._append_log("Queue stopped")

        self.root.after(100, self._poll_events)


def main() -> None:
    root = tk.Tk()
    app = KVoiceWalkGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()







































