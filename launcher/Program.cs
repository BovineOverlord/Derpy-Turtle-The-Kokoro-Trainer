using System.Diagnostics;
using System.Drawing;
using System.Text;
using System.Text.Json;
using System.Windows.Forms;

internal sealed record PythonCandidate(string FileName, string[] PrefixArgs)
{
    public string Display => PrefixArgs.Length == 0 ? FileName : $"{FileName} {string.Join(' ', PrefixArgs)}";
}

internal static class Program
{
    private const string DefaultVcBackend = "rvc";
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromMinutes(30) };
    private static SplashScreen? _splash;

    [STAThread]
    private static int Main(string[] args)
    {
        try { Console.OutputEncoding = Encoding.UTF8; } catch { /* No console attached when running as WinExe. */ }

        var startDirectory = AppContext.BaseDirectory;
        var projectRoot = FindProjectRoot(startDirectory);
        if (projectRoot is null)
        {
            MessageBox.Show(
                "Could not locate project root.\n\nPlace derpy-turtle-kokoro-trainer.exe in the repository root (same folder as gui.py and pyproject.toml).",
                "Derpy Turtle", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        _splash = new SplashScreen(Path.Combine(projectRoot, "assets", "derpyturtle.jpg"));
        _splash.Show();

        var logPath = Path.Combine(projectRoot, "derpy-turtle-launcher.log");
        using var logWriter = new StreamWriter(logPath, append: true, Encoding.UTF8) { AutoFlush = true };

        Log(logWriter, "============================================================");
        Log(logWriter, $"Launcher started: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        Log(logWriter, $"Project root: {projectRoot}");

        var venvPython = Path.Combine(projectRoot, ".venv", "Scripts", "python.exe");

        if (!File.Exists(venvPython))
        {
            _splash.SetStatus("Setting up Python environment…");
            Log(logWriter, "No .venv found. Creating virtual environment...");
            var basePython = DetectSystemPython(projectRoot, logWriter);
            if (basePython is null)
            {
                return Fail(
                    "Python 3.10–3.12 was not found.",
                    "Install Python (with the 'py' launcher), then run derpy-turtle-kokoro-trainer.exe again.",
                    logWriter
                );
            }

            _splash.SetStatus("Creating virtual environment…");
            if (!RunCommand(basePython.FileName, basePython.PrefixArgs.Concat(new[] { "-m", "venv", ".venv" }), projectRoot, logWriter, "Create .venv"))
            {
                return Fail("Failed to create .venv.", "Check derpy-turtle-launcher.log for details.", logWriter);
            }

            venvPython = Path.Combine(projectRoot, ".venv", "Scripts", "python.exe");
            if (!File.Exists(venvPython))
            {
                return Fail(".venv creation finished but python.exe is missing.", "Check your Python installation and try again.", logWriter);
            }
        }

        if (!EnvironmentLooksReady(venvPython, projectRoot, logWriter))
        {
            _splash.SetStatus("Installing dependencies — first run, this may take several minutes…");
            Log(logWriter, "Environment is missing required packages. Running first-time setup...");

            if (!RunCommand(venvPython, new[] { "-m", "pip", "install", "--upgrade", "pip", "uv" }, projectRoot, logWriter, "Install/upgrade pip and uv"))
            {
                return Fail("Failed to install uv in .venv.", "Check internet access and derpy-turtle-launcher.log.", logWriter);
            }

            var uvExe = Path.Combine(projectRoot, ".venv", "Scripts", "uv.exe");
            if (!File.Exists(uvExe))
            {
                return Fail("uv.exe not found after installation.", "Check derpy-turtle-launcher.log for details.", logWriter);
            }

            _splash.SetStatus("Installing project packages — this may take several minutes…");
            if (!RunCommand(uvExe, new[] { "sync" }, projectRoot, logWriter, "Install project dependencies (uv sync)"))
            {
                return Fail("Dependency installation failed.", "Check derpy-turtle-launcher.log for details.", logWriter);
            }

            if (!EnvironmentLooksReady(venvPython, projectRoot, logWriter))
            {
                return Fail("Environment check still failed after setup.", "Open derpy-turtle-launcher.log and verify Python/Tk/CUDA dependencies.", logWriter);
            }
        }

        _splash.SetStatus("Setting up voice conversion backend…");
        var selectedVcBackend = ResolveVcBackend(projectRoot, logWriter);
        if (!EnsureVcBackend(selectedVcBackend, venvPython, projectRoot, logWriter))
        {
            Log(logWriter, $"WARNING: VC backend setup failed for '{selectedVcBackend}'.");
            Log(logWriter, "GUI will still launch. You can retry by running the launcher again after fixing internet/dependency issues.");
        }
        else
        {
            Log(logWriter, $"VC backend ready: {selectedVcBackend}");
        }

        _splash.SetStatus("Launching…");
        Log(logWriter, "Launching GUI...");
        var guiArgs = new List<string> { "gui.py" };
        guiArgs.AddRange(args);

        _splash.Close();
        _splash = null;

        var launchExit = RunCommand(venvPython, guiArgs, projectRoot, logWriter, "Launch GUI", captureOutput: false, waitForExit: true);
        return launchExit ? 0 : 1;
    }

    private static bool EnvironmentLooksReady(string venvPython, string projectRoot, StreamWriter logWriter)
    {
        // Keep this minimal to avoid heavy startup; if these import, core runtime is ready.
        return RunCommand(
            venvPython,
            new[] { "-c", "import tkinter, torch, kokoro, faster_whisper; print('env ok')" },
            projectRoot,
            logWriter,
            "Validate environment",
            captureOutput: true,
            waitForExit: true,
            silentFailure: true
        );
    }

    private static string ResolveVcBackend(string projectRoot, StreamWriter logWriter)
    {
        var envOverride = Environment.GetEnvironmentVariable("DERPY_TURTLE_VC_BACKEND")
            ?? Environment.GetEnvironmentVariable("KVOICEWALK_VC_BACKEND");
        if (!string.IsNullOrWhiteSpace(envOverride))
        {
            var normalizedEnv = NormalizeVcBackend(envOverride);
            Log(logWriter, $"Using VC backend from environment: {normalizedEnv}");
            return normalizedEnv;
        }

        var configPath = Path.Combine(projectRoot, "vc-backend.txt");
        if (!File.Exists(configPath))
        {
            WriteDefaultVcBackendConfig(configPath, logWriter);
            Log(logWriter, $"VC backend config created with default '{DefaultVcBackend}': {configPath}");
            return DefaultVcBackend;
        }

        var configured = ReadFirstConfigValue(configPath);
        if (string.IsNullOrWhiteSpace(configured))
        {
            Log(logWriter, $"VC backend config is empty. Falling back to '{DefaultVcBackend}'.");
            return DefaultVcBackend;
        }

        var normalized = NormalizeVcBackend(configured);
        if (!string.Equals(configured.Trim(), normalized, StringComparison.OrdinalIgnoreCase))
        {
            Log(logWriter, $"VC backend config value '{configured}' normalized to '{normalized}'.");
        }

        return normalized;
    }

    private static string ReadFirstConfigValue(string path)
    {
        foreach (var rawLine in File.ReadLines(path))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }

            return line;
        }

        return string.Empty;
    }

    private static void WriteDefaultVcBackendConfig(string path, StreamWriter logWriter)
    {
        var content = string.Join(
            Environment.NewLine,
            new[]
            {
                "# Derpy Turtle VC backend selection for launcher auto-install.",
                "# Supported values: rvc, sovits, none",
                DefaultVcBackend,
                string.Empty,
            }
        );

        try
        {
            File.WriteAllText(path, content, Encoding.UTF8);
        }
        catch (Exception ex)
        {
            Log(logWriter, $"WARNING: failed to write default vc-backend.txt: {ex.Message}");
        }
    }

    private static string NormalizeVcBackend(string rawValue)
    {
        var value = (rawValue ?? string.Empty).Trim().ToLowerInvariant();
        return value switch
        {
            "rvc" => "rvc",
            "sovits" => "sovits",
            "none" => "none",
            _ => DefaultVcBackend,
        };
    }

    private static bool EnsureVcBackend(string backend, string mainVenvPython, string projectRoot, StreamWriter logWriter)
    {
        if (string.Equals(backend, "none", StringComparison.OrdinalIgnoreCase))
        {
            return WriteVcRuntimeConfig(projectRoot, backend, string.Empty, string.Empty, string.Empty, logWriter);
        }

        var vcPython = EnsureVcVirtualEnv(backend, mainVenvPython, projectRoot, logWriter);
        if (string.IsNullOrWhiteSpace(vcPython))
        {
            return false;
        }

        if (string.Equals(backend, "rvc", StringComparison.OrdinalIgnoreCase))
        {
            return EnsureRvcBackend(vcPython, projectRoot, logWriter);
        }

        if (string.Equals(backend, "sovits", StringComparison.OrdinalIgnoreCase))
        {
            return EnsureSovitsBackend(vcPython, projectRoot, logWriter);
        }

        Log(logWriter, $"Unknown VC backend '{backend}'.");
        return false;
    }

    private static string? EnsureVcVirtualEnv(string backend, string mainVenvPython, string projectRoot, StreamWriter logWriter)
    {
        var venvDir = Path.Combine(projectRoot, $".venv_vc_{backend}");
        var vcPython = Path.Combine(venvDir, "Scripts", "python.exe");
        if (File.Exists(vcPython))
        {
            Log(logWriter, $"VC venv already present: {venvDir}");
            return vcPython;
        }

        _splash?.SetStatus($"Creating isolated {backend} virtual environment…");
        Log(logWriter, $"Creating isolated VC venv: {venvDir}");
        if (!RunCommand(mainVenvPython, new[] { "-m", "venv", venvDir }, projectRoot, logWriter, $"Create VC venv ({backend})"))
        {
            return null;
        }

        if (!File.Exists(vcPython))
        {
            Log(logWriter, $"VC venv created but python missing: {vcPython}");
            return null;
        }

        return vcPython;
    }

    private static bool EnsureRvcBackend(string vcPython, string projectRoot, StreamWriter logWriter)
    {
        if (!IsPipPackageInstalled(vcPython, "rvc-python", projectRoot, logWriter))
        {
            _splash?.SetStatus("Installing rvc-python…");
            // rvc-python depends on omegaconf==2.0.6 metadata that needs pip<24.1.
            if (!RunCommand(vcPython, new[] { "-m", "pip", "install", "--upgrade", "pip<24.1" }, projectRoot, logWriter, "Pin VC pip (<24.1) for rvc-python"))
            {
                return false;
            }

            if (!RunCommand(vcPython, new[] { "-m", "pip", "install", "rvc-python==0.1.5" }, projectRoot, logWriter, "Install rvc-python"))
            {
                return false;
            }
        }

        _splash?.SetStatus("Checking RVC CUDA/PyTorch setup…");
        if (!EnsureRvcCudaTorch(vcPython, projectRoot, logWriter))
        {
            return false;
        }

        if (!IsPipPackageInstalled(vcPython, "rvc-python", projectRoot, logWriter))
        {
            Log(logWriter, "rvc-python is not installed after install attempt.");
            return false;
        }

        _splash?.SetStatus("Downloading default RVC model…");
        if (!EnsureRvcDefaultModel(projectRoot, logWriter, out var modelPath, out var indexPath))
        {
            return false;
        }

        return WriteVcRuntimeConfig(projectRoot, "rvc", vcPython, modelPath, indexPath, logWriter);
    }

    private static bool EnsureRvcCudaTorch(string vcPython, string projectRoot, StreamWriter logWriter)
    {
        var cudaReady = RunCommand(
            vcPython,
            new[] { "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__)" },
            projectRoot,
            logWriter,
            "Validate RVC CUDA Torch",
            captureOutput: true,
            waitForExit: true,
            silentFailure: true
        );
        if (cudaReady)
        {
            return true;
        }

        // rvc-python pulls in CPU torch as a dependency; replace it with the CUDA build.
        _splash?.SetStatus("Installing CUDA-enabled PyTorch for RVC — this may take a while…");
        Log(logWriter, "Installing CUDA-enabled PyTorch for RVC backend.");
        if (!RunCommand(
                vcPython,
                new[] { "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128" },
                projectRoot,
                logWriter,
                "Install RVC CUDA PyTorch"
            ))
        {
            return false;
        }

        return RunCommand(
            vcPython,
            new[] { "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__)" },
            projectRoot,
            logWriter,
            "Validate RVC CUDA Torch after install",
            captureOutput: true,
            waitForExit: true
        );
    }

    private static bool EnsureSovitsBackend(string vcPython, string projectRoot, StreamWriter logWriter)
    {
        if (!IsPipPackageInstalled(vcPython, "so-vits-svc-fork", projectRoot, logWriter))
        {
            _splash?.SetStatus("Installing so-vits-svc-fork…");
            if (!RunCommand(vcPython, new[] { "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools" }, projectRoot, logWriter, "Prepare pip tooling for so-vits-svc"))
            {
                return false;
            }

            if (!RunCommand(vcPython, new[] { "-m", "pip", "install", "so-vits-svc-fork" }, projectRoot, logWriter, "Install so-vits-svc-fork"))
            {
                return false;
            }
        }

        if (!IsPipPackageInstalled(vcPython, "so-vits-svc-fork", projectRoot, logWriter))
        {
            Log(logWriter, "so-vits-svc-fork is not installed after install attempt.");
            return false;
        }

        return WriteVcRuntimeConfig(projectRoot, "sovits", vcPython, string.Empty, string.Empty, logWriter);
    }

    private static bool EnsureRvcDefaultModel(string projectRoot, StreamWriter logWriter, out string modelPath, out string indexPath)
    {
        var modelDir = Path.Combine(projectRoot, "vc_models", "rvc", "default_abe_shinzo");
        Directory.CreateDirectory(modelDir);

        modelPath = Path.Combine(modelDir, "AbeShinzo2.pth");
        indexPath = Path.Combine(modelDir, "added_IVF429_Flat_nprobe_6.index");

        var modelUrl = "https://huggingface.co/AbeShinzo0708/RVC_AbeShinzo/resolve/main/AbeShinzo2.pth";
        var indexUrl = "https://huggingface.co/AbeShinzo0708/RVC_AbeShinzo/resolve/main/added_IVF429_Flat_nprobe_6.index";

        if (!DownloadFileIfMissing(modelUrl, modelPath, logWriter))
        {
            return false;
        }

        if (!DownloadFileIfMissing(indexUrl, indexPath, logWriter))
        {
            return false;
        }

        return true;
    }

    private static bool DownloadFileIfMissing(string url, string destinationPath, StreamWriter logWriter)
    {
        try
        {
            var existing = new FileInfo(destinationPath);
            if (existing.Exists && existing.Length > 0)
            {
                Log(logWriter, $"Already present: {destinationPath}");
                return true;
            }

            var destinationDir = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(destinationDir))
            {
                Directory.CreateDirectory(destinationDir);
            }

            var tmpPath = destinationPath + ".tmp";
            if (File.Exists(tmpPath))
            {
                File.Delete(tmpPath);
            }

            Log(logWriter, $"Downloading {url}");
            _splash?.SetStatus($"Downloading {Path.GetFileName(destinationPath)}…");
            using var response = Http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            if (!response.IsSuccessStatusCode)
            {
                Log(logWriter, $"Download failed ({(int)response.StatusCode} {response.ReasonPhrase}): {url}");
                return false;
            }

            using (var input = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
            using (var output = File.Create(tmpPath))
            {
                input.CopyTo(output);
            }

            if (File.Exists(destinationPath))
            {
                File.Delete(destinationPath);
            }
            File.Move(tmpPath, destinationPath);

            var bytes = new FileInfo(destinationPath).Length;
            Log(logWriter, $"Saved {Path.GetFileName(destinationPath)} ({bytes} bytes)");
            return true;
        }
        catch (Exception ex)
        {
            Log(logWriter, $"Download exception for {url}: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static bool WriteVcRuntimeConfig(string projectRoot, string backend, string vcPython, string rvcModelPath, string rvcIndexPath, StreamWriter logWriter)
    {
        try
        {
            var configPath = Path.Combine(projectRoot, "vc-runtime.json");
            var payload = new Dictionary<string, string?>
            {
                ["backend"] = backend,
                ["vc_python"] = vcPython,
                ["rvc_model"] = rvcModelPath,
                ["rvc_index"] = rvcIndexPath,
                ["command_rvc"] = BuildRvcCommandTemplate(vcPython, rvcModelPath, rvcIndexPath),
                ["updated_utc"] = DateTime.UtcNow.ToString("O"),
            };

            var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(configPath, json, Encoding.UTF8);
            Log(logWriter, $"Wrote VC runtime config: {configPath}");
            return true;
        }
        catch (Exception ex)
        {
            Log(logWriter, $"Failed to write vc-runtime.json: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static string BuildRvcCommandTemplate(string vcPython, string rvcModelPath, string rvcIndexPath)
    {
        if (string.IsNullOrWhiteSpace(vcPython) || string.IsNullOrWhiteSpace(rvcModelPath) || string.IsNullOrWhiteSpace(rvcIndexPath))
        {
            return string.Empty;
        }

        return $"\"{vcPython}\" -m rvc_python cli --input \"{{input_wav}}\" --output \"{{output_wav}}\" --model \"{rvcModelPath}\" --index \"{rvcIndexPath}\" --device cuda:0 --method rmvpe --version v2";
    }

    private static bool IsPipPackageInstalled(string venvPython, string packageName, string projectRoot, StreamWriter logWriter)
    {
        return RunCommand(
            venvPython,
            new[] { "-m", "pip", "show", packageName },
            projectRoot,
            logWriter,
            $"Check pip package ({packageName})",
            captureOutput: true,
            waitForExit: true,
            silentFailure: true
        );
    }

    private static PythonCandidate? DetectSystemPython(string projectRoot, StreamWriter logWriter)
    {
        var candidates = new[]
        {
            new PythonCandidate("py", new[] { "-3.12" }),
            new PythonCandidate("py", new[] { "-3.11" }),
            new PythonCandidate("py", new[] { "-3.10" }),
            new PythonCandidate("python", Array.Empty<string>()),
        };

        foreach (var candidate in candidates)
        {
            Log(logWriter, $"Checking Python candidate: {candidate.Display}");
            var ok = RunCommand(
                candidate.FileName,
                candidate.PrefixArgs.Concat(new[]
                {
                    "-c",
                    "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)"
                }),
                projectRoot,
                logWriter,
                $"Check {candidate.Display}",
                captureOutput: true,
                waitForExit: true,
                silentFailure: true
            );

            if (ok)
            {
                Log(logWriter, $"Using Python candidate: {candidate.Display}");
                return candidate;
            }
        }

        return null;
    }

    private static string? FindProjectRoot(string startDirectory)
    {
        var directory = new DirectoryInfo(startDirectory);
        while (directory is not null)
        {
            var hasGui = File.Exists(Path.Combine(directory.FullName, "gui.py"));
            var hasPyProject = File.Exists(Path.Combine(directory.FullName, "pyproject.toml"));
            if (hasGui && hasPyProject)
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static bool RunCommand(
        string fileName,
        IEnumerable<string> arguments,
        string workingDirectory,
        StreamWriter logWriter,
        string label,
        bool captureOutput = true,
        bool waitForExit = true,
        bool silentFailure = false)
    {
        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = captureOutput,
            RedirectStandardError = captureOutput,
            CreateNoWindow = captureOutput,
        };

        foreach (var arg in arguments)
        {
            psi.ArgumentList.Add(arg);
        }

        Log(logWriter, $"[{label}] {psi.FileName} {string.Join(' ', psi.ArgumentList)}");

        try
        {
            using var process = new Process { StartInfo = psi };

            if (!captureOutput)
            {
                process.Start();
                if (waitForExit)
                {
                    process.WaitForExit();
                    var ok = process.ExitCode == 0;
                    if (!ok && !silentFailure)
                    {
                        Log(logWriter, $"[{label}] failed with exit code {process.ExitCode}");
                    }
                    return ok;
                }

                return true;
            }

            process.OutputDataReceived += (_, eventArgs) =>
            {
                if (!string.IsNullOrWhiteSpace(eventArgs.Data))
                {
                    Log(logWriter, eventArgs.Data!);
                }
            };
            process.ErrorDataReceived += (_, eventArgs) =>
            {
                if (!string.IsNullOrWhiteSpace(eventArgs.Data))
                {
                    Log(logWriter, eventArgs.Data!);
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            if (waitForExit)
            {
                process.WaitForExit();
                var ok = process.ExitCode == 0;
                if (!ok && !silentFailure)
                {
                    Log(logWriter, $"[{label}] failed with exit code {process.ExitCode}");
                }
                return ok;
            }

            return true;
        }
        catch (Exception ex)
        {
            if (!silentFailure)
            {
                Log(logWriter, $"[{label}] exception: {ex.GetType().Name}: {ex.Message}");
            }
            return false;
        }
    }

    private static void Log(StreamWriter writer, string text)
    {
        writer.WriteLine(text);
        Console.WriteLine(text);
    }

    private static int Fail(string headline, string detail, StreamWriter? logWriter = null)
    {
        if (logWriter is not null)
        {
            Log(logWriter, $"ERROR: {headline}");
            Log(logWriter, detail);
        }

        _splash?.Close();
        _splash = null;

        MessageBox.Show(
            $"{headline}\n\n{detail}",
            "Derpy Turtle — Error",
            MessageBoxButtons.OK,
            MessageBoxIcon.Error);

        return 1;
    }
}

internal sealed class SplashScreen
{
    private Form? _form;
    private Label? _statusLabel;
    private readonly string _imagePath;
    private readonly ManualResetEventSlim _ready = new(false);

    public SplashScreen(string imagePath) => _imagePath = imagePath;

    public void Show()
    {
        var thread = new Thread(RunForm);
        thread.SetApartmentState(ApartmentState.STA);
        thread.IsBackground = true;
        thread.Start();
        _ready.Wait(TimeSpan.FromSeconds(5));
    }

    public void SetStatus(string status)
    {
        if (_form is null || _statusLabel is null) return;
        try { _form.Invoke(() => _statusLabel.Text = status); } catch { }
    }

    public void Close()
    {
        if (_form is null) return;
        try { _form.Invoke(() => _form.Close()); } catch { }
    }

    private void RunForm()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        const int W = 480;
        const int ImgH = 360;
        const int StatusH = 44;

        _form = new Form
        {
            FormBorderStyle = FormBorderStyle.None,
            StartPosition = FormStartPosition.CenterScreen,
            TopMost = true,
            BackColor = Color.FromArgb(30, 30, 46),
            ClientSize = new Size(W, ImgH + StatusH),
        };

        if (File.Exists(_imagePath))
        {
            var img = Image.FromFile(_imagePath);
            var picture = new PictureBox
            {
                Image = img,
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = Color.FromArgb(30, 30, 46),
                Bounds = new Rectangle(0, 0, W, ImgH),
            };
            _form.Controls.Add(picture);
        }

        _statusLabel = new Label
        {
            Text = "Starting…",
            ForeColor = Color.FromArgb(205, 214, 244),
            BackColor = Color.FromArgb(49, 50, 68),
            Font = new Font("Segoe UI", 10.5f),
            TextAlign = ContentAlignment.MiddleCenter,
            Bounds = new Rectangle(0, ImgH, W, StatusH),
        };
        _form.Controls.Add(_statusLabel);

        _form.Shown += (_, _) => _ready.Set();
        Application.Run(_form);
    }
}
