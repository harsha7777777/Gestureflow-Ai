# GestureFlow AI

**Control your computer with your own hand gestures, trained by you in minutes, running 100% locally.**

<!-- Demo GIF goes here once Milestone 4 is done. -->

> **Status: early development (Milestone 2 of 7).** Live hand tracking and recording your own
> gesture examples work. Training and computer control come next. See the [roadmap](ROADMAP.md).

## What it does

Record 5–15 examples of a gesture with your webcam, let a lightweight model learn it, and map it
to an action such as play/pause, volume, next track, screenshot or back/forward.

- **Personalized:** you teach it *your* gestures; nothing is hardcoded.
- **Private:** everything runs on your machine. No webcam frames or hand data leave it, and raw
  video is never saved.
- **Safe by design:** confidence threshold, smoothing, cooldowns, a visible "control active"
  indicator and a global emergency hotkey.

## Requirements

| | |
|---|---|
| OS | macOS 11+ on Apple Silicon (M1 or newer, primary), Windows 10/11 |
| Python | 3.12 or 3.13 (3.12 recommended) |
| Hardware | Any webcam (built-in is fine) |

Intel Macs are not supported because MediaPipe does not publish packages for them.

## Install

Each command below is **one line**: copy and paste one at a time.

### macOS

1. **Install Python 3.12** from [python.org/downloads](https://www.python.org/downloads/macos/)
   (pick the latest 3.12.x "macOS 64-bit universal2 installer"). Check it in Terminal:
   ```bash
   python3.12 --version
   ```
2. **Install Git** (macOS offers to install it the first time you run `git`):
   ```bash
   git --version
   ```
3. **Download GestureFlow AI and set it up:**
   ```bash
   git clone https://github.com/harsha7777777/Gestureflow-Ai.git
   cd Gestureflow-Ai
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   python -m gestureflow --download-model
   ```

### Windows

1. **Install Python 3.12** from [python.org/downloads](https://www.python.org/downloads/windows/).
   In the installer, tick **"Add python.exe to PATH"**.
2. **Install Git** from [git-scm.com](https://git-scm.com/download/win) (default options are fine).
3. **Open PowerShell** and run:
   ```powershell
   git clone https://github.com/harsha7777777/Gestureflow-Ai.git
   cd Gestureflow-Ai
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e .
   python -m gestureflow --download-model
   ```
   If PowerShell refuses to run `Activate.ps1`, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

### The hand model

`--download-model` downloads MediaPipe's hand landmark model (about 8 MB, Apache 2.0) from Google
into the `models/` folder and verifies its SHA-256 checksum. This one-time download is the only
network access the app makes. If you skip it, the app offers to download it on first launch.

## Run

Every time you open a new terminal, go to the folder and activate the environment first:

```bash
cd Gestureflow-Ai
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m gestureflow
```

A window opens with your camera feed and your hand's skeleton drawn on it. Use the **Camera**
menu to switch cameras (for example, to an iPhone used as a webcam). Other options:

```bash
python -m gestureflow --camera 1        # start with a specific camera
python -m gestureflow --check           # check the installation
```

**macOS:** the first time, macOS asks whether Terminal (or VS Code) may use the camera. Click
**Allow**, then quit Terminal completely (Cmd+Q), reopen it and run the app again. See
[docs/permissions.md](docs/permissions.md).

## Record your gesture examples

The panel on the right of the window records examples of your gestures. Only the 21 hand
landmark coordinates are saved, never images or video.

1. Type a name, for example `thumbs_up` or `swipe_left`.
2. Choose **Pose (hold still)** for a hand shape, or **Movement (swipe)** for a motion.
3. Click **Record example**. After a 3-second countdown, hold the pose for 1 second, or do the
   movement within 2 seconds. Start moving when the panel says **Do the movement now**, not during
   the countdown.
4. Repeat 5–15 times. Vary your hand position, distance and angle a little each time.
5. Record some examples named **`neutral`**: your hand resting, typing, scratching your head,
   moving casually. This teaches the app when to do *nothing*.

A movement only counts if your hand really moves: either it travels about a palm's width across
the frame (a swipe, even one that comes back), or the fingers clearly change shape without the hand
going anywhere (a grab, a pinch, opening the hand). Otherwise the recording is rejected with a
message saying how far you got. **Undo last** deletes the most recent example of the
selected gesture.

Examples are stored in `user_data/gestures/<name>/`, one small JSON file each. You can delete
that folder at any time. Training on these examples arrives in Milestone 3.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No camera found" (macOS) | System Settings > Privacy & Security > Camera: turn on the app running Python (Terminal, iTerm, VS Code). Quit it with Cmd+Q and reopen. |
| "No camera found" (Windows) | Settings > Privacy & security > Camera: turn on *Camera access* and *Let desktop apps access your camera*. |
| "Camera stopped sending images" | Another app (Zoom, Teams, FaceTime, Photo Booth) is using the camera. Close it and click **Try again**. |
| `python3.12: command not found` | Python 3.12 isn't installed, or the installer didn't finish. Reinstall from python.org. |
| `No matching distribution found for mediapipe` | Wrong Python version (must be 3.12 or 3.13) or an Intel Mac. Check with `python --version`. |
| Download fails with a certificate error | Run `pip install --upgrade certifi` inside the environment and try again. |
| `ModuleNotFoundError: No module named 'gestureflow'` | The environment isn't active. Run the `source .venv/bin/activate` (or Windows) line again. |
| Low FPS | Close other apps using the camera or CPU; plug in your laptop. |

## Running the tests

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

No webcam is needed for the automated tests.

## How it works

```mermaid
flowchart LR
    A[Webcam] --> B[Hand landmarks<br/>MediaPipe]
    B --> C[Features]
    C --> D[Classifier]
    D --> E[Recognition filter<br/>threshold, smoothing,<br/>neutral, cooldown]
    E --> F[Gesture to command<br/>mapping]
    F --> G[Action layer<br/>macOS / Windows]
    G --> H[Operating system]
```

The classifier only knows gesture names, and the action layer only knows command names, so you
can remap a gesture without retraining. Details: [docs/architecture.md](docs/architecture.md).

## Privacy

Webcam frames are processed in memory and discarded; no image or video is ever written to disk.
Only the hand landmark coordinates (21 points per frame) of the examples you choose to record are
stored, in the local `user_data/` folder, which you can delete at any time. Nothing is sent over
the network (the one exception is the one-time hand model download).

## Known limitations

- Intel Macs are not supported (no MediaPipe build).
- Camera names show as "Camera 0", "Camera 1", ... because OpenCV can't read device names.
- One hand is tracked at a time.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](ROADMAP.md).
Released under the [MIT License](LICENSE).
