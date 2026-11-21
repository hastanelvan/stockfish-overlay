# Stockfish Overlay — Visual Chessboard Tracking & Analysis

A Python desktop prototype that detects a visible chessboard, follows moves from the standard starting position, and displays **up to three Stockfish move suggestions** as transparent, click-through arrows.

Built as a hands-on project in **computer vision, state estimation, constraint-based inference, desktop UI development, and engine integration**. It is intended for personal chess study and engine-permitted practice—not for unauthorized assistance in live competitive games.

> **Status:** Experimental / proof of concept. The repository includes source code and automated tests; behavior has not been comprehensively validated on real Windows installations, chess websites, screen layouts, or piece themes.

## What it does

- **Automatically locates an 8 × 8 board** on the primary monitor using image processing; no manual board-area selection is needed in automatic mode.
- **Calibrates using the standard starting position** to learn visual characteristics of the current piece set.
- **Tracks positions through legal-move constraints**, with support for captures, castling, en passant, promotions, and recovery from some missed frames.
- **Re-detects a board** if its window moves, is resized, or temporarily disappears; hides arrows while the board position is uncertain.
- **Runs Stockfish via the UCI protocol** using `python-chess` and requests up to three ranked alternatives with `MultiPV`.
- **Draws a non-interactive, translucent overlay**: green for the top suggestion; red for the second and third choices. These are *alternatives in the same position*, not a three-move sequence.
- **Offers manual mid-game resynchronization** after a successful starting-position calibration, with explicit side-to-move and castling-rights inputs.

The recognition system uses **image-processing heuristics and position constraints**, not a trained neural-network model. Stockfish supplies the chess search and evaluation; this repository does **not** implement a new chess engine.

## How it works

```text
Primary-screen capture
        |
        v
BoardLocator: locate / verify board geometry
        |
        v
BoardTracker: segment 64 squares, compare visual features
        |
        v
Legal transition search (python-chess) + uncertainty checks
        |
        v
Verified board position (FEN)
        |
        v
Stockfish UCI analysis (up to three MultiPV results)
        |
        v
PySide6 click-through arrow overlay
```

If an image is ambiguous or does not match a legal transition, the tracker retains the last confirmed position instead of blindly applying a move. This reduces some failure modes, but does not guarantee correct recognition across arbitrary themes or animations.

## Requirements

- **Windows** (the intended desktop target; Windows behavior has not been fully end-to-end tested)
- **Python 3.11 or 3.12**, with the Python launcher `py` available for the batch-file setup
- A separate, locally downloaded **Stockfish UCI executable** (not included in this repository)
- Dependencies in [`requirements.txt`](requirements.txt): PySide6, python-chess (`chess`), NumPy, and OpenCV
- A single visible, square, clearly patterned chessboard on the **primary monitor**

Download the Stockfish engine from [stockfishchess.org](https://stockfishchess.org/). Stockfish is a third-party engine distributed under its own license; see the official [Stockfish repository](https://github.com/official-stockfish/Stockfish).

## Installation and running

### Option A — Windows batch files

1. Clone or download this repository.
2. Extract it to a writable directory.
3. Run **`install.bat`** to create `.venv` and install Python dependencies.
4. Run **`run_auto.bat`** to launch the automatic tracker.
5. In the app, click **Dosya seç** (Choose file) and select the local Stockfish `.exe`.
6. Open a chessboard in its **standard initial position** and keep it fully visible on your primary display.
7. Set **Siyah taşlar altta** if Black is at the bottom, then click **Otomatik takibi başlat** (Start automatic tracking).

The display includes a toggle for showing suggestions only on the bottom player's turn and a configurable per-position analysis time.

### Option B — command line

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe auto_app.py
```

`run.bat` / `app.py` launch the separate manual-mode prototype. **`run_auto.bat` / `auto_app.py` are the main automatic-tracking entry points.**

## Project structure

```text
stockfish-overlay/
├── auto_app.py          # Main automatic-tracking UI and workflow
├── board_locator.py     # Board geometry detection and verification
├── tracker.py           # Screen-derived features and legal-move tracking
├── app.py               # Stockfish integration, overlay, manual-mode UI
├── tests/               # Synthetic tracker / locator / UI / engine tests
├── requirements.txt     # Python dependencies
├── install.bat          # Windows virtual-environment setup
├── run_auto.bat         # Automatic mode
├── run.bat              # Manual-mode prototype
├── TECHNICAL_NOTES.md   # Design decisions, validation and next steps
└── .gitignore
```

## Tests

The repository includes **15 test functions**, covering synthetic-board tracking, capture sequences, board orientation, castling, en passant, promotion, board detection and recovery, as well as a UI-flow integration test.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Some engine and UI integration tests require a real Stockfish executable and the `STOCKFISH_TEST_ENGINE` environment variable:

```powershell
$env:STOCKFISH_TEST_ENGINE = 'C:\path\to\stockfish.exe'
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests using that environment variable are skipped when it is absent. **The presence of tests is not a claim of real-world accuracy or exhaustive Windows validation.** See [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md) for the validation scope.

## Known limitations

- Initial calibration requires a **standard starting position**; arbitrary mid-game setup cannot be inferred reliably from a single screenshot. Mid-game resync requires prior calibration and manual entry of side to move and castling rights.
- A screenshot alone cannot establish move history, castling rights, or en-passant rights. Resync resets move history and related state; it may be inaccurate immediately after a double pawn move.
- Piece appearance and square colors are learned for the initial theme. Changing piece sets, themes, board orientation, or zoom may require restarting calibration.
- Designed for **one large board on the primary monitor**. Multiple simultaneous boards, unusual 3D/textured styles, overlays, very small boards, and long animations can lead to false detection or loss of tracking.
- Automatic catch-up is limited to short gaps (up to two plies in its legal-transition search); longer gaps may require resynchronization.
- End-to-end reliability on different Windows machines and real chess interfaces has **not** been established. Particular cases such as **captures** should be validated on the target board theme before treating results as reliable.
- Recognition uses heuristics, not machine learning; there are no training metrics or claims of model accuracy.

## Engineering focus and next steps

This project explores a useful general problem: combining **noisy visual observations** with a **structured, rule-constrained state space**. The main challenges are calibrating a board from pixels, detecting when observations are ambiguous, reconciling visual changes with legal moves, and keeping the UI responsive while an engine analyzes positions.

Useful next experiments: collect a labeled set of real board screenshots; measure move-detection accuracy separately for quiet moves and captures; add logs/replay fixtures for failures; quantify confidence thresholds; and compare template matching against a learned piece classifier.

For implementation details and a reproducible validation checklist, see [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).

## Responsible use

Use this project for **offline analysis, self-study, or settings that explicitly allow engine assistance**. Do not use it to gain unauthorized assistance in rated, competitive, or otherwise restricted live games. Respect the rules of the chess platform or event.

## Licensing

TThe original source code is licensed under the MIT License. Stockfish and third-party Python dependencies retain their own licenses.
