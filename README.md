# Stockfish Overlay — Visual Chessboard Tracking & Analysis

A Python desktop prototype that reads a visible chess position, follows subsequent moves, and displays **up to three Stockfish suggestions** as transparent, click-through arrows. You can open an analysis board at the opening, middlegame, or endgame.

Built as a hands-on project in **computer vision, state estimation, constraint-based inference, desktop UI development, and engine integration**. It is intended for personal chess study and engine-permitted practice—not for unauthorized assistance in live competitive games.

> **Status:** Experimental / proof of concept. The repository includes source code and automated tests; behavior has not been comprehensively validated on real Windows installations, chess websites, screen layouts, or piece themes.

## What it does

- **Automatically locates an 8 × 8 board** on the primary monitor using image processing; no manual board-area selection is needed in automatic mode.
- **Recognizes the standard Cburnett 2D pieces used by Lichess** out of the box. For another 2D piece set, import a separate screenshot of its starting arrangement once. Your current game can stay on the move you want to study.
- **Tracks positions through legal-move constraints**, with support for captures, castling, en passant, promotions, and recovery from some missed frames.
- **Re-detects a board** if its window moves, is resized, or temporarily disappears; hides arrows while the board position is uncertain.
- **Offers an Elo chooser and optional human-like moves.** Green shows the recommendation at the chosen settings. Red arrows show alternatives for the same turn.
- **Reads the current position again** after a jump in the game or a sustained tracking mismatch. You can also request a fresh read and correct the side to move or castling rights.

The recognition system uses **image-processing heuristics and position constraints**, not a trained neural-network model. Stockfish supplies the chess search and evaluation; this repository does **not** implement a new chess engine.

## How it works

```text
Primary-screen capture
        |
        v
BoardLocator: locate / verify board geometry
        |
        v
BoardTracker: recognize the visible pieces from built-in or saved templates
        |
        v
Legal transition search (python-chess) + uncertainty checks
        |
        v
Verified board position (FEN)
        |
        v
Stockfish UCI analysis at the selected strength and style
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
6. Open the position you want to study and keep the whole board visible on your primary display.
7. Set **Siyah taşlar altta** if Black is at the bottom, then click **Otomatik takibi başlat** (Start automatic tracking).

If the app has difficulty reading a different piece set, expand **Konumu düzelt veya yeniden oku**, select **Taş teması için örnek resim seç…**, and import a separate screenshot of that set in its standard starting arrangement. A tight crop of the 8 × 8 board works well. The app saves the theme for future sessions.

**Motor gücü** sets the target playing strength. For example, 2500 uses Stockfish's native Elo setting when the engine supports it. The 3500 option uses that target on engines that support it and full strength on engines with a lower limit. Stockfish 19's calibrated settings span 1320–3190 Elo. Actual playing strength also depends on analysis time and hardware. Enable **İnsana benzer hamleler** to favor natural moves such as castling, recapturing, or developing a piece when the engine rates them within 0.15 pawns of its selected move.

When a game opens mid-position, last-move highlights or a king in check can reveal whose turn it is. If both turns are plausible, the app shows White as an assumption and can correct it on the next observed move. For an immediate correction, choose the side to move and click **Yeniden oku ve devam et**. The same panel accepts known castling rights such as `Kq`. You can also show suggestions only on the bottom player's turn and adjust the analysis time.

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
├── piece_templates.py   # Built-in Cburnett references and screenshot import
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

The tests cover synthetic board tracking, midgame recognition, reference import, capture sequences, board orientation, castling, en passant, promotion, board detection and recovery, engine strength and style, and the automatic UI flow.

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

- An independently recognized midgame position has unknown move counts and repetition history. The tracker assumes no castling rights unless you supply them. It can recover an en-passant target from a recognizable last double-pawn push. Positions tracked continuously retain their move history.
- A new piece theme can require a new reference image. Use the orientation setting to match the visible board.
- Designed for **one large board on the primary monitor**. Multiple simultaneous boards, unusual 3D/textured styles, overlays, very small boards, and long animations can lead to false detection or loss of tracking.
- Legal-move catch-up covers up to two plies. After a longer jump the app attempts to identify the visible position again.
- End-to-end reliability on different Windows machines and real chess interfaces has **not** been established. Particular cases such as **captures** should be validated on the target board theme before treating results as reliable.
- Recognition uses heuristics, not machine learning; there are no training metrics or claims of model accuracy.

## Engineering focus and next steps

This project explores a useful general problem: combining **noisy visual observations** with a **structured, rule-constrained state space**. The main challenges are calibrating a board from pixels, detecting when observations are ambiguous, reconciling visual changes with legal moves, and keeping the UI responsive while an engine analyzes positions.

Useful next experiments: collect a labeled set of real board screenshots; measure move-detection accuracy separately for quiet moves and captures; add logs/replay fixtures for failures; quantify confidence thresholds; and compare template matching against a learned piece classifier.

For implementation details and a reproducible validation checklist, see [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).

## Responsible use

Use this project for **offline analysis, self-study, or settings that explicitly allow engine assistance**. Do not use it to gain unauthorized assistance in rated, competitive, or otherwise restricted live games. Respect the rules of the chess platform or event.

## Licensing

The Stockfish binary is **not bundled**.

The original source code is licensed under the MIT License. Stockfish and third-party Python dependencies retain their own licenses.
