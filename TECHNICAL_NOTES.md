# Technical notes — board tracking and uncertainty

These notes describe the code currently in this repository. They are not a claim of production-level testing or original chess-engine research.

## 1. Problem formulation

A single screen image provides an imperfect observation of a chess position. It does **not** directly provide turn history, castling eligibility, en-passant rights, or the actual move played while the board animates. The implementation therefore treats image recognition as **state estimation constrained by chess rules**, rather than inferring an unconstrained position independently on each frame.

## 2. Processing pipeline

**Board discovery (`board_locator.py`).** `BoardLocator` searches a primary-screen capture for the recurring geometry/colors of a square chessboard. It scores candidate regions for chessboard-like structure and refines the rectangle. The automatic UI asks for consecutive consistent detections before accepting a region.

**Calibration and features (`tracker.py`).** `BoardTracker` starts from a known initial chess position. It segments the accepted board into 64 squares, estimates the square background from corners, and extracts simple 24 × 24 foreground silhouettes and brightness features. Initial piece appearances become templates. These are handcrafted features, not learned embeddings.

**State update (`tracker.py`).** New observations are compared against the previous confirmed position. The tracker enumerates legal move transitions through `python-chess`, including short two-ply catch-up, and uses appearance/occupancy consistency to choose plausible updates. An uncertain observation is allowed to fail without changing the confirmed state.

**Engine and overlay (`app.py`, `auto_app.py`).** `Analyzer` asks a local UCI-compatible Stockfish process for ranked alternatives. The automatic UI uses a persistent `EngineSession` and draws the results as translucent arrows using PySide6. Analysis is performed in a worker rather than blocking the Qt event loop.

## 3. Failure-handling decisions

- **Visual ambiguity:** suppress suggestions and retain the last verified board until a later frame is usable.
- **Repositioning or resize:** search for the board again; recheck its location before restoring overlay arrows.
- **Short missed sequence:** consider legal one- or two-ply transitions when matching the next observation.
- **Longer drift:** provide manual resynchronization, but only after initial template calibration and with manually specified side to move and castling rights.
- **Stale engine output:** reject analysis that does not correspond to the current board FEN.

## 4. Evidence and validation scope

The repository includes 15 `unittest` test methods under `tests/`:

- `test_tracker.py`: synthetic board rendering and tracking cases, including captures, ambiguous captures, promotions, castling, en passant, orientation, highlights, resynchronization, and retry behavior.
- `test_locator.py`: synthetic-board localization, relocation, resizing, obscuring, and highlights.
- `test_engine.py`: UCI MultiPV and session reuse, conditional on an external Stockfish binary.
- `test_auto_ui.py`: Qt automatic-tracking flow, conditional on an external Stockfish binary.

These tests primarily use **synthetic boards**. They do not establish performance on live websites or unseen visual themes. Engine-backed tests are skipped if `STOCKFISH_TEST_ENGINE` is not configured.

For a reproducible check on Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:STOCKFISH_TEST_ENGINE = 'C:\path\to\stockfish.exe'
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Current verification note (documentation preparation):** Python source files passed a syntax compilation check. Full test execution was not possible in the preparation environment because `chess` and `PySide6` could not be installed there. Test results should be reported only after running them locally.

## 5. Experiment ideas

1. Create a small, labeled screenshot corpus with moves, captures, promotions, and board themes. Keep training/development and held-out validation screenshots separate.
2. Report move-tracking success rates, false updates, and recovery latency by move type; do not collapse all outcomes to one unqualified accuracy number.
3. Record ambiguous images and replay them as deterministic regression fixtures for the tracker.
4. Compare the current heuristic template score against an image-classification baseline under theme and scale changes.
5. Profile board re-detection and Stockfish analysis latency independently, then evaluate UI responsiveness and capture stability on Windows.

## 6. Authorship and dependencies

This is a Python application that **integrates** Stockfish; it is not a new engine or an ML research model. All strength and chess-evaluation capability come from the externally installed Stockfish binary. `python-chess` provides legal-move operations and the UCI interface; OpenCV/NumPy process board imagery; PySide6 renders the GUI.

If documenting authorship for an application or interview, distinguish between implementation decisions you personally made, assistance from coding tools, and components inherited from third-party dependencies.
