# Technical notes — board tracking and uncertainty

These notes describe the code currently in this repository. They are not a claim of production-level testing or original chess-engine research.

## 1. Problem formulation

A single screen image provides an imperfect observation of a chess position. It does **not** directly provide turn history, castling eligibility, en-passant rights, or the actual move played while the board animates. The implementation therefore treats image recognition as **state estimation constrained by chess rules**, rather than inferring an unconstrained position independently on each frame.

## 2. Processing pipeline

**Board discovery (`board_locator.py`).** `BoardLocator` searches a primary-screen capture for the recurring geometry/colors of a square chessboard. It scores candidate regions for chessboard-like structure and refines the rectangle. The automatic UI asks for consecutive consistent detections before accepting a region.

**Piece recognition (`tracker.py`, `piece_templates.py`).** `BoardTracker` estimates the background of each square from its corners and extracts 24 × 24 foreground silhouettes and brightness features. It compares the pieces with Cburnett references rendered from the SVGs distributed with `python-chess` and with the user's saved theme, if available. Each candidate placement must pass per-square confidence checks and chess position validation. A separate screenshot of a custom set's initial arrangement can supply its twelve piece identities. If the visible game itself happens to show a recognizable starting formation, the tracker can also learn that theme directly.

**State on a fresh read.** When both turns are legal, a recognizable last-move highlight can indicate which side just played. A verified two-square pawn advance can also supply the en-passant square. Without a unique turn or reliable highlight, the tracker marks White as assumed in a midgame position and considers either side for the next observed transition. Castling rights default to none on a midgame read. The correction controls accept a side and known rights. A fresh read starts a new move history.

**State update (`tracker.py`).** New observations are compared against the previous confirmed position. The tracker enumerates legal move transitions through `python-chess`, including short two-ply catch-up, and uses appearance/occupancy consistency to choose plausible updates. An uncertain observation is allowed to fail without changing the confirmed state.

**Engine and overlay (`app.py`, `auto_app.py`).** A worker runs a time-limited Stockfish search on the automatic UI's persistent engine session. Numeric Elo targets use `UCI_LimitStrength` and the executable's advertised `UCI_Elo` range. Selecting 3500 on an engine with a lower ceiling switches off the strength limiter. The green arrow uses the final UCI `bestmove`, which can differ from the top-ranked line at reduced strength. Up to two additional arrows show distinct legal alternatives.

**Human-like option.** At the selected strength the worker examines up to four candidate lines. It can prefer castling, a known recapture, development, a central pawn move or a check within 15 centipawns of the strength-adjusted choice. Scores are compared from the side to move. The tie-break preserves a reported mate, a nonnegative score, and an advantage of at least one pawn when the original choice has one. If the selected move lacks a usable score, it remains the recommendation.

## 3. Failure-handling decisions

- **Visual ambiguity:** suppress suggestions and retain the last verified board until a later frame is usable.
- **Repositioning or resize:** search for the board again; recheck its location before restoring overlay arrows.
- **Short missed sequence:** consider legal one- or two-ply transitions when matching the next observation.
- **Longer drift.** Re-read the visible pieces after sustained mismatch and retry when the image is unclear. The user can also request a fresh read, correct state, or import a theme reference.
- **Stale engine output.** When the board or analysis settings change, discard outstanding results for the earlier request.

## 4. Evidence and validation scope

The `tests/` directory covers the following behavior.

- `test_tracker.py`: synthetic board rendering and tracking cases, including captures, ambiguous captures, promotions, castling, en passant, orientation, highlights, resynchronization, and retry behavior.
- `test_locator.py`: synthetic-board localization, relocation, resizing, obscuring, and highlights.
- `test_piece_templates.py` covers the built-in visual references and screenshot import.
- `test_engine.py` covers Elo settings, selection of the final UCI move, human-like tie-breaking and session reuse. The engine-backed case uses an external Stockfish binary.
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

## 5. Experiment ideas

1. Create a small, labeled screenshot corpus with moves, captures, promotions, and board themes. Keep training/development and held-out validation screenshots separate.
2. Report move-tracking success rates, false updates, and recovery latency by move type; do not collapse all outcomes to one unqualified accuracy number.
3. Record ambiguous images and replay them as deterministic regression fixtures for the tracker.
4. Compare the current heuristic template score against an image-classification baseline under theme and scale changes.
5. Profile board re-detection and Stockfish analysis latency independently, then evaluate UI responsiveness and capture stability on Windows.

## 6. Authorship and dependencies

This is a Python application that **integrates** Stockfish; it is not a new engine or an ML research model. All strength and chess-evaluation capability come from the externally installed Stockfish binary. `python-chess` provides legal-move operations and the UCI interface; OpenCV/NumPy process board imagery; PySide6 renders the GUI.

If documenting authorship for an application or interview, distinguish between implementation decisions you personally made, assistance from coding tools, and components inherited from third-party dependencies.
