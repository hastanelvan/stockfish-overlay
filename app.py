"""Stockfish Overlay Demo — for personal analysis / permitted training games.

Windows/Qt transparent click-through overlay + manual FEN and board area input.
Automatic recognition of the on-screen pieces is intentionally not included.
"""
from __future__ import annotations

import sys
from pathlib import Path
from threading import Timer

import chess
import chess.engine
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class StrengthChooser(QComboBox):
    def __init__(self):
        super().__init__()
        self.addItem("Tam güç", None)
        for elo in [1320, *range(1400, 3200, 100), 3190]:
            self.addItem(f"{elo} Elo", elo)
        self.addItem("3500 Elo (desteklenmezse tam güç)", 3500)


def strength_options(engine, elo):
    options = engine.options
    rating = options.get("UCI_Elo")
    can_limit = rating is not None and "UCI_LimitStrength" in options
    if elo is not None and can_limit and rating.min <= elo <= rating.max:
        return {"UCI_LimitStrength": True, "UCI_Elo": elo}, f"{elo} Elo"
    if elo not in (None, 3500):
        if can_limit:
            raise ValueError(f"Bu motor {rating.min}–{rating.max} Elo aralığını destekliyor.")
        raise ValueError("Bu motor Elo ayarını desteklemiyor.")
    full = {}
    if "UCI_LimitStrength" in options:
        full["UCI_LimitStrength"] = False
    if "Skill Level" in options:
        full["Skill Level"] = options["Skill Level"].max
    label = "Tam güç"
    if elo is not None:
        if can_limit and elo > rating.max:
            label = f"Tam güç (motorun Elo üst sınırı {rating.max})"
        else:
            label = f"Tam güç (motor {elo} Elo ayarını desteklemiyor)"
    return full, label


def last_capture(board):
    if board.move_stack:
        previous = board.copy(stack=1)
        move = previous.pop()
        if previous.is_capture(move):
            return move
    return None


def natural_move_priority(board, move, last_move):
    if board.is_castling(move):
        return 4
    if last_move is not None and move.to_square == last_move.to_square and board.is_capture(move):
        return 3
    home_rank = 0 if board.turn == chess.WHITE else 7
    piece = board.piece_type_at(move.from_square)
    if piece in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) == home_rank:
        if chess.square_rank(move.to_square) != home_rank:
            return 2
    if piece == chess.PAWN and move.to_square in (chess.D4, chess.E4, chess.D5, chess.E5):
        return 1
    return int(board.gives_check(move))


def human_move(board, selected, scores, last_move=None):
    """Break small evaluation ties around Stockfish's strength-adjusted choice."""
    anchor_score = scores.get(selected)
    if anchor_score is None or anchor_score.is_mate():
        return selected
    anchor = anchor_score.pov(board.turn).score()
    preference = (natural_move_priority(board, selected, last_move), 0)
    result = selected
    for move, score in scores.items():
        if score is None or score.is_mate():
            continue
        cp = score.pov(board.turn).score()
        # Stay within 15 centipawns of the selected move, including at limited Elo.
        # Preserve a non-losing evaluation and a clear (one-pawn) advantage.
        if abs(cp - anchor) > 15 or (anchor >= 0 > cp) or (anchor >= 100 > cp):
            continue
        candidate = (natural_move_priority(board, move, last_move), -abs(cp - anchor))
        if candidate > preference:
            result, preference = move, candidate
    return result


class EngineSession:
    """Owned by AutoWindow; accessed by one analysis worker at a time."""
    def __init__(self):
        self.engine = None
        self.path = None

    def get(self, path):
        if self.engine is None or self.path != path or self.engine.returncode.done():
            self.close()
            self.engine = chess.engine.SimpleEngine.popen_uci(path, timeout=10.)
            self.path = path
        return self.engine

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                self.engine.close()
            self.engine = None


class Analyzer(QThread):
    success = Signal(str, str, str)  # UCI, SAN, evaluation
    failure = Signal(str)
    variations = Signal(object)  # Ranked alternative root moves, not successive plies.

    def __init__(self, fen: str, executable: str, seconds: float, session=None, multipv=1,
                 elo=None, human_like=False, last_move=None):
        super().__init__()
        self.multipv = multipv
        self.elo = elo
        self.effective_strength_text = None
        self.human_like = human_like
        self.last_move = last_move
        self.session = session
        self.fen = fen
        self.executable = executable
        self.seconds = seconds

    def run(self):
        engine = None
        try:
            position = chess.Board(self.fen)
            if position.is_game_over():
                self.failure.emit("Oyun bu konumda bitmiş; önerilecek hamle yok.")
                return
            engine = self.session.get(self.executable) if self.session else chess.engine.SimpleEngine.popen_uci(self.executable, timeout=10.0)
            legal_count = position.legal_moves.count()
            count = min(self.multipv, legal_count)
            options, strength_text = strength_options(engine, self.elo)
            # Stockfish's strength limiter considers at least four candidates.
            consider_alternatives = self.human_like or options.get("UCI_LimitStrength")
            lines = min(legal_count, max(count, 4)) if consider_alternatives else count
            with engine.analysis(position, chess.engine.Limit(time=self.seconds),
                                 multipv=lines, options=options) as analysis:
                self.effective_strength_text = strength_text
                # wait() exposes the selected bestmove but has no built-in timeout.
                timeout = Timer(self.seconds + 10.0, engine.close)
                timeout.start()
                try:
                    selected = analysis.wait().move
                    infos = analysis.multipv
                finally:
                    timeout.cancel()
            if selected is None or selected not in position.legal_moves:
                raise ValueError("Stockfish geçerli bir hamle döndürmedi.")

            # At limited Elo the selected move can differ from the first PV.
            # Keep the score attached to the move that actually received it.
            scores = {}
            for info in infos:
                pv = info.get('pv') or []
                if pv and pv[0] in position.legal_moves:
                    if not info.get('lowerbound') and not info.get('upperbound'):
                        scores[pv[0]] = info.get('score')
                    else:
                        scores.setdefault(pv[0], None)
            if self.human_like:
                selected = human_move(position, selected, scores, self.last_move)
            results = []
            for move in dict.fromkeys([selected, *scores]):
                score = scores.get(move)
                evaluation = 'Değerlendirme yok'
                if score is not None:
                    mate = score.white().mate()
                    cp = score.white().score()
                    evaluation = f'Mat {mate:+d}' if mate is not None else f'{cp / 100:+.2f}' if cp is not None else evaluation
                results.append(dict(uci=move.uci(), san=position.san(move), score=evaluation))
                if len(results) == count:
                    break
            self.variations.emit(results)
            first = results[0]
            self.success.emit(first['uci'], first['san'], first['score'])
        except (OSError, chess.engine.EngineError, chess.engine.EngineTerminatedError, ValueError) as error:
            self.failure.emit(str(error))
        except Exception as error:
            self.failure.emit(f"Beklenmeyen hata: {error}")
        finally:
            if engine is not None and self.session is None:
                try:
                    engine.quit()
                except Exception:
                    pass


class Overlay(QWidget):
    """Click-through, frameless overlay spanning the primary monitor."""

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.board_area: QRect | None = None
        self.uci: str | None = None
        self.moves = []
        self.black_bottom = False
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.hide()

    def set_board(self, rectangle: QRect):
        self.board_area = QRect(rectangle).normalized()
        self.update()

    def set_move(self, uci: str | None):
        self.set_moves([uci] if uci else [])

    def set_moves(self, moves):
        self.moves = [m for m in moves[:3] if len(m) >= 4]
        self.uci = self.moves[0] if self.moves else None
        self.update()

    def square_center(self, square: str) -> QPoint:
        assert self.board_area is not None
        file_idx = ord(square[0]) - ord("a")
        rank_idx = int(square[1]) - 1
        if self.black_bottom:
            col, row = 7 - file_idx, rank_idx
        else:
            col, row = file_idx, 7 - rank_idx
        r = self.board_area
        return QPoint(
            round(r.left() + (col + 0.5) * r.width() / 8 - self.geometry().left()),
            round(r.top() + (row + 0.5) * r.height() / 8 - self.geometry().top()),
        )

    def paintEvent(self, event):
        if not self.board_area or not self.moves:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Paint the best last so overlapping alternatives do not bury it.
        for rank in reversed(range(len(self.moves))):
            self.draw_arrow(painter, self.moves[rank], rank)
        painter.end()

    def draw_arrow(self, painter, uci, rank):
        p1, p2 = self.square_center(uci[:2]), self.square_center(uci[2:4])
        dx, dy = p2.x()-p1.x(), p2.y()-p1.y()
        length = (dx*dx+dy*dy)**.5
        if length < 1:
            return
        ux, uy = dx/length, dy/length
        size = min(self.board_area.width(), self.board_area.height())/8
        shaft = size * (.105 if rank == 0 else .085)
        head = min(size*.30, length*.35)
        wing = size * (.19 if rank == 0 else .16)
        start = size*.17
        # One filled silhouette avoids doubled opacity where shaft meets head.
        def point(t, offset):
            return QPointF(p1.x()+ux*t-uy*offset, p1.y()+uy*t+ux*offset)
        neck = length-head
        path = QPainterPath()
        path.moveTo(point(start, -shaft/2))
        path.lineTo(point(neck, -shaft/2))
        path.lineTo(point(neck, -wing))
        path.lineTo(point(length, 0))
        path.lineTo(point(neck, wing))
        path.lineTo(point(neck, shaft/2))
        path.lineTo(point(start, shaft/2))
        path.quadTo(point(start-shaft*.75,0),point(start,-shaft/2))
        path.closeSubpath()
        color = QColor(45, 210, 114, 155) if rank == 0 else QColor(239, 72, 83, 125 if rank == 1 else 100)
        painter.setPen(QPen(QColor(255,255,255,75), max(.7,size*.012), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(color)
        painter.drawPath(path)


class RegionPicker(QWidget):
    selected = Signal(QRect)
    cancelled = Signal()

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.begin: QPoint | None = None
        self.end: QPoint | None = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin = event.position().toPoint()
            self.end = self.begin
            self.update()

    def mouseMoveEvent(self, event):
        if self.begin is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.begin is not None:
            self.end = event.position().toPoint()
            r = QRect(self.begin, self.end).normalized()
            self.hide()
            if r.width() >= 120 and r.height() >= 120:
                r.translate(self.geometry().topLeft())
                self.selected.emit(r)
            else:
                self.cancelled.emit()
            self.deleteLater()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.cancelled.emit()
            self.deleteLater()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 65))
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        painter.drawText(24, 40, "Tahtanın sol üstünden sağ altına sürükle  |  ESC: iptal")
        if self.begin is not None and self.end is not None:
            rectangle = QRect(self.begin, self.end).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rectangle, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor("#4ef18a"), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rectangle)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stockfish Overlay — Eğitim / Analiz")
        self.setMinimumWidth(540)
        self.overlay = Overlay()
        self.picker: RegionPicker | None = None
        self.worker: Analyzer | None = None

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(QLabel("Kişisel analiz / izinli antrenman için şeffaf ok prototipi"))

        self.engine_path = QLineEdit()
        self.engine_path.setPlaceholderText(r"Örnek: C:\Stockfish\stockfish-windows-x64.exe")
        browse = QPushButton("Gözat…")
        browse.clicked.connect(self.choose_engine)
        engine_row = QHBoxLayout()
        engine_row.addWidget(self.engine_path, 1)
        engine_row.addWidget(browse)
        layout.addWidget(QLabel("Stockfish .exe dosyası:"))
        layout.addLayout(engine_row)

        self.fen_entry = QLineEdit(chess.STARTING_FEN)
        layout.addWidget(QLabel("Analiz edilecek pozisyon (FEN):"))
        layout.addWidget(self.fen_entry)

        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.1, 10.0)
        self.seconds.setSingleStep(0.25)
        self.seconds.setValue(0.75)
        self.seconds.setSuffix(" saniye")
        self.strength = StrengthChooser()
        self.strength.currentIndexChanged.connect(self.strength_changed)
        self.human_like = QCheckBox("İnsana benzer hamleler")
        self.human_like.setToolTip("Yakın değerlendirmeli hamlelerde doğal gelişimi, roku ve geri almayı tercih eder.")
        self.human_like.toggled.connect(self.strength_changed)
        self.flip = QCheckBox("Siyah taşlar altta (tahtayı çevir)")
        self.flip.toggled.connect(self.orientation_changed)
        settings = QFormLayout()
        settings.addRow("Analiz süresi:", self.seconds)
        settings.addRow("Motor gücü:", self.strength)
        settings.addRow(self.human_like)
        settings.addRow("Yön:", self.flip)
        layout.addLayout(settings)

        board_button = QPushButton("1. Ekranda tahtayı seç")
        board_button.clicked.connect(self.choose_area)
        self.analyze_button = QPushButton("2. Stockfish ile analiz et → oku göster")
        self.analyze_button.clicked.connect(self.analyze)
        hide_button = QPushButton("Oku gizle / göster")
        hide_button.clicked.connect(self.toggle_arrow)
        layout.addWidget(board_button)
        layout.addWidget(self.analyze_button)
        layout.addWidget(hide_button)

        self.best_move = None
        self.last_capture_position = None
        self.uci_entry = QLineEdit()
        self.uci_entry.setPlaceholderText("Örn. e2e4 (hamleyi elle uygula)")
        apply_move = QPushButton("Hamleyi uygula")
        apply_move.clicked.connect(self.apply_uci)
        self.uci_entry.returnPressed.connect(self.apply_uci)
        move_row = QHBoxLayout()
        move_row.addWidget(self.uci_entry)
        move_row.addWidget(apply_move)
        layout.addWidget(QLabel("İsteğe bağlı: konumu ilerletmek için UCI hamlesi gir"))
        layout.addLayout(move_row)

        self.status_label = QLabel("Tahtayı seç, Stockfish yolunu göster ve analiz et.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        notes = QLabel("Not: Bu ilk sürüm ekran görüntüsünden taş tanımaz; FEN'i elle girersin. "
                       "Yalnızca birincil monitörde tahta seçimi desteklenir.")
        notes.setWordWrap(True)
        layout.addWidget(notes)
        self.setCentralWidget(root)

    def choose_engine(self):
        path, _ = QFileDialog.getOpenFileName(self, "Stockfish çalıştırılabilir dosyası", "", "Windows uygulaması (*.exe);;Tüm dosyalar (*)")
        if path:
            self.engine_path.setText(path)

    def orientation_changed(self, checked):
        self.overlay.black_bottom = checked
        self.overlay.update()

    def strength_changed(self):
        if self.best_move:
            self.best_move = None
            self.overlay.set_move(None)
            self.overlay.hide()
            self.analyze()

    def choose_area(self):
        self.overlay.hide()
        self.picker = RegionPicker()
        self.picker.selected.connect(self.area_selected)
        self.picker.cancelled.connect(lambda: self.status_label.setText("Tahta seçimi iptal edildi."))
        self.picker.show()
        self.picker.activateWindow()

    def area_selected(self, area: QRect):
        self.overlay.set_board(area)
        self.status_label.setText(f"Tahta seçildi: {area.width()} × {area.height()} px. Şimdi analiz et.")
        if self.overlay.uci:
            self.overlay.show()

    def toggle_arrow(self):
        if self.overlay.isVisible():
            self.overlay.hide()
        elif self.overlay.board_area is not None and self.overlay.uci:
            self.overlay.show()
        else:
            self.status_label.setText("Önce tahta seçip bir hamle analiz et.")

    def analyze(self):
        path = self.engine_path.text().strip().strip('"')
        if not Path(path).is_file():
            QMessageBox.warning(self, "Stockfish bulunamadı", "Önce Stockfish .exe dosyasını seç.")
            return
        if self.overlay.board_area is None:
            QMessageBox.warning(self, "Tahta seçilmedi", "Önce ekrandan satranç tahtasını seç.")
            return
        try:
            board = chess.Board(self.fen_entry.text().strip())
            if not board.is_valid():
                raise ValueError("FEN'deki pozisyon geçerli değil.")
        except ValueError as exc:
            QMessageBox.warning(self, "FEN hatası", str(exc))
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.analyze_button.setEnabled(False)
        self.strength.setEnabled(False)
        self.human_like.setEnabled(False)
        self.status_label.setText("Stockfish analiz ediyor...")
        fen = board.fen()
        last_move = None
        if self.last_capture_position and self.last_capture_position[0] == fen:
            last_move = self.last_capture_position[1]
        self.worker = Analyzer(fen, path, self.seconds.value(), elo=self.strength.currentData(),
                               human_like=self.human_like.isChecked(), last_move=last_move)
        self.worker.success.connect(self.analysis_done)
        self.worker.failure.connect(self.analysis_failed)
        self.worker.finished.connect(lambda: self.analyze_button.setEnabled(True))
        self.worker.finished.connect(lambda: self.strength.setEnabled(True))
        self.worker.finished.connect(lambda: self.human_like.setEnabled(True))
        self.worker.start()

    def analysis_done(self, uci, san, score):
        self.best_move = uci
        self.overlay.set_move(uci)
        self.overlay.show()
        self.status_label.setText(f"{self.worker.effective_strength_text} | Önerilen hamle: {san} ({uci}) | {score}")

    def analysis_failed(self, message):
        self.status_label.setText(f"Analiz hatası: {message}")

    def apply_uci(self):
        try:
            board = chess.Board(self.fen_entry.text().strip())
            if not board.is_valid():
                raise ValueError("FEN'deki pozisyon geçerli değil.")
            move = chess.Move.from_uci(self.uci_entry.text().strip().lower())
            if move not in board.legal_moves:
                raise ValueError("Bu konum için yasal bir hamle değil.")
            capture = board.is_capture(move)
            board.push(move)
            fen = board.fen()
            self.last_capture_position = (fen, move) if capture else None
            self.fen_entry.setText(fen)
            self.uci_entry.clear()
            self.overlay.set_move(None)
            self.overlay.hide()
            self.status_label.setText("Hamle uygulandı. Güncel konum için yeniden analiz et.")
        except ValueError as exc:
            QMessageBox.warning(self, "Hamle hatası", str(exc))

    def closeEvent(self, event):
        self.overlay.close()
        if self.picker:
            self.picker.close()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(12000)  # engine subprocess is allowed to finish before closing
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
