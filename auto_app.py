"""Stockfish overlay with no manual FEN; tracks a standard game from move 1.

Designed for personal analysis and permitted engine-assisted practice.
Do not use as engine assistance in live competitive online chess.
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Qt, QRect, QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout,
    QWidget, QLineEdit, QComboBox, QFrame, QGroupBox, QScrollArea,
)

from app import EngineSession, Analyzer, Overlay, RegionPicker
from tracker import BoardTracker, CalibrationError
from board_locator import BoardLocator, BoardRegion, DetectionWorker


class AutoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Stockfish Overlay — Otomatik Antrenman')
        self.setMinimumWidth(550)
        self.overlay = Overlay()
        self.picker = None
        self.detect_worker = None
        self.locator_generation = 0
        self.running = False
        self.resync_retry = False
        self.board_visible = False
        self.region = None
        self.candidate_region = None
        self.candidate_count = 0
        self.last_detection = 0.
        self.detect_geometry = None
        self.last_capture_ok = False
        self.tracker = None
        self.worker = None
        self.engine_session = EngineSession()
        self.generation = 0
        self.last_analyzed_fen = None
        self.pending_analysis = False
        self.excluded_from_capture = False
        self.busy_capture = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.setInterval(100)

        self.settings = QSettings('OverlayLab', 'StockfishOverlay')
        self.setMinimumWidth(580)
        self.resize(640, 870)
        self.setStyleSheet("""
            QMainWindow, QScrollArea, QWidget#canvas { background: #0e131c; color: #dbe4f0; }
            QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #dbe4f0; }
            QLabel#title { font-size: 29px; font-weight: 700; color: #f4f8ff; }
            QLabel#eyebrow { color: #78dca7; font-size: 11px; font-weight: 700; }
            QLabel#muted { color: #8999af; }
            QFrame#card { background: #171f2c; border: 1px solid #28354a; border-radius: 14px; }
            QFrame#card QLabel { background: transparent; border: none; }
            QLabel#section { font-size: 15px; font-weight: 600; color: #f4f8ff; }
            QLineEdit, QComboBox, QDoubleSpinBox { background: #101824; border: 1px solid #34435b;
                border-radius: 7px; padding: 10px; selection-background-color: #286b50; }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: #78dca7; }
            QPushButton { background: #26344a; border: 1px solid #34435b; border-radius: 8px;
                padding: 11px 14px; font-weight: 600; }
            QPushButton:hover { background: #33465f; }
            QPushButton#primary { background: #78dca7; color: #10271d; border: none; }
            QPushButton#primary:hover { background: #98ebbf; }
            QPushButton:disabled { background: #1b2533; color: #56677e; border-color: #28354a; }
            QCheckBox { spacing: 9px; padding: 4px 0; }
            QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #53647d; border-radius: 4px; background: #101824; }
            QCheckBox::indicator:checked { background: #78dca7; border-color: #78dca7; }
            QGroupBox { border: 1px solid #28354a; border-radius: 10px; margin-top: 14px; padding-top: 16px; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 7px; }
            QScrollArea { border: none; }
            QScrollBar:vertical { background: #101824; width: 8px; }
            QScrollBar::handle:vertical { background: #34435b; border-radius: 4px; min-height: 24px; }
        """)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        w.setObjectName('canvas')
        lay = QVBoxLayout(w)
        lay.setContentsMargins(26, 26, 26, 26)
        lay.setSpacing(18)

        def label(text, name='muted', wrap=True):
            item = QLabel(text)
            item.setObjectName(name)
            item.setWordWrap(wrap)
            return item

        def card(title):
            frame = QFrame()
            frame.setObjectName('card')
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(18, 18, 18, 18)
            layout.setSpacing(12)
            layout.addWidget(label(title, 'section'))
            lay.addWidget(frame)
            return layout

        lay.addWidget(label('STOCKFISH / ANALİZ YARDIMCISI', 'eyebrow'))
        lay.addWidget(label('Tahtaya odaklan.', 'title'))
        lay.addWidget(label('Otomatik tahta takibi · Canlı analiz · Üç hamle önerisi'))
        engine_card = card('01  Motor ve takip')
        self.engine = QLineEdit(self.settings.value('engine', ''))
        self.engine.setPlaceholderText('Stockfish dosyasını seç…')
        browse = QPushButton('Dosya seç')
        browse.clicked.connect(self.browse_engine)
        row = QHBoxLayout()
        row.addWidget(self.engine, 1)
        row.addWidget(browse)
        engine_card.addLayout(row)
        self.black_bottom = QCheckBox('Siyah taşlar altta')
        self.black_bottom.toggled.connect(self.set_orientation)
        self.only_bottom = QCheckBox('Önerileri yalnızca benim sıramda göster')
        self.only_bottom.setChecked(True)
        self.only_bottom.toggled.connect(self.sync_arrow_turn)
        engine_card.addWidget(self.black_bottom)
        engine_card.addWidget(self.only_bottom)
        row = QHBoxLayout()
        row.addWidget(label('Hamle başına analiz süresi'))
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(.1, 3.)
        self.seconds.setValue(.35)
        self.seconds.setSingleStep(.05)
        self.seconds.setSuffix(' sn')
        row.addWidget(self.seconds)
        engine_card.addLayout(row)
        engine_card.addWidget(label('Tahtayı başlangıç konumunda görünür tut. Alan seçmen gerekmez; yeri veya boyutu değiştiğinde yeniden bulunur.'))
        self.start_button = QPushButton('Otomatik takibi başlat')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button = QPushButton('Durdur')
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)
        actions = QHBoxLayout()
        actions.addWidget(self.start_button, 1)
        actions.addWidget(self.stop_button)
        engine_card.addLayout(actions)

        live_card = card('02  Canlı durum')
        legend = QLabel('<span style="color:#78dca7">● En iyi hamle</span> &nbsp; '
                        '<span style="color:#f07883">● 2. ve 3. alternatif</span>')
        live_card.addWidget(legend)
        self.status = QLabel('Hazır. Tahtayı açıp otomatik takibi başlat.')
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(56)
        live_card.addWidget(self.status)
        live_card.addWidget(label('Tahta görünmezse oklar gizlenir. Belirsiz görüntülerde son doğrulanmış konum korunur ve okuma yeniden denenir.'))

        recovery = QGroupBox('Maç ortasında konumu yeniden oku')
        recovery.setCheckable(True)
        recovery.setChecked(False)
        recovery_layout = QVBoxLayout(recovery)
        body = QWidget()
        recovery_layout.addWidget(body)
        body_layout = QVBoxLayout(body)
        self.resync_turn = QComboBox()
        self.resync_turn.addItems(['Beyaz oynayacak', 'Siyah oynayacak'])
        self.castling = QLineEdit('-')
        self.castling.setPlaceholderText('KQkq veya -')
        form = QFormLayout()
        form.addRow('Hamle sırası', self.resync_turn)
        form.addRow('Kalan rok hakları', self.castling)
        body_layout.addLayout(form)
        self.resync_button = QPushButton('Yeniden oku ve devam et')
        self.resync_button.clicked.connect(self.resync_position)
        body_layout.addWidget(self.resync_button)
        body_layout.addWidget(label('Aynı oturumda başlangıç kalibrasyonu gerekir. Geçerken alma hakkı ve hamle geçmişi yeniden okumada sıfırlanır.'))
        recovery.toggled.connect(body.setVisible)
        body.hide()
        lay.addWidget(recovery)
        lay.addStretch()
        scroll.setWidget(w)
        self.setCentralWidget(scroll)
        self.resync_retry = False

    def resync_position(self):
        if self.tracker is None:
            QMessageBox.warning(self, 'Kalibrasyon gerekli', 'Taş görsellerini öğrenmek için önce başlangıç konumunda bir kez kalibre et. Takip açıkken maç ortasında yeniden okutabilirsin.')
            return
        if not self.board_visible:
            self.status.setText('Yeniden okumak için tahtanın bulunmasını bekle.')
            return
        self.resync_retry = True
        try:
            self.tracker.resync(self._screenshot(), self.resync_turn.currentIndex() == 0,
                                self.castling.text().strip() or '-')
        except (CalibrationError, ValueError) as exc:
            self.status.setText('Yeniden okuma bekleniyor: ' + str(exc))
            self.overlay.hide()
            return
        self.resync_retry = False
        self.generation += 1
        self.overlay.set_move(None)
        self.overlay.hide()
        self.last_analyzed_fen = None
        self.pending_analysis = True
        self.status.setText('Mevcut tahta yeniden okundu. Takip devam ediyor; hamle geçmişi sıfırlandı.')
        self.request_analysis()

    def browse_engine(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Stockfish .exe dosyasını seç', '', 'Exe (*.exe);;Tüm dosyalar (*)')
        if path:
            self.engine.setText(path)
            self.settings.setValue('engine', path)

    def set_orientation(self, is_flipped):
        if self.tracker:
            self.stop_tracking()
            self.status.setText('Tahta yönü değişti. Yeniden kalibre et.')
        self.overlay.black_bottom = is_flipped
        self.overlay.update()

    @staticmethod
    def _rgb(pix):
        if pix.isNull():
            raise CalibrationError('Ekran görüntüsü alınamadı.')
        image = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        data = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
        return data[:, :image.width()*4].reshape(image.height(), image.width(), 4)[:, :, :3].copy()

    def _capture(self, whole=False):
        should_restore = self.overlay.isVisible() and not self.excluded_from_capture
        if should_restore:
            self.overlay.hide()
            QApplication.processEvents()
        try:
            screen = QApplication.primaryScreen()
            if whole:
                pix = screen.grabWindow(0)
            else:
                area = self.overlay.board_area
                if area is None:
                    raise CalibrationError('Tahta aranıyor.')
                origin = screen.geometry().topLeft()
                pix = screen.grabWindow(0, area.left()-origin.x(), area.top()-origin.y(), area.width(), area.height())
            return self._rgb(pix)
        finally:
            if should_restore and self.board_visible:
                self.overlay.show()

    def begin_detection(self):
        if self.detect_worker is not None:
            return
        self.last_detection = time.monotonic()
        image = self._capture(whole=True)
        geometry = QRect(QApplication.primaryScreen().geometry())
        self.detect_geometry = (geometry, image.shape[1], image.shape[0])
        worker = DetectionWorker(image, self.region, self.locator_generation, self)
        self.detect_worker = worker
        worker.found.connect(self.detection_done)
        worker.failed.connect(self.detection_failed)
        worker.finished.connect(self.detection_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def detection_failed(self, error, generation):
        if generation == self.locator_generation:
            self.board_visible = False
            self.overlay.hide()
            self.status.setText('Tahta arama hatası: ' + error)

    def detection_finished(self):
        self.detect_worker = None

    def detection_done(self, region, generation):
        if generation != self.locator_generation or not self.running:
            return
        if region is None:
            self.candidate_region = None
            self.candidate_count = 0
            self.board_visible = False
            self.overlay.hide()
            self.status.setText('Tahta görünmüyor; ekranda yeniden belirmesi bekleniyor.')
            return
        old = self.candidate_region
        same = old is not None and max(abs(old.x-region.x), abs(old.y-region.y), abs(old.size-region.size)) <= 4
        self.candidate_count = self.candidate_count + 1 if same else 1
        self.candidate_region = region
        if self.candidate_count < 2:
            self.board_visible = False
            self.overlay.hide()
            return
        geometry, width, height = self.detect_geometry
        sx, sy = geometry.width()/width, geometry.height()/height
        rect = QRect(geometry.x()+round(region.x*sx), geometry.y()+round(region.y*sy),
                     round(region.size*sx), round(region.size*sy))
        moved = self.overlay.board_area != rect
        if moved:
            self.overlay.hide()
        self.region = region
        self.overlay.setGeometry(geometry)
        self.overlay.set_board(rect)
        self.board_visible = True
        try:
            image = self._screenshot()
            if BoardLocator.confidence(image) < .88:
                self.board_visible = False
                self.overlay.hide()
                return
            if self.tracker is None:
                self.tracker = BoardTracker(image, black_bottom=self.black_bottom.isChecked())
                self.pending_analysis = True
                self.status.setText('Tahta otomatik bulundu ve kalibre edildi. Takip sürüyor.')
            elif moved:
                self.status.setText('Tahtanın yeni konumu bulundu. Oyun takibi devam ediyor.')
            # A hidden/moved board can have advanced. Observe before restoring arrows.
            self.last_capture_ok = False
        except (CalibrationError, ValueError) as exc:
            self.overlay.hide()
            self.status.setText('Başlangıç dizilişi bekleniyor: ' + str(exc))

    def _screenshot(self):
        return self._capture()

    def _try_capture_exclusion(self):
        self.excluded_from_capture = False
        if sys.platform != 'win32':
            return
        try:
            self.overlay.show()
            hwnd = int(self.overlay.winId())
            self.excluded_from_capture = bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11))
        except (AttributeError, OSError, ValueError):
            pass
        finally:
            self.overlay.hide()

    def start_tracking(self):
        if not Path(self.engine.text().strip().strip('"')).is_file():
            QMessageBox.warning(self, 'Stockfish yok', 'Önce Stockfish .exe dosyasını seç.')
            return
        self.stop_tracking()
        self.running = True
        self.last_detection = 0.
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.engine.setEnabled(False)
        self._try_capture_exclusion()
        self.status.setText('Tahta otomatik aranıyor. Başlangıç dizilişini görünür tut.')
        self.timer.start()
        self.begin_detection()

    def stop_tracking(self):
        self.timer.stop()
        self.generation += 1
        self.locator_generation += 1
        self.running = False
        self.resync_retry = False
        self.board_visible = False
        self.region = None
        self.candidate_region = None
        self.candidate_count = 0
        self.tracker = None
        self.overlay.hide()
        self.overlay.set_move(None)
        self.engine.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pending_analysis = False
        self.last_analyzed_fen = None

    def current_player_is_bottom(self):
        if not self.tracker:
            return False
        bottom_color = not self.black_bottom.isChecked()  # bool True=white
        return self.tracker.board.turn == bottom_color

    def sync_arrow_turn(self):
        if self.only_bottom.isChecked() and not self.current_player_is_bottom():
            self.overlay.hide()
        elif self.tracker:
            self.last_analyzed_fen = None
            self.pending_analysis = True
            self.request_analysis()

    def tick(self):
        if self.busy_capture or not self.running:
            return
        self.busy_capture = True
        try:
            if time.monotonic() - self.last_detection >= .65:
                self.begin_detection()
            if self.tracker is None or not self.board_visible:
                return
            image = self._screenshot()
            if BoardLocator.confidence(image) < .88:
                self.board_visible = False
                self.last_capture_ok = False
                self.overlay.hide()
                self.status.setText('Tahta yer değiştirdi veya kapandı; otomatik aranıyor.')
                return
            if self.resync_retry:
                if time.monotonic() - getattr(self, 'last_resync_attempt', 0) >= .5:
                    self.last_resync_attempt = time.monotonic()
                    self.resync_position()
                return
            seen = self.tracker.observe(image)
            self.last_capture_ok = self.tracker.last_observed == self.tracker.expected(self.tracker.board)
            if not self.last_capture_ok:
                self.overlay.hide()
            if seen is not None:
                if seen.moves:
                    self.overlay.hide()
                    self.overlay.set_move(None)
                    self.pending_analysis = True
                    self.status.setText(f'Otomatik hamle: {", ".join(seen.moves)}. İlk üç öneri hazırlanıyor.')
                else:
                    self.overlay.hide()
                    self.status.setText(seen.message)
            if self.pending_analysis and self.last_capture_ok:
                self.request_analysis()
            elif self.last_capture_ok and self.overlay.moves and self.last_analyzed_fen == self.tracker.board.fen():
                if not self.only_bottom.isChecked() or self.current_player_is_bottom():
                    self.overlay.show()
        except (ValueError, CalibrationError) as exc:
            self.last_capture_ok = False
            self.overlay.hide()
            self.status.setText('Görüntü hatası: ' + str(exc))
        finally:
            self.busy_capture = False

    def request_analysis(self):
        if not self.tracker or not self.pending_analysis or not self.board_visible or not self.last_capture_ok:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        board = self.tracker.board
        if board.is_game_over():
            self.pending_analysis = False
            self.status.setText('Oyun sona erdi.')
            self.overlay.hide()
            return
        if self.only_bottom.isChecked() and not self.current_player_is_bottom():
            self.pending_analysis = False
            self.overlay.hide()
            return
        fen = board.fen()
        if fen == self.last_analyzed_fen:
            self.pending_analysis = False
            return
        self.worker = Analyzer(fen, self.engine.text().strip().strip('"'), self.seconds.value(), self.engine_session, multipv=3)
        self.worker.variations.connect(lambda results, expected=fen, generation=self.generation: self.analysis_done(expected, results) if generation == self.generation else None)
        self.worker.failure.connect(lambda error: self.status.setText('Motor hatası: ' + error))
        self.worker.finished.connect(self.worker_finished)
        self.pending_analysis = False
        self.worker.start()

    def analysis_done(self, analyzed_fen, results):
        if not self.tracker or self.tracker.board.fen() != analyzed_fen:
            return  # Stale engine result from an earlier position.
        self.last_analyzed_fen = analyzed_fen
        if self.only_bottom.isChecked() and not self.current_player_is_bottom():
            self.overlay.hide()
            return
        self.overlay.set_moves([r['uci'] for r in results])
        if self.board_visible and self.last_capture_ok:
            self.overlay.show()
        self.status.setText(' | '.join(f"{i+1}. {r['san']} ({r['score']})" for i, r in enumerate(results)))

    def worker_finished(self):
        if self.tracker and self.pending_analysis:
            QTimer.singleShot(0, self.request_analysis)

    def closeEvent(self, event):
        self.running = False
        self.locator_generation += 1
        self.timer.stop()
        self.overlay.close()
        if self.picker:
            self.picker.close()
        if self.worker and self.worker.isRunning():
            self.worker.wait(11000)
        if self.worker and self.worker.isRunning():
            event.ignore()
            self.status.setText("Motor kapanıyor; birkaç saniye sonra tekrar kapat.")
            return
        if self.detect_worker and self.detect_worker.isRunning():
            self.detect_worker.wait()
        self.engine_session.close()
        self.settings.setValue('engine', self.engine.text().strip())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = AutoWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
