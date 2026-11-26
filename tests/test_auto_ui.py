import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import cv2
import chess
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from auto_app import AutoWindow
from test_tracker import render, render_svg


class ScreenWindow(AutoWindow):
    def _capture(self, whole=False):
        if whole:
            return self.screen.copy()
        area = self.overlay.board_area
        return self.screen[area.y():area.y()+area.height(),
                           area.x():area.x()+area.width()].copy()

@unittest.skipUnless(os.environ.get('STOCKFISH_TEST_ENGINE'), 'STOCKFISH_TEST_ENGINE not set')
class AutoFlowTests(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.profile_path = Path(directory.name) / 'theme.npz'
        settings = QSettings(str(self.profile_path.with_suffix('.ini')), QSettings.Format.IniFormat)
        with patch('auto_app.QSettings', return_value=settings):
            self.window = ScreenWindow()
        self.window.profile_path = self.profile_path
        self.window.engine.setText(os.environ['STOCKFISH_TEST_ENGINE'])
        self.window.seconds.setValue(.15)
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.close()
        self.app.processEvents()

    def display(self, image, x=140, y=90, size=480):
        geometry = self.app.primaryScreen().geometry()
        screen = np.full((geometry.height(), geometry.width(), 3), 40, dtype=np.uint8)
        screen[y:y+size, x:x+size] = cv2.resize(image, (size, size))
        self.window.screen = screen

    def until(self, condition, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.app.processEvents()
            if condition():
                return
            time.sleep(.01)
        self.fail(self.window.status.text())

    def test_start_follow_resize_and_resume(self):
        window = self.window
        board = chess.Board()
        self.display(render(board))
        window.start_tracking()
        self.until(lambda: window.tracker is not None and len(window.overlay.moves) == 3)
        board.push_uci('e2e4')
        self.display(render(board))
        self.until(lambda: window.last_analyzed_fen == board.fen())
        window.screen[:] = 40
        self.until(lambda: not window.board_visible)
        self.display(render(board), 50, 180, 400)
        self.until(lambda: window.board_visible and window.overlay.board_area.width() == 400
                   and window.overlay.isVisible())
        self.assertEqual(window.tracker.board.fen(), board.fen())

    def test_midgame_strength_change_and_jump(self):
        window = self.window
        board = chess.Board()
        for move in ('e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4'):
            board.push_uci(move)
        window.black_bottom.setChecked(True)
        self.display(render_svg(board, flipped=True, lastmove=board.peek()))
        window.strength.setCurrentIndex(window.strength.findData(2500))
        window.start_tracking()
        self.until(lambda: window.worker is not None)
        first_search = window.worker
        self.assertEqual(first_search.elo, 2500)
        window.strength.setCurrentIndex(window.strength.findData(3500))
        window.human_like.setChecked(True)
        self.until(lambda: window.last_analyzed_fen and window.worker is not first_search)
        self.assertEqual(window.worker.elo, 3500)
        self.assertTrue(window.worker.human_like)
        self.assertIn(window.worker.effective_strength_text, window.status.text())
        self.assertEqual(window.tracker.board.turn, chess.BLACK)
        self.assertEqual(window.tracker.board.board_fen(), board.board_fen())
        self.assertTrue(all(chess.Move.from_uci(m) in board.legal_moves for m in window.overlay.moves))

        board.push_uci('g8f6')
        self.display(render_svg(board, flipped=True, lastmove=board.peek()))
        self.until(lambda: window.tracker.board.board_fen() == board.board_fen() and
                   window.last_analyzed_fen == window.tracker.board.fen())
        endgame = chess.Board('7k/7p/8/8/8/8/2R5/1K6 w - - 0 1')
        self.display(render_svg(endgame, flipped=True))
        self.until(lambda: window.last_analyzed_fen == endgame.fen())
        self.assertTrue(window.tracker.turn_assumed)
        window.resync_turn.setCurrentIndex(2)
        window.resync_position()
        endgame.turn = chess.BLACK
        self.until(lambda: window.last_analyzed_fen == endgame.fen())
        self.assertFalse(window.tracker.turn_assumed)
