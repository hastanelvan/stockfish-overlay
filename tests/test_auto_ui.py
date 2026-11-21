import os
import time
import unittest
import numpy as np
import cv2
import chess
from PySide6.QtWidgets import QApplication
from auto_app import AutoWindow
from test_tracker import render

@unittest.skipUnless(os.environ.get('STOCKFISH_TEST_ENGINE'), 'STOCKFISH_TEST_ENGINE not set')
class AutoFlowTests(unittest.TestCase):
    def test_start_follow_resize_and_resume(self):
        app=QApplication.instance() or QApplication([])
        class ScreenWindow(AutoWindow):
            def _capture(self,whole=False):
                if whole: return self.screen.copy()
                area=self.overlay.board_area
                return self.screen[area.y():area.y()+area.height(),area.x():area.x()+area.width()].copy()
        window=ScreenWindow()
        window.engine.setText(os.environ['STOCKFISH_TEST_ENGINE'])
        window.only_bottom.setChecked(False)
        geometry=app.primaryScreen().geometry()
        board=chess.Board()
        def display(x,y,size):
            screen=np.full((geometry.height(),geometry.width(),3),40,dtype=np.uint8)
            screen[y:y+size,x:x+size]=cv2.resize(render(board),(size,size))
            window.screen=screen
        def until(condition,timeout=12):
            start=time.monotonic()
            while time.monotonic()-start<timeout:
                app.processEvents()
                if condition(): return
                time.sleep(.01)
            self.fail(window.status.text())
        try:
            display(140,90,480)
            window.start_tracking()
            until(lambda: window.tracker is not None and len(window.overlay.moves)==3)
            board.push_uci('e2e4')
            display(140,90,480)
            until(lambda:window.tracker.board.fen()==board.fen() and window.last_analyzed_fen==board.fen())
            window.screen[:]=40
            until(lambda: not window.board_visible)
            display(50,180,400)
            until(lambda: window.board_visible and window.overlay.board_area.width()==400 and window.overlay.isVisible())
            self.assertEqual(window.tracker.board.fen(),board.fen())
        finally:
            window.close()
            app.processEvents()
