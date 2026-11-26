import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
from PySide6.QtGui import QImage

from piece_templates import import_reference_image
from tracker import BoardTracker, CalibrationError
from test_tracker import render


class ThemeImportTests(unittest.TestCase):
    def test_reference_in_full_screenshot_can_be_reused_with_other_orientation(self):
        board = chess.Board()
        screenshot = np.full((611, 997, 3), 35, dtype=np.uint8)
        screenshot[51:531, 137:617] = render(board, flipped=True)
        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder) / 'örnek.png'
            image = QImage(screenshot.data, 997, 611, 997 * 3, QImage.Format.Format_RGB888)
            self.assertTrue(image.save(str(photo)))
            profile = Path(folder) / 'theme.npz'
            import_reference_image(photo, black_bottom=True, profile_path=profile)
            self.assertTrue(profile.is_file())

            for san in ('e4', 'e5', 'Nf3', 'Nc6', 'Bb5'):
                board.push_san(san)
            tracker = BoardTracker(render(board), turn=board.turn, profile_path=profile)
            self.assertEqual(tracker.board.board_fen(), board.board_fen())

            with self.assertRaises(CalibrationError):
                import_reference_image(photo, black_bottom=False, profile_path=profile)
            self.assertIsNotNone(BoardTracker.load_profile(profile))
