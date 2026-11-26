import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import chess
import chess.svg
from xml.etree import ElementTree as ET
from PySide6.QtCore import QByteArray
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from tracker import BoardTracker, CalibrationError


def render(board, flipped=False, highlighted=()):
    image = np.zeros((480,480,3), dtype=np.uint8)
    for i in range(64):
        row,col=divmod(i,8)
        square=chess.square(7-col,row) if flipped else chess.square(col,7-row)
        tile=image[row*60:(row+1)*60,col*60:(col+1)*60]
        tile[:]=[180,190,170] if (row+col)%2==0 else [85,110,90]
        if square in highlighted:
            tile[:]=[150,170,60]
        piece=board.piece_at(square)
        if piece:
            color=235 if piece.color else 12
            tile[16:47,22:38]=color
            # Each type has an asymmetric silhouette.
            tile[13:18+piece.piece_type*3,12:24]=color
            tile[37:46,30:40+piece.piece_type*2]=color
    return image


class TrackerTests(unittest.TestCase):
    def advance(self, tracker, board, uci, flipped=False):
        board.push_uci(uci)
        image=render(board,flipped)
        self.assertIsNone(tracker.observe(image))
        result=tracker.observe(image)
        self.assertIsNotNone(result)
        self.assertTrue(result.moves, result.message)
        self.assertEqual(tracker.board.fen(),board.fen())

    def test_capture_sequence_both_orientations(self):
        for flipped in (False,True):
            board=chess.Board(); tracker=BoardTracker(render(board,flipped),flipped)
            for move in ['e2e4','d7d5','e4d5','d8d5','b1c3','d5e5','f1e2']:
                self.advance(tracker,board,move,flipped)

    def test_ambiguous_capture_source(self):
        initial=chess.Board(); tracker=BoardTracker(render(initial))
        board=chess.Board('7k/8/2n1r3/3Q4/8/8/8/K7 w - - 0 1')
        tracker.board=board.copy();tracker.transition_cache.clear()
        self.advance(tracker,board,'d5c6')

    def test_midgame_resync(self):
        tracker=BoardTracker(render(chess.Board()))
        board=chess.Board()
        for move in ['e2e4','d7d5','e4d5','d8d5']:
            board.push_uci(move)
        result=tracker.resync(render(board), board.turn, 'KQkq')
        self.assertEqual(result.board_fen(),board.board_fen())
        self.advance(tracker,result,'b1c3')

    def test_underpromotion(self):
        tracker=BoardTracker(render(chess.Board()))
        board=chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
        tracker.board=board.copy();tracker.transition_cache.clear()
        self.advance(tracker,board,'a7a8n')

    def test_castling_and_en_passant(self):
        for fen,move in [('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1g1'),
                         ('7k/8/8/3pP3/8/8/8/7K w - d6 0 1','e5d6')]:
            tracker=BoardTracker(render(chess.Board()));board=chess.Board(fen)
            tracker.board=board.copy();tracker.transition_cache.clear()
            self.advance(tracker,board,move)

    def test_two_ply_catchup(self):
        board=chess.Board();tracker=BoardTracker(render(board))
        board.push_uci('e2e4');board.push_uci('e7e5')
        tracker.observe(render(board));result=tracker.observe(render(board))
        self.assertEqual(result.moves,['e2e4','e7e5'])

    def test_highlight_and_animation(self):
        board=chess.Board();tracker=BoardTracker(render(board))
        self.assertIsNone(tracker.observe(render(board,highlighted=(chess.E2,chess.E4))))
        transient=render(board);transient[360:420,240:300]=100
        tracker.observe(transient)
        self.assertIsNone(tracker.observe(render(board)))
        self.assertEqual(tracker.board.fen(),board.fen())


class RobustnessTests(unittest.TestCase):
    def test_color_mismatch_retried_without_occupancy_change(self):
        board=chess.Board();tracker=BoardTracker(render(board))
        board.push_uci('e2e4')
        wrong=board.copy();wrong.set_piece_at(chess.E4,chess.Piece(chess.PAWN,chess.BLACK))
        tracker.observe(render(wrong))
        result=tracker.observe(render(wrong))
        self.assertFalse(result.moves)
        self.assertEqual(tracker.board.turn,chess.WHITE)
        # Correct image has exactly the same occupied squares: must retry anyway.
        for _ in range(6):
            result=tracker.observe(render(board))
            if result and result.moves: break
        self.assertEqual(tracker.board.fen(),board.fen())

    def test_resync_both_piece_colors_after_movement(self):
        board=chess.Board();tracker=BoardTracker(render(board))
        for move in ['e2e4','e7e5','g1f3','b8c6','f1c4','f8c5']:
            board.push_uci(move)
        result=tracker.resync(render(board,highlighted=(chess.F8,chess.C5)),board.turn,'KQkq')
        self.assertEqual(result.board_fen(),board.board_fen())

    def test_small_dot_on_empty_square(self):
        board=chess.Board();tracker=BoardTracker(render(board))
        image=render(board)
        image[267:273,267:273]=25
        self.assertEqual(tracker.read(image),tracker.expected(board))


def render_svg(board, flipped=False, lastmove=None, lichess_marks=()):
    drawing = chess.svg.board(board, orientation=not flipped, coordinates=False,
                              size=480, lastmove=lastmove,
                              check=board.king(board.turn) if lichess_marks else None)
    if lichess_marks:
        svg = ET.fromstring(drawing)
        ns = '{http://www.w3.org/2000/svg}'
        # Chessground's check gradient reaches the farthest corner of the square.
        gradient = svg.find(f".//{ns}radialGradient[@id='check_gradient']")
        gradient.set('r', '70.71%')
        for stop, offset in zip(gradient, ('0%', '25%', '89%')):
            stop.set('offset', offset)
        for square in lichess_marks:
            square_rect = next(r for r in svg.findall(ns+'rect')
                               if chess.square_name(square) in r.get('class', '').split())
            x, y = float(square_rect.get('x')), float(square_rect.get('y'))
            size = float(square_rect.get('width'))
            if board.piece_at(square):
                # Occupied capture targets get a translucent tint in the corners.
                ring = ET.SubElement(svg.find(ns+'defs'), ns+'radialGradient',
                                     {'id': 'dest-ring', 'r': '70.71%'})
                for level, opacity in [('80%', '0'), ('80%', '.3')]:
                    ET.SubElement(ring, ns+'stop', {'offset': level, 'stop-color': '#145500',
                                                   'stop-opacity': opacity})
                marker = ET.Element(ns+'rect', dict(square_rect.attrib, fill='url(#dest-ring)'))
            else:
                marker = ET.Element(ns+'circle', {'cx': str(x+size/2), 'cy': str(y+size/2),
                    'r': str(size*.22/2**.5), 'fill': '#14551e', 'opacity': '.5'})
            svg.insert(list(svg).index(square_rect)+1, marker)
        drawing = ET.tostring(svg, encoding='unicode')
    image = QImage(480, 480, QImage.Format.Format_RGB888)
    with QPainter(image) as painter:
        QSvgRenderer(QByteArray(drawing.encode())).render(painter)
    return np.frombuffer(image.constBits(), np.uint8).reshape(480, 480, 3).copy()


class ColdStartTests(unittest.TestCase):
    def test_lichess_check_capture_ring_and_legal_dots(self):
        for fen, marks in [
            ('6rk/5p2/8/8/8/8/7R/K7 b - - 0 1', (chess.G7, chess.G6, chess.H2)),
            ('k7/8/8/8/8/8/1r4R1/1K6 w - - 0 1', (chess.C1, chess.C2, chess.B2)),
        ]:
            board = chess.Board(fen)
            self.assertTrue(board.is_check())
            for flipped in (False, True):
                image = render_svg(board, flipped, lichess_marks=marks)
                tracker = BoardTracker(image, flipped)
                self.assertEqual(tracker.board.board_fen(), board.board_fen())
                self.assertEqual(tracker.board.turn, board.turn)
                tracker.observe(image)
                self.assertTrue(tracker.last_position_matches)

    def test_unrecognized_foreground_is_still_rejected(self):
        board = chess.Board('6rk/5p2/8/8/8/8/7R/K7 b - - 0 1')
        image = render_svg(board, lichess_marks=(chess.G7, chess.H2))
        # An unrelated, opaque centered rectangle occupies an empty square.
        image[250:290, 250:290] = (200, 20, 30)
        with self.assertRaises(CalibrationError):
            BoardTracker(image)

    def test_midgame_start_and_next_move_with_both_orientations(self):
        for flipped in (False, True):
            board = chess.Board()
            for move in ('e2e4', 'e7e5', 'g1f3'):
                board.push_uci(move)
            tracker = BoardTracker(render_svg(board, flipped, board.peek()), flipped)
            self.assertEqual(tracker.board.board_fen(), board.board_fen())
            self.assertEqual(tracker.board.turn, chess.BLACK)
            self.assertFalse(tracker.turn_assumed)
            self.assertFalse(tracker.board.castling_rights)
            board.push_uci('b8c6')
            image = render_svg(board, flipped, board.peek())
            self.assertIsNone(tracker.observe(image))
            self.assertEqual(tracker.observe(image).moves, ['b8c6'])

    def test_turn_from_check_or_next_observed_move(self):
        checked = chess.Board('7k/8/8/8/8/8/7R/K7 b - - 0 1')
        self.assertEqual(BoardTracker(render_svg(checked)).board.turn, chess.BLACK)
        board = chess.Board('7k/8/8/8/7p/8/R7/K7 b - - 0 1')
        tracker = BoardTracker(render_svg(board))
        self.assertTrue(tracker.turn_assumed)
        self.assertEqual(tracker.board.turn, chess.WHITE)
        board.push_uci('h4h3')
        tracker.observe(render_svg(board))
        self.assertEqual(tracker.observe(render_svg(board)).moves, ['h4h3'])
        self.assertEqual(tracker.board.fen(), board.fen())
        self.assertFalse(tracker.turn_assumed)

    def test_en_passant_from_highlighted_double_push(self):
        board = chess.Board('7k/3p4/8/4P3/8/8/8/K7 b - - 0 1')
        board.push_uci('d7d5')
        tracker = BoardTracker(render_svg(board, lastmove=board.peek()))
        self.assertEqual(tracker.board.turn, board.turn)
        self.assertTrue(tracker.board.has_legal_en_passant())

    def test_custom_profile_survives_restart(self):
        board = chess.Board()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'theme.npz'
            BoardTracker(render(board), profile_path=path)
            board.push_uci('e2e4')
            tracker = BoardTracker(render(board, flipped=True), True, turn=chess.BLACK,
                                   castling='KQkq', profile_path=path)
            self.assertEqual(tracker.board.fen(), board.fen())

    def test_replacement_with_same_occupied_squares_is_detected(self):
        board = chess.Board('7k/5np1/8/8/8/8/3BR3/K7 w - - 0 1')
        tracker = BoardTracker(render_svg(board))
        board.set_piece_at(chess.D2, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.E2, chess.Piece(chess.BISHOP, chess.WHITE))
        image = render_svg(board)
        for _ in range(5):
            tracker.observe(image)
        self.assertFalse(tracker.last_position_matches)
        self.assertGreaterEqual(tracker.missed, 2)
        self.assertEqual(tracker.resync(image).board_fen(), board.board_fen())
        self.assertTrue(tracker.last_position_matches)

    def test_rejects_invalid_or_unreadable_positions(self):
        board = chess.Board()
        board.remove_piece_at(chess.E1)
        with self.assertRaises(CalibrationError):
            BoardTracker(render_svg(board))

if __name__=='__main__': unittest.main()
