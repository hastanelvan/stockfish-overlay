import unittest
import numpy as np
import chess
from tracker import BoardTracker


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

if __name__=='__main__': unittest.main()
