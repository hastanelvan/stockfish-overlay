"""Set STOCKFISH_TEST_ENGINE to exercise an actual UCI engine."""
import os
import unittest
from unittest.mock import MagicMock

import chess
import chess.engine
from app import Analyzer, EngineSession, human_move, last_capture, strength_options


class StrengthTests(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.engine.options = {
            "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", False, None, None, None),
            "UCI_Elo": chess.engine.Option("UCI_Elo", "spin", 1320, 1320, 3190, None),
            "Skill Level": chess.engine.Option("Skill Level", "spin", 20, 0, 20, None),
        }

    def test_rating_and_full_strength_options(self):
        self.assertEqual(strength_options(self.engine, 2500), ({"UCI_LimitStrength": True, "UCI_Elo": 2500}, "2500 Elo"))
        full = {"UCI_LimitStrength": False, "Skill Level": 20}
        self.assertEqual(strength_options(self.engine, None), (full, "Tam güç"))
        self.assertEqual(strength_options(self.engine, 3500), (full, "Tam güç (motorun Elo üst sınırı 3190)"))
        self.engine.options["UCI_Elo"] = chess.engine.Option("UCI_Elo", "spin", 1320, 1320, 3600, None)
        self.assertEqual(strength_options(self.engine, 3500), ({"UCI_LimitStrength": True, "UCI_Elo": 3500}, "3500 Elo"))
        with self.assertRaisesRegex(ValueError, "1320–3600"):
            strength_options(self.engine, 1200)

    def test_limited_bestmove_precedes_pvs_with_its_own_score(self):
        position = chess.Board()
        moves = [chess.Move.from_uci(move) for move in ("e2e4", "d2d4", "c2c4", "g1f3")]
        search = self.engine.analysis.return_value.__enter__.return_value
        search.wait.return_value = chess.engine.BestMove(moves[3], None)
        search.multipv = [
            {"pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(50 - i * 10), chess.WHITE)}
            for i, move in enumerate(moves)
        ]
        session = MagicMock()
        session.get.return_value = self.engine
        results, errors = [], []
        worker = Analyzer(position.fen(), "engine", .1, session, multipv=3, elo=2500)
        worker.variations.connect(results.append)
        worker.failure.connect(errors.append)
        worker.run()
        self.assertFalse(errors, errors)
        self.assertEqual(worker.effective_strength_text, "2500 Elo")
        self.assertEqual([r["uci"] for r in results[0]], ["g1f3", "e2e4", "d2d4"])
        self.assertEqual(results[0][0]["score"], "+0.20")
        self.assertEqual(self.engine.analysis.call_args.kwargs["options"], {"UCI_LimitStrength": True, "UCI_Elo": 2500})

class HumanMoveTests(unittest.TestCase):
    def test_sound_development_keeps_black_perspective_and_elo_anchor(self):
        board = chess.Board()
        board.push_san("e4")
        quiet, develops = [board.parse_san(san) for san in ("a6", "Nf6")]
        # The scores below are from White's point of view, with Black to move.
        for anchor, alternative, expected in [
            (-25, -20, develops), (-5, 5, quiet), (-105, -98, quiet), (-15, 30, quiet),
            (90, 0, quiet),  # A stronger alternative shouldn't undo an Elo-limited choice.
        ]:
            with self.subTest(anchor=anchor, alternative=alternative):
                scores = {move: chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)
                          for move, cp in [(quiet, anchor), (develops, alternative)]}
                self.assertEqual(human_move(board, quiet, scores), expected)
        scores[quiet] = chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)
        self.assertEqual(human_move(board, quiet, scores), quiet)
        self.assertEqual(human_move(board, quiet, {}), quiet)

    def test_recapture_uses_available_move_history(self):
        board = chess.Board()
        board.push_san("e4")
        board.push_san("d5")
        self.assertIsNone(last_capture(board))
        board.push_san("exd5")
        capture = last_capture(board)
        self.assertEqual(capture, board.peek())
        develop, recapture = board.parse_san("Nf6"), board.parse_san("Qxd5")
        scores = {move: chess.engine.PovScore(chess.engine.Cp(20), chess.BLACK)
                  for move in (develop, recapture)}
        self.assertEqual(human_move(board, develop, scores, capture), recapture)
        self.assertEqual(human_move(board, develop, scores), develop)


@unittest.skipUnless(os.environ.get('STOCKFISH_TEST_ENGINE'), 'STOCKFISH_TEST_ENGINE not set')
class EngineTests(unittest.TestCase):
    def test_multipv_and_persistent_engine(self):
        session=EngineSession()
        try:
            initial=chess.Board()
            board=chess.Board('8/8/8/8/8/r1k5/8/K7 w - - 0 1')
            for position,count,elo,human in [(initial,3,2500,False),(board,1,3500,True)]:
                results=[];errors=[]
                worker=Analyzer(position.fen(),os.environ['STOCKFISH_TEST_ENGINE'],.25,session,
                                multipv=3,elo=elo,human_like=human)
                worker.variations.connect(results.append)
                worker.failure.connect(errors.append)
                worker.run()
                self.assertFalse(errors,errors)
                maximum = session.engine.options["UCI_Elo"].max
                expected = f"{elo} Elo" if elo <= maximum else f"Tam güç (motorun Elo üst sınırı {maximum})"
                self.assertEqual(worker.effective_strength_text, expected)
                self.assertEqual(len(results[0]),count)
                self.assertTrue(all(chess.Move.from_uci(r['uci']) in position.legal_moves for r in results[0]))
                if position is initial:
                    first_engine=session.engine
                else:
                    self.assertIs(session.engine,first_engine)
        finally:
            session.close()
