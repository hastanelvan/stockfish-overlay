"""Set STOCKFISH_TEST_ENGINE to exercise an actual UCI engine."""
import os
import unittest
import chess
from app import Analyzer, EngineSession

@unittest.skipUnless(os.environ.get('STOCKFISH_TEST_ENGINE'), 'STOCKFISH_TEST_ENGINE not set')
class EngineTests(unittest.TestCase):
    def test_multipv_and_persistent_engine(self):
        session=EngineSession()
        try:
            initial=chess.Board()
            board=chess.Board('8/8/8/8/8/r1k5/8/K7 w - - 0 1')
            for position,count in [(initial,3),(board,1)]:
                results=[];errors=[]
                worker=Analyzer(position.fen(),os.environ['STOCKFISH_TEST_ENGINE'],.10,session,multipv=3)
                worker.variations.connect(results.append)
                worker.failure.connect(errors.append)
                worker.run()
                self.assertFalse(errors,errors)
                self.assertEqual(len(results[0]),count)
                self.assertTrue(all(chess.Move.from_uci(r['uci']) in position.legal_moves for r in results[0]))
                if position is initial:
                    first_engine=session.engine
                else:
                    self.assertIs(session.engine,first_engine)
        finally:
            session.close()
