import unittest
import numpy as np
import cv2
import chess
from board_locator import BoardLocator, BoardRegion
from test_tracker import render

class LocatorTests(unittest.TestCase):
    def screen(self, x=170,y=90,size=480):
        image=np.full((900,1400,3),40,dtype=np.uint8)
        board=render(chess.Board())
        if size!=480: board=cv2.resize(board,(size,size))
        image[y:y+size,x:x+size]=board
        return image

    def test_locate_and_follow_move_resize(self):
        previous=None
        for x,y,size in [(170,90,480),(600,220,560),(30,80,320)]:
            with self.subTest(size=size):
                region=BoardLocator().find(self.screen(x,y,size),previous)
                self.assertIsNotNone(region)
                self.assertLessEqual(abs(region.x-x),5)
                self.assertLessEqual(abs(region.y-y),5)
                self.assertLessEqual(abs(region.size-size),8)
                previous=region

    def test_reject_empty_and_obscured(self):
        self.assertIsNone(BoardLocator().find(np.full((600,900,3),50,dtype=np.uint8)))
        image=self.screen();image[150:400,200:550]=30
        self.assertLess(BoardLocator.confidence(image,BoardRegion(170,90,480)),.88)

    def test_highlights_tolerated(self):
        board=render(chess.Board(),highlighted=(chess.E2,chess.E4))
        self.assertGreaterEqual(BoardLocator.confidence(board),.88)

if __name__=='__main__': unittest.main()
