import os
if os.name == 'nt':
    from PIL import ImageGrab
    from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY, R_KEY
    KEYLIST = [W_KEY, A_KEY, S_KEY, D_KEY]
    RESET_KEY = R_KEY
    def tap(key):
        PressKey(key)
        ReleaseKey(key)

else:
    import pyscreenshot as ImageGrab
    from pynput.keyboard import Key, Controller
    KEYLIST = ['w', 'a', 's', 'd']
    RESET_KEY = 'r'
    keyboard = Controller()
    def tap(key):
        keyboard.press(key)
        keyboard.release(key)

import numpy as np
import cv2
import time
from collections import deque
import pickle

from screen import Screen


class Game:

    def __init__(self,):
        self.score = 0
        self.score_buffer = deque([0, 0, 0])
        self.keylist = KEYLIST
        self.done = False
        self._count = 0
        self.screen = Screen()

    def reset(self,):
        tap(RESET_KEY)
        self.score = 0
        self.score_buffer = deque([0, 0, 0])


    def update_score(self):
        # takes three consecutive reads of the same score to consider it
        if self.score_buffer[0] != self.score_buffer[1] or self.score_buffer[0] != self.score_buffer[2]:
            return

        ### if increased buffer size, for efficient comparison use ###
        # if self.score_buffer.count(self.score_buffer[0]) != len(self.score_buffer):
        #     return self.score

        consider = self.score_buffer[0]

        # new score has to be higher:
        if consider <= self.score:
            return

        # difference in score should be divisible by 4
        if (consider - self.score) % 4 != 0:
            return

        self.score = consider

    def render(self, save_sample=False):
        self._count += 1

        if save_sample:
            grid_img, detected_score, game_over = self.screen.read(save_sample=self._count)
        else:
            grid_img, detected_score, game_over = self.screen.read()

        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(1)

        _ = self.score_buffer.popleft()
        self.score_buffer.append(detected_score)
        self.update_score()

        self.done = game_over

        return grid_img, self.score, self.done

    def step(self, action):
        tap(self.keylist[action])

    def destroy(self):
        cv2.destroyAllWindows()



g = Game()
g.render()