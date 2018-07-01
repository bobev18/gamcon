# Check the score recognition success while randomly pressing WASD

import os
if os.name == 'nt':
    from ctypes import windll
    user32 = windll.user32
    user32.SetProcessDPIAware()
    from PIL import ImageGrab
    from util.directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY, R_KEY
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
# from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY
import os, pickle
import random
from collections import deque            # For storing moves

from environment import Game

# GRID_COLOR = [185, 172, 160]
# MIN_CONTOUR_AREA = 50
# RESIZED_IMAGE_WIDTH = 20
# RESIZED_IMAGE_HEIGHT = 30


def update_score(old_score, score_buffer):
    # takes three consecutive reads of the same score to consider it
    if score_buffer[0] != score_buffer[1] or score_buffer[0] != score_buffer[2]:
        return old_score

    ### if increased buffer size, for efficient comparison use ###
    # if score_buffer.count(score_buffer[0]) != len(score_buffer):
    #     return old_score

    consider = score_buffer[0]

    # new score has to be higher:
    if consider <= old_score:
        return old_score

    # difference in score should be divisible by 4
    if (consider - old_score) % 4 != 0:
        return old_score

    return consider


def render(screen, score_buffer):
    grid_img, detected_score, game_over = screen.read()

    if cv2.waitKey(2) & 0xFF == ord('q'):
        screen.destroy()
        exit(1)

    _ = score_buffer.popleft()
    score_buffer.append(detected_score)
    score = update_score()

    return grid_img, detected_score, score, game_over


def main():
    last_time = time.time()
    score = 0

    print('in sublime Ctrl+Break(PrintScr) to cancel build')
    print('get game window in focus')
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    env = Game()
    grid, score, gameover = env.render()
    keylist = env.keylist

    while(True):
        PressKey(random.choice(keylist))

        grid, score, gameover = env.render()
        # the raw detected score will be in the last cell of the buffer
        detected_score = env.score_buffer[-1]

        # print('Loop took {} seconds'.format(time.time()-last_time))
        print('read:', detected_score, 'buffer:', env.score_buffer, 'score:', score)
        last_time = time.time()

        key_pressed = False
        while not key_pressed:

            if cv2.waitKey(5) & 0xFF == ord('r'): # reset score
                score = 0
                detected_score = 0
                env.reset()
                key_pressed = True

            if cv2.waitKey(5) & 0xFF == ord('q'): # exit
                key_pressed = True
                env.destroy()
                exit(1)

            if cv2.waitKey(5) & 0xFF == ord('n'): # next action
                key_pressed = True


        # the above requires focus on the "app" i.e display windos in order to work
        # if the focus is on the game, as input requres, this doesn't work

        print('switch back to game to execute action')
        for i in list(range(2))[::-1]:
            print(i+1)
            time.sleep(1)

main()
