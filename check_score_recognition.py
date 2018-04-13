# Check the score recognition success while randomly pressing WASD


# https://gabrielecirulli.github.io/2048/
# Concept:
#  use a scaled down image of the state, and input it as pixels 64x64 should do
#  gamescore will have to be aquired for the purpose of Deep Q Learning
#   (https://www.youtube.com/watch?v=79pmNdyxEGo)

import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY
import os, pickle
import random
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves

GRID_COLOR = [185, 172, 160]
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SCORE_MODEL_FILENAME = 'score_model.pickle'

# load score model
try:
    with open(SCORE_MODEL_FILENAME, 'rb') as f:
        score_model = pickle.load(f)
except FileNotFoundError:
    print('failed to load score model from', SCORE_MODEL_FILENAME)
    exit(1)

def preprocess_score_image(original):
    # takes single RGB image of the score and return numpy array of
    # flattened gray-sacle images of the individual digits, ordered from left to right

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    image = bw_image.copy()
    img_contours, contours, hierarchy = cv2.findContours(image,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(contour)
            cv2.rectangle(original, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)
            imgROI = image[intY:intY+intH, intX:intX+intW]
            resized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            flattened = resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            digits.append((intX, flattened,))

    flattened_digits =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT), dtype=np.int)
    for _, digit in sorted(digits, key=lambda z: z[0]):
        flattened_digits = np.append(flattened_digits, digit, 0)

    return flattened_digits

def col_filter(image, color, threshold=20):
    # returns an image where all areas of non-target color are black

    lower = np.array([ z-threshold for z in color ])
    upper = np.array([ z+threshold for z in color ])
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image, image, mask= mask)
    res = cv2.medianBlur(res, 5);
    res = cv2.medianBlur(mask, 5);

    return res

def split_rois(image, grid_color=GRID_COLOR):
    # there are two regions we want:
    # 1. the big grid
    # 2. the score box
    # both are displayed in boxes defined by the GRID_COLOR

    def find_range_start_end(sumlist):
        # in a list of values find the boundries of the first non-zero region of values

        region_detected = False
        region_start = 0
        for i in range(len(sumlist)):
            if sumlist[i] > 0 and not region_detected:
                region_detected = True
                region_start = i

            if sumlist[i] == 0 and region_detected:
                break

        return region_start, i

    # split rois horizontally
    workscreen = col_filter(image, grid_color, threshold=2)
    h_sums = workscreen.sum(axis=1)
    score_screen_start_y, split_at_y = find_range_start_end(h_sums)

    score_screen = image[score_screen_start_y:split_at_y]
    grid_screen = image[split_at_y:]

    # additionall cut horizontally to drop the best score
    workscreen = col_filter(score_screen, grid_color, threshold=2)
    v_sums = workscreen.sum(axis=0)
    score_screen_start_x, score_screen_end_x = find_range_start_end(v_sums)

    # also cut in half vertically to drop the "score" label
    score_screen = score_screen[int(score_screen.shape[0]/2):score_screen.shape[0],
                                 score_screen_start_x:score_screen_end_x]

    return score_screen, grid_screen, split_at_y

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


def main():
    last_time = time.time()
    score = 0
    score_buffer = deque([0, 0, 0])
    keylist = [W_KEY, A_KEY, S_KEY, D_KEYd]

    # print('in sublime Ctrl+Break(PrintScr) to cancel build')
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    while(True):
        PressKey(random.choice(keylist))

        screen =  np.array(ImageGrab.grab(bbox=(50, 50, 1200, 1400)))
        score_screen, grid_screen, y_offset = split_rois(screen)
        flat_score_digits = preprocess_score_image(score_screen)
        try:
            detected_score = int(''.join([ chr(score_model.predict([z])) for z in flat_score_digits ]))
        except ValueError as e:
            print('failed score detection:', e)

        _ = score_buffer.popleft()
        score_buffer.append(detected_score)

        score = update_score(score, score_buffer)

        # print('Loop took {} seconds'.format(time.time()-last_time))
        print('read:', detected_score, 'buffer:', score_buffer, 'score:', score)
        last_time = time.time()

        viewport = cv2.resize(screen, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('window', viewport)
        cv2.imshow('window2', score_screen)

        if cv2.waitKey(5) & 0xFF == ord('r'): # reset score
            score_buffer = deque([detected_score]*3)
            score = detected_score

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()