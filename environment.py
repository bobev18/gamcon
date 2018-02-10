import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY, R_KEY
from collections import deque
import pickle


GRID_COLOR = [185, 172, 160]
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SCORE_MODEL_FILENAME = 'score_model.pickle'
STATE_SIZE = (64, 64)
SCREEN_CAPTURE_ZONE = (0, 0, 1200, 1080)


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

def preprocess_score_image(original, save_sample=False):
    # takes single RGB image of the score and return numpy array of
    # flattened gray-sacle images of the individual digits, ordered from left to right

    if save_sample:
        colored = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        cv2.imwrite('samples\sample_' + str(save_sample) + '.png', colored)

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

class Game:

    def __init__(self,):
        self.score = 0
        self.score_buffer = deque([0, 0, 0])
        self.keylist = [W_KEY, A_KEY, S_KEY, D_KEY]
        self.done = False
        self._count = 0
        # load score model
        try:
            with open(SCORE_MODEL_FILENAME, 'rb') as f:
                self.score_model = pickle.load(f)
        except FileNotFoundError:
            print('failed to load score model from', self.score_model_FILENAME)
            exit(1)

    def reset(self,):
        PressKey(R_KEY)
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

    def render(self,):
        self._count += 1
        screen =  np.array(ImageGrab.grab(bbox=SCREEN_CAPTURE_ZONE))
        score_screen, grid_screen, y_offset = split_rois(screen)
        flat_score_digits = preprocess_score_image(score_screen, self._count)

        # display stuff
        viewport = cv2.resize(screen, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('window', viewport)
        cv2.imshow('window2', score_screen)

        try:
            detected_score = int(''.join([ chr(self.score_model.predict([z])) for z in flat_score_digits ]))
        except ValueError as e:
            detected_score = -1
            print('failed score detection:', e)

        _ = self.score_buffer.popleft()
        self.score_buffer.append(detected_score)

        self.update_score()

        grid_shape = col_filter(grid_screen, GRID_COLOR)
        img_contours, contours, hierarchy = cv2.findContours(grid_shape,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

        rois = []

        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                [intX, intY, intW, intH] = cv2.boundingRect(contour)
                cv2.rectangle(grid_screen, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)
                imgROI = grid_screen[intY:intY+intH, intX:intX+intW]
                rois.append(imgROI)

        if len(rois) > 0:
            grid_roi = rois[0]
            gray_grid = cv2.cvtColor(grid_roi, cv2.COLOR_BGR2GRAY)
            grid = cv2.resize(gray_grid, STATE_SIZE)
            done = False
            cv2.imshow('grid', grid_roi)
        else:
            grid =  np.zeros(STATE_SIZE, dtype=np.int)
            done = True
            cv2.imshow('grid', grid)

        self.observation_space = grid
        self.done = done

        return grid, self.score, done

    def step(self, action):
        PressKey(self.keylist[action])



g = Game()
g.render()