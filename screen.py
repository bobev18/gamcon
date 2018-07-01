import os
if os.name == 'nt':
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
from collections import deque
import pickle


GRID_COLOR = [185, 172, 160]
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SCORE_MODEL_FILENAME = 'data/score_model.pickle'
STATE_SIZE = (64, 64)
SCREEN_CAPTURE_ZONE = (0, 0, 1200, 1080)

class Screen:

    def __init__(self, capture_zone=SCREEN_CAPTURE_ZONE):
        self.capture_zone = capture_zone
        self.zone_width = capture_zone[2] - capture_zone[0]
        self.zone_height = capture_zone[3] - capture_zone[1]
        self.raw_screen = np.zeros((self.zone_width, self.zone_height), dtype=np.int)
        # load score ML model
        try:
            with open(SCORE_MODEL_FILENAME, 'rb') as f:
                self.score_model = pickle.load(f)
        except FileNotFoundError:
            print('failed to load score model from', self.score_model_FILENAME)
            exit(1)

    def _col_filter(self, image, color, threshold=20):
        '''returns an image where all areas of non-target color are black'''

        lower = np.array([ z-threshold for z in color ])
        upper = np.array([ z+threshold for z in color ])
        mask = cv2.inRange(image, lower, upper)
        res = cv2.bitwise_and(image, image, mask= mask)
        res = cv2.medianBlur(res, 5);
        res = cv2.medianBlur(mask, 5);

        return res

    def _split_rois(self, image, grid_color=GRID_COLOR):
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
        workscreen = self._col_filter(image, grid_color, threshold=2)
        h_sums = workscreen.sum(axis=1)
        score_screen_start_y, split_at_y = find_range_start_end(h_sums)

        score_screen = image[score_screen_start_y:split_at_y]
        grid_screen = image[split_at_y:]

        # additionall cut horizontally to drop the best score
        workscreen = self._col_filter(score_screen, grid_color, threshold=2)
        v_sums = workscreen.sum(axis=0)
        score_screen_start_x, score_screen_end_x = find_range_start_end(v_sums)

        # also cut in half vertically to drop the "score" label
        score_screen = score_screen[int(score_screen.shape[0]/2):score_screen.shape[0],
                                     score_screen_start_x:score_screen_end_x]

        return score_screen, grid_screen, split_at_y

    def _preprocess_score_image(self, original, save_sample=False):
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

    def _grab(self):
        self.raw_screen = np.array(ImageGrab.grab(bbox=self.capture_zone))

    def read(self, save_sample=False, display_rois=True):
        '''
            captures and processes screen - returns:
              - grid image to be fed in ML
              - detected score
              - game over indicator
        '''

        self._grab()
        score_screen, grid_screen, y_offset = self._split_rois(self.raw_screen)

        flat_score_digits = self._preprocess_score_image(score_screen, save_sample)

        game_over, grid = self.check_if_done(grid_screen)

        # display stuff
        if display_rois:
            viewport = cv2.resize(self.raw_screen, (0,0), fx=0.5, fy=0.5)
            self.display('grab_zone', viewport)
            self.display('score', score_screen)
            self.display('grid', grid)

        try:
            detected_score = int(''.join([ chr(self.score_model.predict([z])) for z in flat_score_digits ]))
        except ValueError as e:
            detected_score = -1
            print('failed score detection:', e)

        return grid, detected_score, game_over

    def check_if_done(self, grid_screen):
        grid_shape = self._col_filter(grid_screen, GRID_COLOR)
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

        return done, grid

    def display(self, window, image):
        cv2.imshow(window, image)
