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
from collections import deque
import pickle


GRID_COLOR = [185, 172, 160]
SCORE_COLOR = [235,235,235]
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SCORE_MODEL_FILENAME = 'data/score_model2.pickle'
STATE_SIZE = (64, 64)
SCREEN_CAPTURE_ZONE = (0, 0, 1000, 1440)
MAX_SMALL_ZONE_SIZE = 7
MAX_PARTIAL_ZONE_SIZE = 11

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

    def __col_filter(self, image, color, threshold=20):
        '''returns an image where all areas of non-target color are black'''

        lower = np.array([ z-threshold for z in color ])
        upper = np.array([ z+threshold for z in color ])
        mask = cv2.inRange(image, lower, upper)
        res = cv2.bitwise_and(image, image, mask= mask)
        res = cv2.medianBlur(res, 5);
        res = cv2.medianBlur(mask, 5);

        return res

    def __find_range_start_end(self, sumlist):
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



    def _split_rois(self, image, grid_color=GRID_COLOR):
        # there are two regions we want:
        # 1. the big grid
        # 2. the score box
        # both are displayed in boxes defined by the GRID_COLOR


        # split rois horizontally
        workscreen = self.__col_filter(image, grid_color, threshold=2)
        h_sums = workscreen.sum(axis=1)
        score_screen_start_y, split_at_y = self.__find_range_start_end(h_sums)
        score_screen = image[score_screen_start_y:split_at_y]
        grid_screen = image[split_at_y:]

        # additionall cut horizontally to drop the best score
        workscreen = self.__col_filter(score_screen, grid_color, threshold=2)
        v_sums = workscreen.sum(axis=0)
        score_screen_start_x, score_screen_end_x = self.__find_range_start_end(v_sums)

        # also cut in half vertically to drop the "score" label
        score_screen = score_screen[int(score_screen.shape[0]/2):score_screen.shape[0],
                                     score_screen_start_x:score_screen_end_x]

        return score_screen, grid_screen, split_at_y

    # def _split_score_digits(self, image, color=GRID_COLOR, corner_cut_size=1, verbose=0):
    #     workscreen = self.__col_filter(image, color=color, threshold=20)
    #     # sum over axis "0" i.e. count the pixels of filtered color in each column
    #     v_sums = workscreen.sum(0)

    #     # cut corners (because of the round edges of the UI element, there are
    #     #                           light color pixels in the corners of the img)
    #     v_sums = v_sums[corner_cut_size:-corner_cut_size]

    #     zones = []
    #     consider_x = 0
    #     for delme in range(100):
    #         start_x, end_x = self.__find_range_start_end(v_sums[consider_x:])

    #         # because self.__find_range_start_end gives results relative to the input
    #         start_x +=  consider_x
    #         end_x += consider_x

    #         subimage = image[0:workscreen.shape[0], start_x:end_x + 1]
    #         consider_x = end_x + 1

    #         zones.append([subimage, start_x, end_x])
    #         if start_x == consider_x or sum(v_sums[consider_x:]) == 0:
    #             break

    #     return zones

    def _split_score_digits(self, image, color=GRID_COLOR, corner_cut_size=1, verbose=0):
        # filter by color
        workscreen = self.__col_filter(image, color=color, threshold=20)

        # sum over axis "0" i.e. sum the filterd color pixels in each column
        v_sums = workscreen.sum(0)
        if verbose > 0:
            print('h_sums lenght prior to corner cut:', len(v_sums))

        # cut corners (because of the round edges of the UI element, there are light color pixels in the corners of the img)
        v_sums = v_sums[corner_cut_size:-corner_cut_size]

        if verbose > 0:
            print('v_sums', v_sums)

        zones = []
        consider_x = 0
        for delme in range(100):
            start_x, end_x = self.__find_range_start_end(v_sums[consider_x:])

            # because find_range_start_end gives results relative to the input
            start_x +=  consider_x
            end_x += consider_x

            subimage = image[0:workscreen.shape[0], start_x:end_x + 1]
            consider_x = end_x + 1

            if verbose > 0:
                print(delme, start_x, end_x)

            zones.append([subimage, start_x, end_x])
            if start_x == consider_x or sum(v_sums[consider_x:]) == 0:
                break

        return zones

    def _merge_small_zones(self, zones, image, verbose=0):
        '''determine if single digit was split in two zones - "small" & "partial"'''

        part_sizes = ([ z[2]-z[1] for z in zones ])
        if verbose > 0:
            print('zones sizes', part_sizes)
        new_zones = []

        if verbose > 0:
            print('# zones', len(zones))

        if len(zones) > 1:
            i = 1
            while i < len(zones):   # + 1:
                if verbose > 0:
                    print('processing zones', i-1, '&', i)
                    print('part_sizes[i-1]', part_sizes[i-1])
                    print('part_sizes[i]', part_sizes[i])
                if part_sizes[i-1] < MAX_SMALL_ZONE_SIZE and part_sizes[i] <= MAX_PARTIAL_ZONE_SIZE:
                    subimage = image[0:image.shape[0], zones[i-1][1]:zones[i][2]+1]
                    new_zones.append([subimage, zones[i-1][1], zones[i][2]])
                    i += 1
                    if verbose > 0:
                        print('merged, thus advancing cycle to:', i-1, '&', i,
                                'further +1 increment will execute at normal cycle end')

                    # check if single final zone and add it to the final result,
                    # because it cannot be processed as pair with subsequent one
                    if i == len(zones)-1:
                        new_zones.append(zones[i])

                else:
                    new_zones.append(zones[i-1])

                i += 1
                if verbose > 0:
                    print('ending=?  (i-1,i)', (i-1, i), '; goalpole: i<', len(zones), i == len(zones) +1 )

        else:
            new_zones = zones

        if verbose > 0:
            print([z[1:] for z in new_zones])

        return new_zones

    def _score_digit_rois(self, image, corner_cut_size=1, verbose=0):
        zones = self._split_score_digits(image, color=SCORE_COLOR, corner_cut_size=corner_cut_size, verbose=verbose)
        for i, zone in enumerate(zones):
            self.display('split_zone' + str(i), image, bound=[zone[1], 0, zone[2], image.shape[1]])
        zones = self._merge_small_zones(zones, image, verbose=verbose)

        return zones

    def _flatten_score(self, score_image, save_sample=False):
        '''
        takes single RGB image of the score and return numpy array of
        flattened gray-sacle images of the individual digits, ordered from left to right
        '''

        digit_zones = self._score_digit_rois(score_image, corner_cut_size=2, verbose=0)
        digits = []
        for digit in digit_zones:
            gray = cv2.cvtColor(digit[0], cv2.COLOR_BGR2GRAY)
            _, bw_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            resized = cv2.resize(bw_image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            digits.append(resized)

        flattened_digits = [z.reshape((RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)) for z in digits]
        return flattened_digits

    def _unflatten(self, image):
        return np.reshape(image, (RESIZED_IMAGE_HEIGHT, -1))

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

        if save_sample:
            colored = cv2.cvtColor(score_screen, cv2.COLOR_BGR2RGB)
            cv2.imwrite('sometest\sample_' + str(save_sample) + '.png', colored)

        flat_score_digits = self._flatten_score(score_screen, save_sample)

        game_over, grid = self.check_if_done(grid_screen)

        # display stuff
        if display_rois:
            viewport = cv2.resize(self.raw_screen, (0,0), fx=0.5, fy=0.5)
            self.display('grab_zone', viewport)
            self.display('score', score_screen)
            self.display('grid', grid)

            digit_zones = self._score_digit_rois(score_screen, corner_cut_size=2, verbose=0)
            # print(l)
            for i, dz in enumerate(digit_zones):
                self.display('digitzone' + str(i), dz[0])

        try:
            # detected_score = int(''.join([ chr(self.score_model.predict([z])) for z in flat_score_digits ]))
            print('# digits in score:', len(flat_score_digits))
            predictions = [ self.score_model.predict([z]) for z in flat_score_digits ]
            for i, predicts in enumerate(predictions):
                unflat = self._unflatten(flat_score_digits[i])
                self.display('unflat' + str(i), unflat)
                print('predictions for digit', i, ':', predicts)
            detected_score = int(''.join([ str(z[0]) for z in predictions ]))
            print('detected_score', detected_score)
        except ValueError as e:
            detected_score = -1
            print('failed score detection:', e)

        return grid, detected_score, game_over

    def check_if_done(self, grid_screen):
        grid_shape = self.__col_filter(grid_screen, GRID_COLOR)
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

    def display(self, window_name, image, bound=None):
        local_image = image.copy()
        if bound:
            cv2.rectangle(local_image, (bound[0], bound[1]), (bound[2], bound[3]), (0, 0, 255), 2)
        cv2.imshow(window_name, local_image)
