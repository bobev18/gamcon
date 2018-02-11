# https://gabrielecirulli.github.io/2048/
# Initial concept:
#  Recognize the value at each cell to build representation of the state
# Issues:
#  1. due to the difference in contrast it's difficult to get uniformity between samples of different digits
#  2. there is no variation in the digits, so MNIST type of approach is overkill
#  3. considering the sample generation, it's obvious I can detect the number by the color - ML not needed.
#
# New concept:
#  use a scaled down image of the state, and input it as pixels 64x64 should do
#  gamescore will have to be aquired for the purpose of Deep Q Learning
#   (https://www.youtube.com/watch?v=79pmNdyxEGo)
#    - capture samples of the digits and train recognizer
#    - parsing from HTML will not work as it's updated by JS not via HTTP


import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY, R_KEY
# import pyautogui
import os, pickle

GRID_COLOR = [185, 172, 160]
SAMPLE_FOLDER = 'samples'
SCORE_FILENAME = 'score_digits.png'
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

def split_rois(image, grid_color=GRID_COLOR):
    # there are two regions we want:
    # 1. the big grid
    # 2. the score box
    # both are displayed in boxes defined by the GRID_COLOR

    def find_range_start_end(sumlist):
        region_detected = False
        region_start = 0
        for i in range(len(sumlist)):
            if sumlist[i] > 0 and not region_detected:
                region_detected = True
                region_start = i

            if sumlist[i] == 0 and region_detected:
                break

        return region_start, i


    # cut horizontally
    workscreen = col_filter(image, grid_color, threshold=2)
    h_sums = workscreen.sum(1)

    score_screen_start_y, split_at_y = find_range_start_end(h_sums)

    # start_x_score = int(image.shape[0]/3)
    # end_x_score = image.shape[0]
    # start_y_score = 0
    # end_y_score = split_at_y

    # print(start_x_score,end_x_score,start_y_score,end_y_score)
    # print(image.shape)

    # increased best score tends to push the score to the left
    # score_screen = image[score_screen_start_y:split_at_y, int(image.shape[1]/3):image.shape[1]]
    score_screen = image[score_screen_start_y:split_at_y]

    # additionally cut vertically the score screen from the best score
    # allso cut in half vertically to drop the "score" label
    workscreen = col_filter(score_screen, grid_color, threshold=2)
    v_sums = workscreen.sum(0)
    score_screen_start_x, score_screen_end_x = find_range_start_end(v_sums)
    score_screen = score_screen[int(score_screen.shape[0]/2):score_screen.shape[0],
                                 score_screen_start_x:score_screen_end_x]


    grid_screen = image[split_at_y:]

    return score_screen, grid_screen, split_at_y

def save_score(score_screen):
    cv2.imwrite(os.path.join(SAMPLE_FOLDER, SCORE_FILENAME), cv2.cvtColor(score_screen, cv2.COLOR_BGR2RGB))

def digit_splitter(image):
    pass

def col_filter(image, color, threshold=20):
    # filters specific color from the image
    lower = np.array([ z-threshold for z in color ])
    upper = np.array([ z+threshold for z in color ])
    print('filtering by color range:', lower, upper)
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image, image, mask= mask)
    # cv2.GaussianBlur(res, (5,5), 0);
    res = cv2.medianBlur(res, 5);
    res = cv2.medianBlur(mask, 5);

    return res

def corner_filter(image):
    # filter that finds edges in the direction of forwards slash (/)
    # these are found in the top right and bottom left corners of the grid boxes
    kernel = np.array([[-1, -1, -1, -1, 4],
                       [-1, -1, -1, 4, -1],
                       [-1, -1, 4, -1, -1],
                       [-1, 4, -1, -1, -1],
                       [4, -1, -1, -1, -1]])

    dst = cv2.filter2D(image, -1, kernel)
    return dst

def col_pick(image, coords=[170,300]):
    # reads the corlor at a X,Y coords of an image
    pick = image[coords[0],coords[1]].tolist()
    return pick

def output(image, some_color, coords, convert=False, indicator=False):
    # displays modifications applied on the whole image;
    # in addtion overlays two blocks - one with provided color and one with the 'inverted' color
    if convert:
        work_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    else:
        work_image = image

    if indicator:
        cv2.circle(work_image, tuple([coords[0]-50, coords[1]-50]), 10, [255,0,0])

        vertices = np.array([[100,100], [150,100], [150,150], [100,150]], np.int32)
        cv2.fillPoly(work_image, [vertices], some_color)
        vertices = np.array([[200,100], [250,100], [250,150], [200,150]], np.int32)
        inverted_color = some_color
        if isinstance(some_color, list):
            inverted_color = [ z for z in some_color[::-1] ]
        else:
            inverted = 255 - some_color
        cv2.fillPoly(work_image, [vertices], inverted_color)

    return work_image

def find_grid(workscreen, y_offset):
    # detects the grid
    workscreen = col_filter(workscreen, GRID_COLOR, threshold=2)
    workscreen = corner_filter(workscreen)

    # mat = np.matrix(workscreen)
    v_sums = workscreen.sum(0)
    h_sums = workscreen.sum(1)

    ys = h_sums.argsort()[-8:]
    xs = v_sums.argsort()[-8:]

    xs = np.sort(xs)
    ys = np.sort(ys)

    # adjust the grid boxes with the offset of the ROI region
    ys = [ z+y_offset for z in ys ]

    # print('xs', xs, '         ys', ys)

    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    boxes = []
    for row in pairwise(ys):
        for col in pairwise(xs):
            boxes.append(list(zip(col, row)))

    print('boxes', boxes)
    for box in boxes:
        print('box', box)
        cv2.rectangle(workscreen,*box,[255,0,0])

    return workscreen, boxes

def readout(boxes):
    # captures individual boxes

    for b in range(1):
        onebox = list(boxes[b][0] + boxes[b][1])
        onebox = [ z+50 for z in onebox ]

        image = np.array(ImageGrab.grab(bbox=tuple(onebox)))

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = (255 - img_gray)

        # black and white
        ret, bw_image = cv2.threshold(img_gray, 127,255,cv2.THRESH_BINARY)

        image = cv2.resize(bw_image, (8, 8))

    return bw_image



def main():
    grid = []
    score_screen = []
    pick = GRID_COLOR
    coords = [170,300]
    last_time = time.time()
    workscreen = []
    indicator = False
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(50, 50, 1200, 1400)))

        score_screen, grid_screen, y_offset = split_rois(screen)
        flat_score_digits = preprocess_score_image(score_screen)
        score = int(''.join([ chr(score_model.predict([z])) for z in flat_score_digits ]))


        # new_screen = process_img(screen)
        if len(workscreen) == 0:
            workscreen = screen.copy()

        print('Loop took {} seconds'.format(time.time()-last_time), score)

        last_time = time.time()
        cv2.imshow('window', output(workscreen, pick, coords, indicator=indicator))
        if len(score_screen) > 0:
            cv2.imshow('window2', score_screen)

        if cv2.waitKey(5) & 0xFF == ord('r'): #reset to "fresh" screen
            workscreen = screen.copy()

        if cv2.waitKey(5) & 0xFF == ord('i'): #toggle color picker indicator
            indicator = not indicator

        if cv2.waitKey(5) & 0xFF == ord('f'):
            # workscreen = col_filter(screen, pick, threshold=5)
            workscreen = col_filter(workscreen, pick, threshold=2)

        if cv2.waitKey(5) & 0xFF == ord('g'):
            workscreen = corner_filter(workscreen)
            print(workscreen)

        if cv2.waitKey(5) & 0xFF == ord('t'):    #trigger processing
            score_screen, grid_screen, y_offset = split_rois(screen)
            workscreen, grid = find_grid(grid_screen.copy(), y_offset)

        if cv2.waitKey(5) & 0xFF == ord('p'):
            pick = col_pick(workscreen, coords)

        if cv2.waitKey(5) & 0xFF == ord('w'):
            coords[1] -= 10

        if cv2.waitKey(5) & 0xFF == ord('a'):
            coords[0] -= 10

        if cv2.waitKey(5) & 0xFF == ord('s'):
            coords[1] += 10

        if cv2.waitKey(5) & 0xFF == ord('d'):
            coords[0] +=10

        if cv2.waitKey(5) & 0xFF == ord('y'):
            # break score into separate digits
            flat_score_digits = preprocess_score_image(score_screen)
            print([ chr(score_model.predict([z])) for z in flat_score_digits ])

        if cv2.waitKey(5) & 0xFF == ord('z'):
            save_score(score_screen)
            print('SAVED')

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()