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
from directkeys import ReleaseKey, PressKey, W, A, S, D
import pyautogui

GRID_COLOR = [185, 172, 160]

def detect_rois(image, grid_color=GRID_COLOR):
    # there are two regions we want:
    # 1. the big grid
    # 2. the score box
    # both are displayed in boxes defined by the GRID_COLOR
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

def find_grid(workscreen):
    # detects the grid
    workscreen = col_filter(workscreen, GRID_COLOR, threshold=2)
    workscreen = corner_filter(workscreen)
    # mat = np.matrix(workscreen)
    v_sums = workscreen.sum(0)
    h_sums = workscreen.sum(1)

    for i, hs in enumerate(h_sums.flatten()):
        print(i, hs)

    print('-'*10)
    for i, vs in enumerate(v_sums):
        print(i, vs)

    ys = h_sums.argsort()[-8:]
    xs = v_sums.argsort()[-8:]
    
    xs = np.sort(xs)
    ys = np.sort(ys)

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
    pick = GRID_COLOR
    coords = [170,300]
    last_time = time.time()
    workscreen = []
    indicator = False
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(50, 50, 1200, 1400)))
        # new_screen = process_img(screen)
        if len(workscreen) == 0:
            workscreen = screen.copy()
        if len(grid) > 0:
            w3 = readout(grid)
            cv2.imshow('window3', w3)
            
            print('Loop took {} seconds'.format(time.time()-last_time), coords, pick)
        else:
            print('Loop took {} seconds'.format(time.time()-last_time), coords, pick)

        last_time = time.time()
        # cv2.imshow('window', new_screen)
        # cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window2', output(workscreen, pick, coords, indicator=indicator))
        if cv2.waitKey(25) & 0xFF == ord('r'): #reset to "fresh" screen
            workscreen = screen.copy()

        if cv2.waitKey(25) & 0xFF == ord('i'): #toggle color picker indicator
            indicator = not indicator
        
        if cv2.waitKey(25) & 0xFF == ord('f'):
            # workscreen = col_filter(screen, pick, threshold=5)
            workscreen = col_filter(workscreen, pick, threshold=2)

        if cv2.waitKey(25) & 0xFF == ord('g'):
            workscreen = corner_filter(workscreen)
            print(workscreen)

        if cv2.waitKey(25) & 0xFF == ord('o'):
            workscreen, grid = find_grid(screen.copy())

        if cv2.waitKey(25) & 0xFF == ord('p'):
            pick = col_pick(workscreen, coords)

        if cv2.waitKey(25) & 0xFF == ord('w'):
            coords[1] -= 10

        if cv2.waitKey(25) & 0xFF == ord('a'):
            coords[0] -= 10

        if cv2.waitKey(25) & 0xFF == ord('s'):
            coords[1] += 10

        if cv2.waitKey(25) & 0xFF == ord('d'):
            coords[0] +=10
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()