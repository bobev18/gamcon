import os
if os.name == 'nt':
    from PIL import ImageGrab
    from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY, R_KEY
    KEYLIST = [W_KEY, A_KEY, S_KEY, D_KEY]
    RESET_KEY = R_KEY
    def tap(key):
        PressKey(key)
        ReleaseKey(key)
    from ctypes import windll
    user32 = windll.user32
    user32.SetProcessDPIAware()
    BOX = (3280, 15, 3830, 550)

else:
    import pyscreenshot as ImageGrab
    from pynput.keyboard import Key, Controller
    KEYLIST = ['w', 'a', 's', 'd']
    RESET_KEY = 'r'
    keyboard = Controller()
    def tap(key):
        keyboard.press(key)
        keyboard.release(key)
    BOX = (525, 150, 650, 280)

    # Using box bigger than the screen 
    # BOX = (3280, 15, 3830, 550)
    # results in error:
    #
    #     Traceback (most recent call last):
    #   File "mapper.py", line 380, in <module>
    #     main()
    #   File "mapper.py", line 321, in main
    #     cv2.imshow('window', output(workscreen, pick, coords, indicator=indicator))
    #   File "mapper.py", line 210, in output
    #     work_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.error: OpenCV(3.4.2) /io/opencv/modules/imgproc/src/color.hpp:253: error: (-215:Assertion failed) VScn::contains(scn) && VDcn::contains(dcn) && VDepth::contains(depth) in function 'CvtHelper'
    #


import numpy as np
import cv2
import time
import os, pickle


# WALL_COLOR = [185, 172, 160]
WALL_COLOR = [100, 110, 120]
FOG_COLOR = [3, 54, 73]
SAMPLE_FOLDER = 'samples'
SCORE_FILENAME = 'score_digits.png'
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SCORE_MODEL_FILENAME = 'data/score_model.pickle'
INDICATOR_STEP = 2


def save_frame(screen2save, fname=False):

    if not fname:
        cv2.imwrite(os.path.join(SAMPLE_FOLDER, SCORE_FILENAME), cv2.cvtColor(screen2save, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(os.path.join(SAMPLE_FOLDER, fname), cv2.cvtColor(screen2save, cv2.COLOR_BGR2RGB))

def col_filter(image, color, threshold=[20, 20, 20]):
    # filters specific color from the image
    lower = np.array([ z-threshold[i] for i,z in enumerate(color) ])
    upper = np.array([ z+threshold[i] for i,z in enumerate(color) ])
    # upper = np.array([ z+threshold for z in color ])
    print('filtering by color range:', lower, upper)
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image, image, mask= mask)
    # cv2.GaussianBlur(res, (5,5), 0);
    res = cv2.medianBlur(res, 5);
    res = cv2.medianBlur(mask, 5);

    return res

def col_pick(image, coords=[170,300]):
    # reads the corlor at a X,Y coords of an image
    pick = image[coords[1],coords[0]].tolist()
    return pick

def output(image, some_color, coords, convert=True, indicator=False):
    # displays modifications applied on the whole image;
    # in addtion overlays two blocks - one with provided color and one with the 'inverted' color

    verbose = 0

    if verbose > 0:
        print('image shape', image.shape, 'cell type', image.dtype)
        cv2.imshow('verbose', image)

        try:
            work_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            for row in image:
                [print(z, end=' ') for z in row]
                # print('| 99', row[99])
            


    if convert:
        work_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    else:
        work_image = image

    if indicator:
        cv2.circle(work_image, tuple([coords[0], coords[1]]), 5, [255,0,0])

        # vertices = np.array([[100,100], [150,100], [150,150], [100,150]], np.int32)
        # cv2.fillPoly(work_image, [vertices], some_color)
        # vertices = np.array([[200,100], [250,100], [250,150], [200,150]], np.int32)
        # inverted_color = some_color
        # if isinstance(some_color, list):
        #     inverted_color = [ z for z in some_color[::-1] ]
        # else:
        #     inverted = 255 - some_color
        # cv2.fillPoly(work_image, [vertices], inverted_color)

    return work_image

# def readout(boxes):
#     # captures individual boxes

#     for b in range(1):
#         onebox = list(boxes[b][0] + boxes[b][1])
#         onebox = [ z+50 for z in onebox ]

#         image = np.array(ImageGrab.grab(bbox=tuple(onebox)))

#         img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         img_gray = (255 - img_gray)

#         # black and white
#         ret, bw_image = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

#         image = cv2.resize(bw_image, (8, 8))

#     return bw_image

def stitch_it(images):
    stitcher = cv2.createStitcher(False)
    rez = stitcher.stitch(images)
    return rez[1]



def main():
    grid = []
    score_screen = []
    pick = WALL_COLOR
    coords = [99, 70]
    # coords = [10, 10]
    last_time = time.time()
    workscreen = []
    indicator = True
    all_stitched = None
    map_buffer = []
    counter = 0
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=BOX))
        # new_screen = process_img(screen)
        if len(workscreen) == 0:
            workscreen = screen.copy()
        # if counter % 5 == 0:
        #     map_buffer.append(workscreen)

        tmp_image = cv2.cvtColor(workscreen, cv2.COLOR_BGR2RGB)
        print('Loop took {} seconds'.format(time.time()-last_time), coords, pick, \
            len(map_buffer), '2pick', col_pick(tmp_image, coords), 'scr_pick', screen[coords[1]][coords[0]])

        last_time = time.time()
        cv2.imshow('window', output(workscreen, pick, coords, indicator=indicator))

        if not isinstance(all_stitched, type(None)):
            cv2.imshow('window2', all_stitched)

        if cv2.waitKey(5) & 0xFF == ord('r'): #reset to "fresh" screen
            workscreen = screen.copy()

        if cv2.waitKey(5) & 0xFF == ord('i'): #toggle color picker indicator
            indicator = not indicator

        if cv2.waitKey(5) & 0xFF == ord('f'):
            # workscreen = col_filter(screen, pick, threshold=50)
            # try:
            #     workscreen = col_filter(workscreen, pick, threshold=2)
            # except:
            # for row in screen:
            #     print('99', row[99])

            # workscreen = screen # DEBUGGGG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            filtered_screen = col_filter(workscreen, pick, threshold=[50, 50, 120])
            cv2.imshow('filtered', filtered_screen)

        if cv2.waitKey(5) & 0xFF == ord('t'):    #trigger processing
            print('# IMAGES = ', len(map_buffer))
            all_stitched = stitch_it(map_buffer)
            map_buffer = []

        if cv2.waitKey(5) & 0xFF == ord('p'):
            pick = col_pick(workscreen, coords)
            print('picked col:', pick)

        if cv2.waitKey(5) & 0xFF == ord('w'):
            coords[1] -= INDICATOR_STEP

        if cv2.waitKey(5) & 0xFF == ord('a'):
            coords[0] -= INDICATOR_STEP

        if cv2.waitKey(5) & 0xFF == ord('s'):
            coords[1] += INDICATOR_STEP

        if cv2.waitKey(5) & 0xFF == ord('d'):
            coords[0] += INDICATOR_STEP

        if cv2.waitKey(5) & 0xFF == ord('z'):
            for i, j in enumerate(map_buffer):
                save_frame(j, 'map' + str(i) + '.png')
            print('SAVED')

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        counter += 1

main()