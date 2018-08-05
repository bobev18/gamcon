import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY
# import pyautogui
import os, pickle

from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()


SAMPLE_FOLDER = 'samples'

def stitch_it(images):
    stitcher = cv2.createStitcher(False)
    rez = images[0]
    for i in images[1:]:
        rez = stitcher.stitch(rez, i)
    return rez

def output(image, convert=True):
    # displays modifications applied on the whole image;
    # in addtion overlays two blocks - one with provided color and one with the 'inverted' color
    if convert:
        work_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    else:
        work_image = image


    return work_image

maps = []

for i in range(0, 39, 5):
    test_image = cv2.imread(os.path.join(SAMPLE_FOLDER, 'map'+str(i)+'.png'))
    print((test_image))
    maps.append(test_image)
print(type(maps[0]))
print(len(maps[0]))
# cv2.imshow('map0', maps[0])
# cv2.imshow('map8', maps[8])

# _ = input('Enter your input:')


# stitcher = cv2.createStitcher(False)
stitcher = cv2.createStitcher(try_use_gpu=True)
rez = stitcher.stitch(maps)


if not isinstance(rez[1], type(None)):
    cv2.imshow('window', rez[1])

while(True):
    if cv2.waitKey(5) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
