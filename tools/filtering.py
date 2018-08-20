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


import numpy as np
import cv2
import time
import os, pickle


# with open('/home/bob/Pictures/5.jpg', 'rt') as f:
#     data = f.read()
# print(len(data))
# print('-----------')

image = cv2.imread('/home/bob/Pictures/small.jpg', 1)
print(image.shape)


def col_filter(image, color, color2=None, threshold=[20, 20, 20]):
    # filters specific color from the image
    lower = np.array([ z-threshold[i] for i,z in enumerate(color) ])
    upper = np.array([ z+threshold[i] for i,z in enumerate(color) ])
    print('filtering by color range:', lower, upper)
    mask = cv2.inRange(image, lower, upper)
    cv2.imshow('mask', mask)
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('biwise', res)

    if color2 != None:
        lower2 = np.array([ z-threshold[i] for i,z in enumerate(color2) ])
        upper2 = np.array([ z+threshold[i] for i,z in enumerate(color2) ])
        print('filtering by color range2:', lower2, upper2)
        mask2 = cv2.inRange(image, lower2, upper2)
        cv2.imshow('mask2', mask2)
        alt = cv2.bitwise_and(image, image, mask=mask2)
        cv2.imshow('biwise2', alt)

        mix = cv2.bitwise_or(res, alt)
        cv2.imshow('mix', mix)
        return mix

    return res

image = image[7:250, 1035:1275]

cols = [125, 135, 150, 120, 120, 140]
old_i =  index = 0
old_c = []


while(True):

    if cols != old_c or index != old_i:
        filtered_screen = col_filter(image, cols[:3], cols[3:], threshold=[40, 40, 60])
        cv2.imshow('filtered', filtered_screen)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(5) & 0xFF == ord('w'):
        cols[index] += 10

    if cv2.waitKey(5) & 0xFF == ord('a'):
        index -= 1

    if cv2.waitKey(5) & 0xFF == ord('s'):
        cols[index] -= 10

    if cv2.waitKey(5) & 0xFF == ord('d'):
        index += 1

    old_i = index
    old_c = cols.copy()

cv2.destroyAllWindows()
