
import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W, A, S, D
import pyautogui


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    vertices = np.array([[50,50],[50,700], [700,700], [700,50]], np.int32)
    processed_img = roi(processed_img, [vertices])
    return processed_img

def get_frame_coords(original_image):
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower = np.array([236, 226, 216])
    # lower = np.array([216, 236, 226])
    upper = np.array([240, 230, 220])
    # upper = np.array([220, 240, 230])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(original_image, original_image, mask= mask)

    vertices = np.array([[100,100], [150,100], [150,150], [100,150]], np.int32)
    cv2.fillPoly(hsv, [vertices], [230, 220, 210])
    vertices = np.array([[200,100], [250,100], [250,150], [200,150]], np.int32)
    cv2.fillPoly(hsv, [vertices], [246, 236, 226])
    # cv2.fillPoly(original_image, [vertices], [0,0,255])
    
    # return mask
    return hsv
    # return original_image
    # return res



def main():
    last_time = time.time()
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(40, 140, 800, 800)))
        new_screen = process_img(screen)
        print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        # cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window2', get_frame_coords(screen))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()