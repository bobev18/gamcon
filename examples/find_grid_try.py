# implements tools for capruring ROIs (region of interest)
# captures screen and displays results in a window. While focus on the display window,
# allows holding key (press down for few seconds) to execute functions:
#   - r - refresh output screen
#   - f - apply collor filter; the color is prefixed to [185, 172, 160]
#   - g - apply custom filter that captures corners
#   - o - apply grid detection - sorts the coords of the discovered corners and groups them to form the grid
#   - p - pick color under invisible coursor (starts at [170,300]); outputs rectangle with the selected
#         color in the output window and RGB values in console
#   - wasd - navigate the invisible coursor
#   - q - quit

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
from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY
# import pyautogui

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0, stratify=digits.target)

logistic = LogisticRegression().fit(X_train, y_train)
logistic.score(X_test, y_test)

digit = [[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
       [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
       [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
       [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
       [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
       [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
       [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
       [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]]

flatten_digit = np.array( [item for sublist in digit for item in sublist])

prediction = logistic.predict_proba(flatten_digit.reshape(1, -1))
print(prediction)
prediction = list(prediction)
print(prediction.index(max(prediction)))


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    cv2.rectangle(masked, tuple(vertices[0][0]), tuple(vertices[0][2]), (255, 0, 0))
    return masked

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    vertices = np.array([[100,100],[100,1000], [1000,1000], [1000,100]], np.int32)
    processed_img = roi(processed_img, [vertices])
    return processed_img

def col_filter(image, color, threshold=20):
    print('filtering by:', color)
    lower = np.array([ z-threshold for z in color ])
    upper = np.array([ z+threshold for z in color ])
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image, image, mask= mask)
    # cv2.GaussianBlur(res, (5,5), 0);
    res = cv2.medianBlur(res, 5);
    res = cv2.medianBlur(mask, 5);

    return res

def top_right_filter(image):
    pass
    kernel = np.array([[-1, -1, -1, -1, 4],
                       [-1, -1, -1, 4, -1],
                       [-1, -1, 4, -1, -1],
                       [-1, 4, -1, -1, -1],
                       [4, -1, -1, -1, -1]])

    dst = cv2.filter2D(image, -1, kernel)
    return dst

def col_pick(image, coords=[170,300]):
    pick = image[coords[0],coords[1]].tolist()
    return pick

def output(image, some_color, coords, convert=False, indicator=False):
    if convert:
        work_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    else:
        work_image = image

    if indicator:
        cv2.circle(work_image, tuple(coords), 10, [255,0,0])

    vertices = np.array([[100,100], [150,100], [150,150], [100,150]], np.int32)
    cv2.fillPoly(work_image, [vertices], some_color)
    vertices = np.array([[200,100], [250,100], [250,150], [200,150]], np.int32)
    inverted_color = some_color
    if isinstance(some_color, list):
        inverted_color = [ z for z in some_color[::-1] ]
    cv2.fillPoly(work_image, [vertices], inverted_color)

    return work_image

def find_grid(workscreen):
    workscreen = col_filter(workscreen, [185, 172, 160], threshold=2)
    workscreen = top_right_filter(workscreen)
    # mat = np.matrix(workscreen)
    v_sums = workscreen.sum(0)
    h_sums = workscreen.sum(1)
    print(h_sums.size)

    # with open('delme.txt','tw') as f:
    #     f.write(str(h_sums))

    np.savetxt("delme.txt", h_sums, delimiter="\n")

    flat = list(h_sums.flatten())

    print(len(h_sums), len(flat))

    for i, hs in enumerate(h_sums.flatten()):
        print(i, hs)

    print('-'*10)
    for i, vs in enumerate(v_sums):
        print(i, vs)

    ys = h_sums.argsort()[-8:]
    xs = v_sums.argsort()[-8:]
    print('max ? over h_sum', ys)
    print('max over v_sum', xs)

    xs = np.sort(xs)
    ys = np.sort(ys)

    print('xs', xs, '         ys', ys)

    # for x in xs:
    #     cv2.line(workscreen, (x,100),(x,1000), [255,0,0])

    # for y in ys:
    #     cv2.line(workscreen, (100,y),(1000,y), [255,0,0])

    # xs [217 353 378 539 540 677 702 838]
    # ys [282 418 443 580 581 605 742 903]

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

def bump_contrast(image):
    #-----Converting image to LAB Color model-----------------------------------
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    # cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)

    return final



def readout(boxes, model):

    for b in range(1):
        onebox = list(boxes[b][0] + boxes[b][1])
        onebox = [ z+50 for z in onebox ]


        image = np.array(ImageGrab.grab(bbox=tuple(onebox)))
        # contrast
        # image = bump_contrast(image)

        # oldimg = cv2.imread('samples\sample.png',1)
        # combined = np.concatenate((oldimg, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), axis=1)
        # cv2.imwrite('samples\sample.png', combined)

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = (255 - img_gray)

        # black and white
        ret, bw_image = cv2.threshold(img_gray, 127,255,cv2.THRESH_BINARY)

        image = cv2.resize(bw_image, (8, 8))
        print(image)
        prediction = model.predict_proba(flatten_digit.reshape(1, -1))
        print(prediction)
        prediction = list(prediction)
        print(prediction.index(max(prediction)))


    return bw_image, prediction


def main():
    grid = []
    pick = 0
    coords = [170,300]
    last_time = time.time()
    workscreen = []
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(50, 50, 1200, 1200)))
        new_screen = process_img(screen)
        if len(workscreen) == 0:
            workscreen = screen.copy()
        if len(grid) > 0:
            w3, pred = readout(grid, logistic)
            cv2.imshow('window3', w3)

            print('Loop took {} seconds'.format(time.time()-last_time), coords, pick, pred)
        else:
            print('Loop took {} seconds'.format(time.time()-last_time), coords, pick)

        last_time = time.time()
        cv2.imshow('window', new_screen)
        # cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window2', output(workscreen, pick, coords))
        if cv2.waitKey(25) & 0xFF == ord('r'): #reset to "fresh" screen
            workscreen = screen.copy()

        if cv2.waitKey(25) & 0xFF == ord('f'):
            # workscreen = col_filter(screen, pick, threshold=5)
            workscreen = col_filter(workscreen, [185, 172, 160], threshold=2)

        if cv2.waitKey(25) & 0xFF == ord('g'):
            workscreen = top_right_filter(workscreen)
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