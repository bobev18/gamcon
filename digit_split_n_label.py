# uses stored image with digits from 0 to 9 to split them into individual arrays and annotate them
# i.e. adds labels & prepares the data for ML use

import numpy as np
import cv2
import sys
# import time

# import numpy as np
# import cv2
# import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
# font = cv2.FONT_HERSHEY_SIMPLEX
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

original = cv2.imread("samples\score_digits.png")

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("window1", gray)

# gray = cv2.GaussianBlur(gray, (5,5), 0)

# filter image from grayscale to black and white
_, bw_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("window2", bw_image)

image = bw_image.copy()

img_contours, npa_contours, npa_hierarchy = cv2.findContours(image,
                                             cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)



flattened_images =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT), dtype=np.int)

classifications = []

chars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),]

for i, npa_contour in enumerate(npa_contours):
    if cv2.contourArea(npa_contour) > MIN_CONTOUR_AREA:
        [intX, intY, intW, intH] = cv2.boundingRect(npa_contour)

        cv2.rectangle(original, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)
        imgROI = image[intY:intY+intH, intX:intX+intW]
        resized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # iterator_img = np.zeros((RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH), dtype=np.int)
        # cv2.putText(iterator_img,str(i),(1,1), font, 1, 255)
        # print(i)
        # print(iterator_img.shape, imgROIResized.shape)
        # combined = np.concatenate((imgROIResized, iterator_img), axis=1)
        cv2.imshow("imgROI", imgROI)
        cv2.imshow("resized"+str(i), resized)
        cv2.imshow("original", original)
        # cv2.imshow("combined"+str(i), iterator_img)

        char = cv2.waitKey(0)
        if char == 27:
            sys.exit()
        elif char in chars:
            classifications.append(char)
            flattened = resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            flattened_images = np.append(flattened_images, flattened, 0)


print(classifications)
# fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

# npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

# print "\n\ntraining complete !!\n"


np.savetxt("delme_classifications.txt", classifications, fmt='%i')
np.savetxt("delme_flattened_images.txt", flattened_images, fmt='%i')


# intChar = cv2.waitKey(0)
# if intChar == 27:
#     sys.exit()

cv2.destroyAllWindows()

