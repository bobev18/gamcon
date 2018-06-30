# spliting overlapped images to individuall digits FAILS
## started Jupyter NB to find alternative approach to splitting



import os
import math
import numpy as np
# import matplotlib.pyplot as plt
import itertools

# from skimage import data
from skimage import transform as tf
from skimage.transform import rotate

import cv2

MIN_CONTOUR_AREA = 50

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

SAMPLE_FOLDER = 'F:\projlogs\gamcon_samples'


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

# def show_raw(images, labels):
#     fig, ax = plt.subplots(ncols=len(images), sharex=False, sharey=False)
#     for i in range(len(images)):
#         ax[i].imshow(images[i], cmap=plt.cm.gray)
#         ax[i].axis('off')
#         if labels is not None:
#             ax[i].set_title(labels[i])

#     plt.show()

# def show_all(images, labels=None):
#     if labels is not None:
#         image_sets = list(grouper(images, 6, np.zeros(images[0].shape, dtype=np.int)))
#         label_sets = list(grouper(labels, 6, ''))
#         for i, image_set in enumerate(image_sets):
#             show_raw(image_set, labels=label_sets[i])
#     else:
#         [ show_raw(z, ['']*6) for z in grouper(images, 6, np.zeros(images[0].shape, dtype=np.int)) ]


def preprocess_score_image(original):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    image = bw_image.copy()
    img_contours, npa_contours, npa_hierarchy = cv2.findContours(image,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    for i, npa_contour in enumerate(npa_contours):
        if cv2.contourArea(npa_contour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npa_contour)
            cv2.rectangle(original, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)
            imgROI = image[intY:intY+intH, intX:intX+intW]
            resized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            flattened = resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            digits.append((intX, flattened,))

    flattened_digits =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT), dtype=np.int)
    for _, digit in sorted(digits, key=lambda z: z[0]):
        flattened_digits = np.append(flattened_digits, digit, 0)

    return flattened_digits


SAMPLES_COUNT = 1002


# load classifications
classifications = []
with open(os.path.join(SAMPLE_FOLDER, 'score_smaples_values.txt'), 'rt') as f:
    test = [ z.strip() for z in f.readlines()]
    classifications = list(itertools.chain.from_iterable(test))
    # classifications.extend([z for z in chars])

print(len(test), test[:15])
print(sum([len(z) for z in test]))
# print(chars[:15])
print(len(classifications), classifications[:15])
# print(len(training_images), len(classifications))



# load sample images
training_samples = []
training_images = []
for i in range(1, SAMPLES_COUNT+1):
    img = cv2.imread(os.path.join(SAMPLE_FOLDER, 'sample_' + str(i) + '.png'))
    # training_images.extend(preprocess_score_image(img))
    training_samples.append(img)
    parts = preprocess_score_image(img)
    if len(test[i]) != len(parts):
        print(i, len(test[i]), len(parts))
    training_images.extend(parts)

print('img smpls', len(training_samples))



if len(training_images) != len(classifications):
    exit(1)


classifications = np.loadtxt("classifications.txt") #, , fmt='%i')
flattened_images = np.loadtxt("flattened_images.txt")#,  fmt='%i')

images = [ np.reshape(z, (RESIZED_IMAGE_HEIGHT, -1)) for z in flattened_images ]
[chr(int(z)) for z in classifications]

show_all(images, labels=[chr(int(z)) for z in classifications])


test_source = cv2.imread("samples/test_score.png")
plt.imshow(test_source, cmap=plt.cm.gray)
plt.show()

test = preprocess_score_image(test_source)
for digit in test:
    print(digit.shape)


test_images = [ np.reshape(z, (RESIZED_IMAGE_HEIGHT, -1)) for z in test ]
show_all(test_images)

# Test sample was taken at 30x30 instead of the 20x30 used by the training samples; will reshape the test sample


print(flattened_images.shape)
print(classifications.shape)

from sklearn import svm

clf = svm.NuSVC()
clf.fit(flattened_images, classifications)

for digit in test:
    predict = clf.predict([digit])
    print(chr(int(predict)))

predictions = [ clf.predict([z]) for z in test ]
show_all(test_images, labels=[ chr(z) for z in predictions])


# Grid search

def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

more_params = {'clf__nu' : [0.8, 0.65, 0.5, 0.35, 0.2],
            'clf__kernel' : ['linear', 'poly', 'rbf', 'sigmoid', ],
            'clf__shrinking' : [True, False],
            'clf__tol' : [0.01, 0.003, 0.001, 0.0003, 0.0001],
            'clf__decision_function_shape' : ['ovo', 'ovr'],
        }

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# need more samples to have the CV

double_samples = np.concatenate((flattened_images, flattened_images), axis=0)
double_classes = np.concatenate((classifications, classifications), axis=0)


def grid_search():
    params = {
        'clf__random_state' : [1337]
    }

    params.update(more_params)

    pipeline = Pipeline([
        ('clf', svm.NuSVC())
    ])

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=2, n_jobs=4)

    grid_search.fit(double_samples, double_classes )
    report(grid_search.cv_results_)

grid_search()

clf = svm.NuSVC(decision_function_shape='ovo', kernel='linear', nu=0.8, random_state=1337, shrinking=True, tol=0.01)
clf.fit(flattened_images, classifications)

test_images = [ np.reshape(z, (RESIZED_IMAGE_HEIGHT, -1)) for z in test ]
show_all(test_images)


for digit in test:
    predict = clf.predict([digit])
    print(chr(int(predict)))

predictions = [ clf.predict([z]) for z in test ]
show_all(test_images, labels=[ chr(z) for z in predictions])


import pickle
with open('score_model2.pickle', 'wb') as f:
    pickle.dump(clf, f)

with open('score_model2.pickle', 'rb') as f:
    clf2 = pickle.load(f)
first_digit = clf2.predict([test[0]])
print('first digit of the score sample is', chr(first_digit))