

digit_split_n_label.py -- produces "flattened_images.txt" and "flattened_images.txt" via interactive processing(splitting) and labling of digits 0 to 9 based on raw image "samples/score_digits.png"

flattened_images.txt -- flattened digit images
classifications.txt -- labels for the "flattened_images.txt"

mydigits.ipynb -- uses flattened images of digits to add distortions
single_digits.ipynb -- uses flattened images of digits to grid search ML params, saves to "score_model.pickle"

train_on_overlapping_digits.ipynb -- uses raw images of scores with overlap to
                               == split digits
                               == flatten images
                               == train ML





unchecked ==
  learn_alt.py
  environment.py


the new digit split seems to miss digits


TODO:

 - adjust the score update method
 - run tests with overlapping and record more samples

 - encapsulate keyboard interactions
 - find a way to send signal to the app, while focus is on the game (i.e. via mouse)
 - refactor score
 - clean/refresh requirements
 - have the min constants dependent on environment

