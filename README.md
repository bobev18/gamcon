

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


TODO:

 - push/clean legacy files in folders:
   = i.e. capture_screen_try in folder "examples"
   = score_getter in "tools"
   = pickles and flat files in "data"
   ...

 - clean/refresh requirements

