# NOT EXECUTING ON LINUX; executes ok on win, but ML fails to learn...

# https://gabrielecirulli.github.io/2048/
# Model https://github.com/llSourcell/deep_q_learning/blob/master/03_PlayingAgent.ipynb  (https://www.youtube.com/watch?v=79pmNdyxEGo)
#
# Current implementation will drop color info to adhere to the tensor dimensionality in the example
# implementation with color will be in the next iteration

# Because of the score animation, the reading is incosistent.
# Since the input delay/time is not significant for the game result,
# I will make 3 screen captures per action input (adding dely between action and screen read is an alternative)

import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY, R_KEY

import os, pickle
import random
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves

GRID_COLOR = [185, 172, 160]
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SCORE_MODEL_FILENAME = 'score_model.pickle'
STATE_SIZE = (64, 64)

SCREEN_CAPTURE_ZONE = (0, 0, 1200, 1080)

# load score model
try:
    with open(SCORE_MODEL_FILENAME, 'rb') as f:
        score_model = pickle.load(f)
except FileNotFoundError:
    print('failed to load score model from', SCORE_MODEL_FILENAME)
    exit(1)

def preprocess_score_image(original):
    # takes single RGB image of the score and return numpy array of
    # flattened gray-sacle images of the individual digits, ordered from left to right

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    image = bw_image.copy()
    img_contours, contours, hierarchy = cv2.findContours(image,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(contour)
            cv2.rectangle(original, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)
            imgROI = image[intY:intY+intH, intX:intX+intW]
            resized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            flattened = resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            digits.append((intX, flattened,))

    flattened_digits =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT), dtype=np.int)
    for _, digit in sorted(digits, key=lambda z: z[0]):
        flattened_digits = np.append(flattened_digits, digit, 0)

    return flattened_digits

def col_filter(image, color, threshold=20):
    # returns an image where all areas of non-target color are black

    lower = np.array([ z-threshold for z in color ])
    upper = np.array([ z+threshold for z in color ])
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image, image, mask= mask)
    res = cv2.medianBlur(res, 5);
    res = cv2.medianBlur(mask, 5);

    return res

def split_rois(image, grid_color=GRID_COLOR):
    # there are two regions we want:
    # 1. the big grid
    # 2. the score box
    # both are displayed in boxes defined by the GRID_COLOR

    def find_range_start_end(sumlist):
        # in a list of values find the boundries of the first non-zero region of values

        region_detected = False
        region_start = 0
        for i in range(len(sumlist)):
            if sumlist[i] > 0 and not region_detected:
                region_detected = True
                region_start = i

            if sumlist[i] == 0 and region_detected:
                break

        return region_start, i

    # split rois horizontally
    workscreen = col_filter(image, grid_color, threshold=2)
    h_sums = workscreen.sum(axis=1)
    score_screen_start_y, split_at_y = find_range_start_end(h_sums)

    score_screen = image[score_screen_start_y:split_at_y]
    grid_screen = image[split_at_y:]

    # additionall cut horizontally to drop the best score
    workscreen = col_filter(score_screen, grid_color, threshold=2)
    v_sums = workscreen.sum(axis=0)
    score_screen_start_x, score_screen_end_x = find_range_start_end(v_sums)

    # also cut in half vertically to drop the "score" label
    score_screen = score_screen[int(score_screen.shape[0]/2):score_screen.shape[0],
                                 score_screen_start_x:score_screen_end_x]

    return score_screen, grid_screen, split_at_y

def update_score(old_score, score_buffer):
    # takes three consecutive reads of the same score to consider it
    if score_buffer[0] != score_buffer[1] or score_buffer[0] != score_buffer[2]:
        return old_score

    ### if increased buffer size, for efficient comparison use ###
    # if score_buffer.count(score_buffer[0]) != len(score_buffer):
    #     return old_score

    consider = score_buffer[0]

    # new score has to be higher:
    if consider <= old_score:
        return old_score

    # difference in score should be divisible by 4
    if (consider - old_score) % 4 != 0:
        return old_score

    return consider

def capture(score, score_buffer):
    screen =  np.array(ImageGrab.grab(bbox=SCREEN_CAPTURE_ZONE))
    score_screen, grid_screen, y_offset = split_rois(screen)
    flat_score_digits = preprocess_score_image(score_screen)

    # display stuff
    viewport = cv2.resize(screen, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('window', viewport)
    cv2.imshow('window2', score_screen)

    try:
        detected_score = int(''.join([ chr(score_model.predict([z])) for z in flat_score_digits ]))
    except ValueError as e:
        detected_score = -1
        print('failed score detection:', e)

    _ = score_buffer.popleft()
    score_buffer.append(detected_score)

    score = update_score(score, score_buffer)

    grid_shape = col_filter(grid_screen, GRID_COLOR)
    img_contours, contours, hierarchy = cv2.findContours(grid_shape,
                                             cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

    rois = []

    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(contour)
            cv2.rectangle(grid_screen, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)
            imgROI = grid_screen[intY:intY+intH, intX:intX+intW]
            rois.append(imgROI)

    if len(rois) > 0:
        grid_roi = rois[0]
        gray_grid = cv2.cvtColor(grid_roi, cv2.COLOR_BGR2GRAY)
        grid = cv2.resize(gray_grid, STATE_SIZE)
        done = False
        cv2.imshow('grid', grid_roi)
    else:
        grid =  np.zeros(STATE_SIZE, dtype=np.int)
        done = True
        cv2.imshow('grid', grid)

    return grid, score, score_buffer, done

def main():
    last_time = time.time()
    score = 0
    score_buffer = deque([0, 0, 0])
    keylist = [W_KEY, A_KEY, S_KEY, D_KEY]

    # print('in sublime Ctrl+Break(PrintScr) to cancel build')
    # this loop is time to move focus on the 2048 game window
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    # TF INIT
    model = Sequential()

    # model.add(Conv2D(32, kernel_size=(5,5), strides=2, input_shape=(2,) + STATE_SIZE, activation='relu'))
    # # to add RGB:
    # # input_shape=(3,)+STATE_SIZE
    # model.add(Conv2D(16, kernel_size=(3,3), input_shape=(34,34,32), activation='relu'))

    model.add(Dense(20, input_shape=(2,) + STATE_SIZE, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(4, init='uniform', activation='linear'))    # Same number of outputs as possible actions

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Parameters
    action_register = deque()                                # Register where the actions will be stored

    observetime = 500                          # Number of timesteps we will be acting on the game and observing results
    epsilon = 0.7                              # Probability of doing a random move
    gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
    mb_size = 50                               # Learning minibatch size


    # FIRST STEP: Knowing what each action does (Observing)

    # observation = env.reset()                     # Game begins ############ <<< find way to autofocus the game window
    observation, score, score_buffer, done = capture(score, score_buffer)
    # print(observation.shape)
    obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
    state = np.stack((obs, obs), axis=1)
    done = False
    t = 0
    pause = False
    while (t < observetime):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 4, size=1)[0]
        else:
            Q = model.predict(state)          # Q-values predictions
            action = np.argmax(Q)             # Move with highest Q-value is the chosen one

        # observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
        # ------------------ replaced with ------------------
        PressKey(keylist[action])

        for frame in range(3):
            observation_new, score, score_buffer, done = capture(score, score_buffer)

            done = done or (score == 2048) # TODO - states above 2048
            reward = score
            print(t, ')', 'done', done, 'reward', reward)
        #  --------------- end of replacement ---------------

        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
        action_register.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new         # Update state
        if done:
            # env.reset()           # Restart game if it's finished
            reset_counter = 0
            # while (score > 0):
            PressKey(R_KEY)
            print('attempting reset', reset_counter, 'score', score)
            reset_counter += 1
            observation_new, score, score_buffer, done = capture(score, score_buffer)
            time.sleep(0.3)

            done = False
            score = 0
            score_buffer = deque([0, 0, 0])
            observation_new, score, score_buffer, done = capture(score, score_buffer)
            obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
            state = np.stack((obs, obs), axis=1)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(5) & 0xFF == ord('p'):
            print('pausing of 10 seconds')
            for i in range(10,0,-1):
                print(i+1)
                time.sleep(1)

        # time.sleep(1)
        t += 1

    print('Observing Finished')
    t = -1

    # SECOND STEP: Learning from the observations (Experience replay)

    minibatch = random.sample(action_register, mb_size)                              # Sample some moves

    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, len(keylist)))

    for i in range(0, mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]

    # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

    # Train network to output the Q function
        model.train_on_batch(inputs, targets)
    print('Learning Finished')

    # THIRD STEP: Play!

    # observation = env.reset()
    PressKey(R_KEY)
    time.sleep(0.3)
    done = False
    score = 0
    score_buffer = deque([0, 0, 0])
    observation, score, score_buffer, done = capture(score, score_buffer)
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    tot_reward = 0

    while not done and t < observetime*2:
        # env.render()                    # Uncomment to see game running
        Q = model.predict(state)
        action = np.argmax(Q)

        # observation, reward, done, info = env.step(action)
        PressKey(keylist[action])
        observation, score, score_buffer, done = capture(score, score_buffer)
        done = done or (score == 2048) # TODO - states above 2048
        reward = score
        print(t, ')', 'action', 'WASD'[action], 'reward', reward,)

        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
        tot_reward += reward

        if cv2.waitKey(5) & 0xFF == ord('p'):
            print('pausing of 10 seconds')
            for i in range(10,0,-1):
                print(i+1)
                time.sleep(1)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        t += 1

main()

#