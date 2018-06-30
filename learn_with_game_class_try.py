# Executes both on Win and LINUX, but ends in infinite loop on the final cycle

import random
import time, datetime
import os
# import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
import scipy

import os
if os.name == 'nt':
    from PIL import ImageGrab
    from directkeys import ReleaseKey, PressKey, W_KEY, A_KEY, S_KEY, D_KEY, R_KEY
    KEYLIST = [W_KEY, A_KEY, S_KEY, D_KEY]
    RESET_KEY = R_KEY
    def tap(key):
        PressKey(key)
        ReleaseKey(key)
    DUMP_FOLDER = 'd:\\temp'

else:
    import pyscreenshot as ImageGrab
    from pynput.keyboard import Key, Controller
    KEYLIST = ['w', 'a', 's', 'd']
    RESET_KEY = 'r'
    keyboard = Controller()
    def tap(key):
        keyboard.press(key)
        keyboard.release(key)
    DUMP_FOLDER = '/media/bob/DATA/temp'


import cv2
import pickle

GRID_COLOR = [185, 172, 160]
MIN_CONTOUR_AREA = 30
MIN_CONTOUR_AREA_GRID = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SCORE_MODEL_FILENAME = 'score_model.pickle'
STATE_SIZE = (64, 64)
SCREEN_CAPTURE_ZONE = (0, 0, 1200, 1080)

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

def preprocess_score_image(original, save_sample=False):
    # takes single RGB image of the score and return numpy array of
    # flattened gray-sacle images of the individual digits, ordered from left to right

    if save_sample:
        colored = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        cv2.imwrite('samples\sample_' + str(save_sample) + '.png', colored)

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

class Game:

    def __init__(self,):
        self.score = 0
        self.score_buffer = deque([0, 0, 0])
        self.keylist = KEYLIST
        self.done = False
        self._count = 0
        # load score model
        try:
            with open(SCORE_MODEL_FILENAME, 'rb') as f:
                self.score_model = pickle.load(f)
        except FileNotFoundError:
            print('failed to load score model from', self.score_model_FILENAME)
            exit(1)

    def reset(self,):
        tap(RESET_KEY)
        # self.score = 0
        # self.score_buffer = deque([0, 0, 0])


    def update_score(self):
        # takes three consecutive reads of the same score to consider it
        if self.score_buffer[0] != self.score_buffer[1] or self.score_buffer[0] != self.score_buffer[2]:
            return

        ### if increased buffer size, for efficient comparison use ###
        # if self.score_buffer.count(self.score_buffer[0]) != len(self.score_buffer):
        #     return self.score

        consider = self.score_buffer[0]

        # new score has to be higher:
        if consider <= self.score:
            return

        # difference in score should be divisible by 4
        if (consider - self.score) % 4 != 0:
            return

        self.score = consider

    def screen_capture(self, save_sample=False):
        self._count += 1
        screen =  np.array(ImageGrab.grab(bbox=SCREEN_CAPTURE_ZONE))
        score_screen, grid_screen, y_offset = split_rois(screen)
        if save_sample:
            flat_score_digits = preprocess_score_image(score_screen, self._count)
        else:
            flat_score_digits = preprocess_score_image(score_screen)


        # display stuff
        viewport = cv2.resize(screen, (0,0), fx=0.3, fy=0.3)
        cv2.imshow('window', viewport)
        cv2.imshow('window2', score_screen)

        return grid_screen, flat_score_digits


    def render(self, force_score_change=False, save_sample=False):
        # print('rendering...')
        detected_score = -1
        old_score = self.score
        start_time = datetime.datetime.now()
        runtime = start_time - start_time
        while (force_score_change and old_score - self.score == 0 and runtime.seconds < 1) or detected_score < 0:
            print(force_score_change, ';;', old_score - self.score, detected_score, runtime.seconds)
            grid_screen, flat_score_digits = self.screen_capture(save_sample)

            try:
                text_score = ''.join([ chr(self.score_model.predict([z])) for z in flat_score_digits ])
                print('text_score', text_score)
                detected_score = int(text_score)
            except ValueError as e:
                detected_score = -1
                # print('failed score detection:', e)

            _ = self.score_buffer.popleft()
            self.score_buffer.append(detected_score)

            self.update_score()
            runtime = datetime.datetime.now() - start_time

        # detected_score = -1
        # while detected_score < 0:
        #     grid_screen, flat_score_digits = self.screen_capture(save_sample)

        #     try:
        #         detected_score = int(''.join([ chr(self.score_model.predict([z])) for z in flat_score_digits ]))
        #     except ValueError as e:
        #         detected_score = -1
        #         print('failed score detection:', e)

        # _ = self.score_buffer.popleft()
        # self.score_buffer.append(detected_score)

        # self.update_score()

        grid_shape = col_filter(grid_screen, GRID_COLOR)
        img_contours, contours, hierarchy = cv2.findContours(grid_shape,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

        rois = []

        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA_GRID:
                [intX, intY, intW, intH] = cv2.boundingRect(contour)
                cv2.rectangle(grid_screen, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)
                imgROI = grid_screen[intY:intY+intH, intX:intX+intW]
                rois.append(imgROI)

        if len(rois) > 0:
            # print(type(rois[0]))
            # print(len(rois[0]))
            # print(rois[0].shape)
            # print([cv2.contourArea(z) for z in rois])
            grid_roi = rois[0]
            gray_grid = cv2.cvtColor(grid_roi, cv2.COLOR_BGR2GRAY)
            grid = cv2.resize(gray_grid, STATE_SIZE)
            done = False
            # cv2.imshow('grid', grid_roi)
            cv2.imshow('grid', grid)
        else:
            grid =  np.zeros(STATE_SIZE, dtype=np.int)
            done = True
            cv2.imshow('grid', grid)

        self.observation_space = grid
        self.done = done
        if done:
            print('DONE '*10)

        return grid, self.score, done

    def step(self, action):
        tap(self.keylist[action])



class Player:
    def __init__(self, img_size, num_actions):
        self.img_size = img_size
        self.num_actions = num_actions
        self.q_table = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.buildCNN()

    def buildCNN(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5,5), input_shape=(64,64,1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(28, activation='relu'))
        # Output layer with two nodes representing Left and Right cart movements
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def write_state(self, state, action, rewardm, next_state, done):
        self.q_table.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions), 'r'
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) , 'm'

    def replay(self, batch_size):
        minibatch = random.sample(self.q_table, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # the model predicts REWARDs given a state
            # the relation to the actions, arises from the consistency when
            #   selecting action - output_node_id -> action_id

            # a good explanations is at:
            # https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)

            # updates the "predicted score" of the given state & action,
            #   the other available actions remain unchanged
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        tstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        pathfile = os.path.join(DUMP_FOLDER, 'model_'+tstamp)
        self.model.save(pathfile)

        return pathfile



# g = Game()
# g.render()
# while(True):
#     if cv2.waitKey(5) & 0xFF == ord('r'):
#         g.render()

#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

# exit(1)



# How many times to play the game
EPISODES = 500
MAX_ACTIONS_IN_EPISODE = 500

# def rgb2gray(rgb):
#     r,g,b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     return 0.2989 * r + 0.5870 * g + 0.1140 * b

# def shapeState(img):
#     # resize image and make grayscale
#     resized_gray = rgb2gray(scipy.misc.imresize(img,(64,64)))/ 255.0
#    shaped = resized_gray.reshape(1,64,64,1)
#    return shaped

def shapeState(img):
    return img.reshape(1,64,64,1)

# env = gym.make('CartPole-v1')
env = Game()
next_state, reward, done = env.render()
old_reward = reward
print(env.observation_space.shape)
print(next_state)
state_size = env.observation_space.shape[0]

# feed 64 by 64 grayscale images into CNN
state_size = (64,64)
action_size = len(env.keylist)
agent = Player(state_size, action_size)
done = False
batch_size = 32

max_score = 0
frames = []
best = []

print('on 1366x768, browser scale down no further than 80%')
print('if it fails to detect score - adjust browser scaling')
print('if score roi misses 1 - reduce MIN_CONTOUR_AREA')
print('get game window in focus')
for i in list(range(5))[::-1]:
    print(i+1)
    time.sleep(1)

# env.render()
# while(True):
#     if cv2.waitKey(5) & 0xFF == ord('r'):
#         env.render()

#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

# exit(1)


for e in range(EPISODES):
    frames = []
    score = -1
    while score != 0:
        state = env.reset()
        state, score, done = env.render(True)
        old_score = score
    print('reset complete with score', score )
    done = False
    # apparently the Conv2D wants a 4D shape like (1,64,64,1)
    state = shapeState(state)
    for time in range(MAX_ACTIONS_IN_EPISODE):
        action, action_type = agent.act(state)



        #!# in the subsequent line, it is unclear what is the object that normally would be
        #                                                             assigned to next_state, but
        #   in the original code the variable is overwritten by the `next_state = shapeState(pix)`
        #   i.e. with image representation of the state;
        #   the goal seems to be to obtain the reward & done values from the `env.action()` call;
        #   while the subsequent call `pix  = env.render()` is to actually get the image capture...

        # next_state, reward, done, _ = env.step(action)
        # pix  = env.render()
        env.step(action)
        pix, score, done = env.render(True)
        # state, reward, done = env.render()
        # while(True):
        # if cv2.waitKey(2) & 0xFF == ord('r'):
        #     env.render()

        if cv2.waitKey(1)%256 == ord('q'):
            cv2.destroyAllWindows()
            exit(1)

        frames.append(pix)
        next_state = shapeState(pix)
        reward = score - old_score if not done else -500
        old_score = score
        print(time, ')', score, reward, action_type, ['w', 'a', 's', 'd'][action])
        agent.write_state(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            if time > max_score:
                max_score = time
                best = frames
            break
    if len(agent.q_table) > batch_size:
        agent.replay(batch_size)

print("Best Score: {}".format(max_score))
saved = agent.save_model()
print('model saved at:', saved)
