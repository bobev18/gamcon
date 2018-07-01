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

from environment import Game

if os.name == 'nt':
    DUMP_FOLDER = 'd:\\temp'
else:
    DUMP_FOLDER = '/media/bob/DATA/temp'

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
            return random.randrange(self.num_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

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

# How many times to play the game
EPISODES = 5

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

# feed 64 by 64 grayscale images into CNN
state_size = (64,64)
action_size = len(env.keylist)
agent = Player(state_size, action_size)
done = False
batch_size = 32

max_score = 0
frames = []
best = []

print('get game window in focus')
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

for e in range(EPISODES):
    frames = []
    reward = -1
    while reward !=0:
        state = env.reset()
        state, reward, done = env.render()
    # apparently the Conv2D wants a 4D shape like (1,64,64,1)
    state = shapeState(state)
    for time in range(500):
        action = agent.act(state)

        #!# in the subsequent line, it is unclear what is the object that normally would be
        #                                                             assigned to next_state, but
        #   in the original code the variable is overwritten by the `next_state = shapeState(pix)`
        #   i.e. with image representation of the state;
        #   the goal seems to be to obtain the reward & done values from the `env.action()` call;
        #   while the subsequent call `pix  = env.render()` is to actually get the image capture...

        # next_state, reward, done, _ = env.step(action)
        # pix  = env.render()
        env.step(action)
        pix, reward, done = env.render()

        frames.append(pix)
        next_state = shapeState(pix)
        reward = reward if not done else -500
        print(time, ')', reward)
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
env.destroy()