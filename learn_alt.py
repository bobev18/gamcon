import random
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
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# How many times to play the game
EPISODES = 5000

def rgb2gray(rgb):
    r,g,b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def shapeState(img):
    # resize image and make grayscale
    resized_gray = rgb2gray(scipy.misc.imresize(img,(64,64)))/ 255.0
    shaped = resized_gray.reshape(1,64,64,1)
    return shaped

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
# feed 64 by 64 grayscale images into CNN
state_size = (64,64)
action_size = env.action_space.n
agent = player(state_size, action_size)
done = False
batch_size = 32

max_score = 0
frames = []
best = []

for e in range(EPISODES):
    frames = []
    state = env.reset()
    state = env.render(mode='rgb_array')
    state = shapeState(state)
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        pix  = env.render(mode='rgb_array')
        frames.append(pix)
        next_state = shapeState(pix)
        reward = reward if not done else -10
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