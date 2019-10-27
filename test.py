import gym
from gym.wrappers import Monitor
import numpy as np
import cv2
from collections import deque
from agent import Agent

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env

def preprocess(frame):
    # Got some ideas from https://github.com/ageron/tiny-dqn
    mspacman_color = np.array([210, 164, 74]).mean()
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(frame,kernel,iterations = 2)
    img = frame[1:176:2,::2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img==mspacman_color] = 0 
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img.reshape(88, 80, 1)

env = wrap_env(gym.make("MsPacman-v4"))
state_size = (88, 80, 1)
action_size = 5
agent = Agent(state_size, action_size, 5)
# agent.load("")

state = preprocess(env.reset())
frame_stack = deque(maxlen=4)
frame_stack.append(state)

for skip in range(90):
    env.render()
    env.step(0)

while True:
    
    env.render()
    state = sum(frame_stack)/len(frame_stack)
    
    action = agent.act(np.expand_dims(state.reshape(88, 80, 1), axis=0))
    next_state, reward, done, _ = env.step(action)
    
    next_state = preprocess(next_state)
    frame_stack.append(next_state)
    next_state = sum(frame_stack)/len(frame_stack)

    state = next_state
    
    if done:
        break

env.close()
