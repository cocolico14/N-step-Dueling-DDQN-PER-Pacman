import random
import gym
import numpy as np
import cv2
from collections import deque
from tqdm import tqdm
from agent import Agent

def preprocess(frame):
    # Got some ideas from https://github.com/ageron/tiny-dqn
    mspacman_color = np.array([210, 164, 74]).mean()
    img = frame[1:176:2,::2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img==mspacman_color] = 0 
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return np.expand_dims(img.reshape(88, 80, 1), axis=0)


env = gym.make('MsPacman-v4') # Skip 4 Frames
state_size = (88, 80, 1)
action_size = env.action_space.n
agent = Agent(state_size, action_size, 4) # 4-step return

episodes = 20000
batch_size = 32
total_time = 0 
all_rewards = 0
done = False

# Initializing Buffer
while len(agent.buffer) < 10000:
  state = preprocess(env.reset())
  frame_stack = deque(maxlen=3) # Deque for getting mean of 3 frames instead of stacking
  frame_stack.append(state)

  for skip in range(90): # Skip first 3 seconds of the game
      env.step(0)

  for time in range(10000):

      state = sum(frame_stack)/len(frame_stack)

      action = agent.act(state)
      next_state, reward, done, _ = env.step(action)

      next_state = preprocess(next_state)
      frame_stack.append(next_state)
      next_state = sum(frame_stack)/len(frame_stack)
      
      td_error = agent.calculate_td_error(state, action, reward, next_state, done)
      
      agent.store(state, action, reward, next_state, done, td_error)

      state = next_state

      if done:
          break
  
print("buffer initialized")

for e in tqdm(range(0, episodes)):
    total_reward = 0
    game_score = 0
    state = preprocess(env.reset())
    frame_stack = deque(maxlen=3)
    frame_stack.append(state)
    
    for skip in range(90):
        env.step(0)
    
    for time in range(20000):
        total_time += 1
        
        if total_time % agent.update_rate == 0:
            agent.update_target_model()
        
        state = sum(frame_stack)/len(frame_stack)
        
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        next_state = preprocess(next_state)
        frame_stack.append(next_state)
        next_state = sum(frame_stack)/len(frame_stack)
        
        td_error = agent.calculate_td_error(state, action, reward, next_state, done)

        agent.store(state, action, reward, next_state, done, td_error)
        
        state = next_state
        total_reward += reward
        
        if done:
            all_rewards += total_reward
            print("episode: {}/{}, reward: {}, avg reward: {}"
                  .format(e+1, episodes, total_reward, all_rewards/(e+1)))
            break
            
        agent.replay(batch_size)

    if (e+1) % 500 == 0:
      print("model saved on epoch", e)
    #   agent.save("")