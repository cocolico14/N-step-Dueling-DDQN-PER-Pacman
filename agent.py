import random
from itertools import count
import numpy as np
import heapq
from collections import deque
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Subtract, Add, Input, Lambda
from keras.optimizers import Adam

class Agent:
  
    def __init__(self, state_size, action_size, n_step=3):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = []
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.cnt = count()
        
        self.alpha = 0.6 # Amount of Prioritization
        self.gamma = 0.99 # Discount Factor
        self.epsilon = 1.0 # Max Prob for Explore
        self.epsilon_min = 0.1 # Min Prob for Explore
        self.epsilon_decay = 0.995 # Decay Rate for Epsilon
        self.update_rate = 1000  # Freq of Network Update
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    def _build_model(self):
      
        inputs = Input(shape=(self.state_size))
        
        x = Conv2D(32, (8, 8), strides=4, padding='same', activation='relu')(inputs)
        x = Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
        x = Flatten()(x)
        
        # Dueling Network
        val = Dense(1, activation='linear')(x)
        advantage = Dense(self.action_size, activation='linear')(x)
        
        # Using Mean for Advantage
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
        advantage = Subtract()([advantage, mean])
        outputs = Add()([val, advantage])
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam())
        
        return model

    def store(self, state, action, reward, next_state, done, td_error):
      
        # n-step queue for calculating return of n previous steps
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
          return
        
        l_reward, l_next_state, l_done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            l_reward = r + self.gamma * l_reward * (1 - d)
            l_next_state, l_done = (n_s, d) if d else (l_next_state, l_done)
        
        l_state, l_action = self.n_step_buffer[0][:2]

        t = (l_state, l_action, l_reward, l_next_state, l_done)
        heapq.heappush(self.buffer, (-td_error, next(self.cnt), t))
        if len(self.buffer) > 100000:
            self.buffer = self.buffer[:-1]
        heapq.heapify(self.buffer)

    def act(self, state):
      
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        
        # Semi Stochastic Prioritization
        prioritization = int(batch_size*self.alpha)
        batch_prioritized = heapq.nsmallest(prioritization, self.buffer)
        batch_uniform = random.sample(self.buffer, batch_size-prioritization)
        batch = batch_prioritized + batch_uniform
        
        batch = [e for (_, _, e) in batch]
        
        for state, action, reward, next_state, done in batch:
            
            if not done:
                n_s = np.expand_dims(next_state.reshape(88, 80, 1), axis=0)
                # Double DQN
                m_a = np.argmax(self.model.predict(n_s)[0])
                target = (reward + self.gamma * self.target_model.predict(n_s)[0][m_a])
            else:
                target = reward
                
            c_s = np.expand_dims(state.reshape(88, 80, 1), axis=0)
            target_f = self.model.predict(c_s)
            target_f[0][int(action)] = target
            self.model.fit(c_s, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def calculate_td_error(self, state, action, reward, next_state, done):
        if not done:
            n_s = np.expand_dims(next_state.reshape(88, 80, 1), axis=0)
            m_a = np.argmax(self.model.predict(n_s)[0])
            target = (reward + self.gamma * self.target_model.predict(n_s)[0][m_a])
        else:
            target = reward

        c_s = np.expand_dims(state.reshape(88, 80, 1), axis=0)
        target_f = self.model.predict(c_s)[0][action]
        
        return target_f - target

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name):
        self.model.save_weights(name)