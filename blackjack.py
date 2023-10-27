import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from blackjackEnv import BlackjackEnvironment

class DQNAgent:
    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=kwargs.get('memory_size', 2000))
        self.gamma = kwargs.get('gamma', 0.95)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.9995)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def load(self, name):
        self.model = load_model(name)


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=2000, batch_size=32, print_interval=100):
    env = BlackjackEnvironment()
    state_size = 2
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    previous_win_percentage = 0

    for episode in range(1, episodes + 1):
        state = np.reshape(env.reset(), [1, state_size])
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if reward > 0:
                stats['wins'] += 1
            elif reward < 0:
                stats['losses'] += 1
            else:
                stats['draws'] += 1
            
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            if done:
                total_games = stats['wins'] + stats['losses']
                win_percentage = (stats['wins'] / total_games * 100) if total_games != 0 else 0
                print(f"Episode: {episode}, Total Reward: {total_reward}, Stats: {stats}, Win Percentage: {win_percentage:.2f}%, Epsilon:{agent.epsilon:.2f}")
                break
        
        if episode % print_interval == 0:
            total_games = sum(stats.values())
            win_percentage = (stats['wins'] / total_games) * 100
            percentage_change = ((win_percentage - previous_win_percentage) / previous_win_percentage) * 100 if previous_win_percentage != 0 else 0
            print(f"Episode: {episode}/{episodes}, Win Percentage: {win_percentage:.2f}%, Percentage Change: {percentage_change:.2f}%, Epsilon: {agent.epsilon:.2f}")
            previous_win_percentage = win_percentage

    agent.model.save('blackjack_model.h5')

if __name__ == "__main__":
    train_agent()


