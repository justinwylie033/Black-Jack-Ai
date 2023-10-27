from blackjackEnv import BlackjackEnvironment
from blackjack import DQNAgent
import numpy as np

def evaluate_agent(model_path, episodes=100):
    env = BlackjackEnvironment()
    state_size = 2
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)  # Load the trained model
    agent.epsilon = 0  # Set epsilon to 0 to disable exploration
    
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    
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
            
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            
            if done:
                total_games = stats['wins'] + stats['losses']
                win_percentage = (stats['wins'] / total_games * 100) if total_games != 0 else 0
                print(f"Episode: {episode}, Total Reward: {total_reward}, Stats: {stats}, Win Percentage: {win_percentage:.2f}%")
                break

if __name__ == "__main__":
    evaluate_agent('blackjack_model.h5')
