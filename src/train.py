import torch
import numpy as np

def train_dqn(env, agent, episodes, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, target_update=10):
    epsilon = epsilon_start
    for episode in range(episodes):
        state = env.reset()
        print(state)
        state = np.transpose(state, (2, 0, 1))  # Channels first pour PyTorch
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))
            agent.store_transition((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            agent.update()

        if episode % target_update == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
