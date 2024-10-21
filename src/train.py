import numpy as np
from collections import deque
from utils import preprocess_frame
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_dqn(env, agent, episodes, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, target_update=10):
    epsilon = epsilon_start
    for episode in tqdm(range(episodes), desc="Training Progress"):
        logging.info(f"Starting episode {episode}")
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        frame = preprocess_frame(state)
        state_stack = deque([frame] * 4, maxlen=4)
        done = False
        total_reward = 0

        while not done:
            state_input = np.stack(state_stack, axis=0)
            action_idx, action = agent.choose_action(state_input, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_frame = preprocess_frame(next_state)
            state_stack.append(next_frame)
            next_state_input = np.stack(state_stack, axis=0)

            agent.store_transition((state_input, action_idx, reward, next_state_input, done))

            total_reward += reward
            agent.update()

            logging.info(f"Action taken: {action}, Reward: {reward}, Done: {done}")

        if episode % target_update == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        logging.info(f"Episode {episode} finished, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
