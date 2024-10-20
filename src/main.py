from dqn_agent import DQNAgent
from train import train_dqn
import gymnasium as gym
import ale_py


gym.register_envs(ale_py)


def main():
    env = gym.make("ALE/Pong-v5")
    input_shape = (4, 84, 84)  # 4 frames of 84x84 pixels
    actions = [0, 2, 3]  # NOOP, UP, DOWN
    num_actions = len(actions)
    agent = DQNAgent(input_shape, num_actions, actions)
    train_dqn(env, agent, episodes=500)


if __name__ == "__main__":
    main()
