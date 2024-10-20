from dqn_agent import DQNAgent
from train import train_dqn
import gym

# def list_pong_envs():
#     envs = gym.envs.registry.values()
#     pong_envs = [env.id for env in envs if "Pong" in env.id]
#     print("Environnements Pong disponibles :")
#     for env_id in pong_envs:
#         print(env_id)
#     return pong_envs
  

def main():
    env = gym.make("Pong-v0")
    input_shape = (4, 84, 84) 
    num_actions = env.action_space.n

    agent = DQNAgent(input_shape, num_actions)
    train_dqn(env, agent, episodes=500)

if __name__ == "__main__":
  main()
