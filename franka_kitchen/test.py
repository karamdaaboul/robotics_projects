import time
import gymnasium as gym
import numpy as np
from wrappers import RoboGymObservationWrapper
from replay_buffer import ReplayBuffer
from networks import *
from agent import *

if __name__ == '__main__':
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 500
    replay_buffer_size = 1000000
    task = "microwave"
    task_no_spaces = task.replace(" ", "_")
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    update_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 256

    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human')
    env = RoboGymObservationWrapper(env, goal=task)

    observation, info = env.reset()
    
    observation_size = observation.shape[0]

    agent = Agent(observation_size, env.action_space, gamma, tau, 
                  alpha, target_update_interval, hidden_size, learning_rate,
                  goal=task)

    memory = ReplayBuffer(replay_buffer_size, input_size=observation_size, 
                          n_actions=env.action_space.shape[0])
    
    agent.load_checkpoint(checkpoint_dir ='checkpoints/checkpoint_120', evaluate=True)

    agent.test(env, episodes=2, max_episode_steps=max_episode_steps)

    env.close()