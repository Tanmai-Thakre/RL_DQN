import flappy_bird_gymnasium
import gymnasium as gym
import torch
from DQN import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make("FlappyBird-v0", render_mode="human")


if torch.backend.mps.isavailable():
    device = "mps"
elif torch.backend.cuda.isavailable():
    device = "cuda"
else:
    device = "cpu"
    
class Agent:
    def __init__(self , param_set):
        self.param_set = param_set
        
        with open ("parameters.yaml" , 'r') as file:
            all_params_set = yaml.safe_load(file)
            parameters = all_params_set[param_set]
            
        self.replay_memory_size = parameters["replay_memory_size"]
        self.mini_batch_size = parameters["mini_batch_size"]
        
        self.netwaork_sync_rate = parameters["netwaork_sync_rate"]      
        self.reward_threshold = parameters["reward_threshold"]
        
        self.epsilon_min = parameters["epsilon_min"]
        self.epsilon_init = parameters["epsilon_init"]
        self.epsilon_decay = parameters["epsilon_decay"]
        
        self.alpha = parameters["alpha"]
        self.gamma = parameters["gamma"]
                

    def run(self ,is_training = True , render = False ):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        
        action_num = env.action_space.n
        state_num = env.observation_space.shape[0]
        
        policy_dqn = DQN(state_num , action_num).to(device)
        
        if is_training:
            memory = ReplayMemory(10000)
        
        
        for episode in itertools:
            state, _ = env.reset()
            terminated = False
            episode_reward = 0
            episode_len = 0
            
            while not terminated:
                # Next action:
                action = env.action_space.sample()

                # Processing:
                next_state, reward, terminated, _, _ = env.step(action)
                
                if is_training:
                    memory.append((state,action,next_state,reward,terminated))
                
                state = next_state
                episode_reward += reward
                episode_len += 1
                
            print(f"Episode : {episode+1} => Rewards : {episode_reward}")
            
        # env.close() ----> Manually ending 
