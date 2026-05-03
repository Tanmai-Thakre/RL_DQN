import flappy_bird_gymnasium
import gymnasium as gym
import random
import torch
from DQN import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
    
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR , exist_ok=True)

env = gym.make("FlappyBird-v0", render_mode="human")
    
    
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
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        
        # saving parameter data to RUNS_DIR file 
        self.LOG_FILE = os.path.join(RUNS_DIR , f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR , f"{self.param_set}.pt")
                

    def run(self ,is_training = True , render = False ):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        
        state_num = env.observation_space.shape[0]
        action_num = env.action_space.n
        
        policy_dqn = DQN(state_num , action_num).to(device)
        
        if is_training:
            memory = ReplayMemory(10000)
            epsilon = self.epsilon_init

            target_dqn = DQN(state_num , action_num).to(device)
            # copy wt & bias vals from policy network => target network
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            steps = 0
            
            self.optimizer = optim.Adam(policy_dqn.parameters() , lr = self.alpha)          
            
            best_reward = float("-inf")
            
        else :   
            # Load best model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
        
        
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state , dtype=torch.float , device = device)
            
            terminated = False
            episode_reward = 0
            episode_len = 0
            
            while (not terminated and episode_reward < self.reward_threshold):
                # Next action:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()                                      # Explore
                    action = torch.tensor(action , dtype= torch.long , device = device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()           # Exploit 
                        action = torch.tensor(action , dtype=torch.long , device = device)

                # Processing:
                next_state, reward, terminated, _, _ = env.step(action)
                episode_reward += reward
                
                # Create tensors 
                reward = torch.tensor(reward , dtype = torch.float , device = device)
                next_state = torch.tensor(next_state, dtype = torch.float , device = device)
                
                if is_training:
                    memory.append((state,action,next_state,reward,terminated))
                    steps += 1 
                
                state = next_state
                
                episode_len += 1
                
            print(f"Episode : {episode+1} => Rewards : {episode_reward} & Epsilon : {epsilon}")
                
            # EPSILON DECAY 
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay , self.epsilon_min)   
                
                # save best model
                if episode_reward > best_reward:
                    log_msg = f"Best reward : {best_reward} for episode : {episode+1}"
                    
                    with open (self.LOG_FILE , "a") as file:
                        file.write(f"{log_msg} \n")
                        
                    torch.save(policy_dqn.state_dict() , self.MODEL_FILE)
                    best_reward = episode_reward
                
            if is_training and len(memory) > self.mini_batch_size:
                # Get sample
                mini_batch = memory.sample(self.mini_batch_size)
                
                self.optimize( mini_batch , policy_dqn , target_dqn)
                
                # sync network
                if steps > self.netwaork_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
                          
            
        # env.close() ----> Manually ending 


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # calculate target Q-values - if terminations=true => zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

            
        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)