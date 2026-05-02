from collections import deque
import random 

# Creating FIFO queue experience relay

class ReplayMemory():
    def __init__(self , maxlen , seed=None):
        self.memory = deque([], maxlen=maxlen)
        
    def append(self, new_exp):
        self.memory = deque.append(new_exp)
        
    def sample(sample_size):
        return random(self.memory , sample_size)
    
    def __len__(self):
        return len(self.memory)