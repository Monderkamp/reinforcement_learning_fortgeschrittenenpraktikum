import numpy as np

class agent:
    def __init__(self):
        self.Q = np.zeros((3,3))
    
    def mod(self):
        self.Q[1,1] += 1.0

AG = agent()
AG.mod()
AG.mod()
AG.mod()
print(AG.Q)