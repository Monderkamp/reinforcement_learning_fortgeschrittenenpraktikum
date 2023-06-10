# A1
# MODULE
import numpy as np

# KLASSEN
class environment:
     def __init__(self):
        self.N_states = 100
        self.target_position = 8
        self.starting_position = 30
    
class agent:
    def __init__(self,env_):
        self.N_episodes = 20000
        self.tmax_MSD = 100
        
        self.x = 1
        #self.Q = np.zeros((env_.N_states,3))
        #self.alpha = None
        #self.gamma = None
        #self.epsilon = 1.0
        #self.target_reward = 10.0
        #self.zero_fraction = 0.9

        self.D = 0.5                # Diffusionskonstante (wurde für A 1.6 manuell geändert)
        self.P_diffstep = 2*self.D  # Wahrscheinlichkeit für einen Diffusionsschritt
        
        if self.P_diffstep > 1.0:
            print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
            print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
            exit()
    
    # Diffusion
    def random_step(self):
        if np.random.rand() < self.P_diffstep:
            if np.random.randint(0,2) == 1:
                self.x += 1
            else:
                self.x -= 1
    
    def adjust_epsilon(self,episode):
        pass
            
    def choose_action(self):
        pass

    def perform_action(self,env_):
        pass
    
    def update_Q(self,env_):
        pass
    
    def stoch_obstacle(self,env_):
        pass

