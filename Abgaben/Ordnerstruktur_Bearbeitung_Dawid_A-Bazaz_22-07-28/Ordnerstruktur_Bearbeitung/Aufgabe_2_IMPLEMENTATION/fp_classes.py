import numpy as np

class environment:
     def __init__(self):
        self.N_states = 100
        self.target_position = 8
        self.starting_position = 30
         
        self.obstacle_interval = np.arange(9,12)
        self.P_obstacle = 0.0
    
class agent:
    def __init__(self,env_):
        self.N_episodes = 10**4
        self.tmax_MSD = 100
        
        self.x = 1
        self.Q = np.zeros((env_.N_states,3))
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 1.0
        self.target_reward = 10.0
        self.zero_fraction = 0.9

        self.D = 1/8
        self.P_diffstep = 2 * self.D # (1/tau) mit a = 1

        # eigener Code
        self.N_state = env_.N_states
        
        self.x_old = None
        
        if self.P_diffstep > 1.0:
            print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
            print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
            exit()
    
    def correct_out_bound(self):
        if self.x < 0: 
            self.x = self.N_state - 1
        elif self.x > (self.N_state - 1): 
            self.x = 0


    def random_step(self):
        if self.P_diffstep > np.random.rand():
            self.chosen_action = np.random.choice([0, 2])
        
        
    def adjust_epsilon(self,episode):
        end = self.zero_fraction * self.N_episodes
        if episode >= end:
            self.epsilon = 0
        else:
            t = episode / end
            self.epsilon = (1 -t) * 1
            
    def choose_action(self):
        """
        wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
        Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
        """
        self.chosen_action = np.random.randint(0,3)

        if (1 - self.epsilon) > np.random.rand():
            row = self.Q[self.x]
            max = np.max(row)
            if np.count_nonzero(row == max) > 1: pass
            self.chosen_action = np.argmax(row)

        
    def perform_action(self,env_):
        """
        Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
        """
        if self.chosen_action == 0: self.x -= 1
        if self.chosen_action == 2: self.x += 1

        self.correct_out_bound()
    
    def update_Q(self,env_):
        """
        Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
        """
        R = 0 #-self.target_reward * 10
        if self.x == env_.target_position and self.chosen_action == 1: R = self.target_reward

        i = self.x_old
        j = self.chosen_action
        i_ = self.x
        oldQ = self.Q[i][j]
        row = self.Q[i_]
        otherQ = np.max(row)

        
        self.Q[i][j] = oldQ + self.alpha * (R + self.gamma * otherQ - oldQ)

    
    def stoch_obstacle(self,env_):
        pass

