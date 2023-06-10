# A4
# MODULE
import numpy as np

# KLASSEN
class environment:
     def __init__(self):
        self.N_states = 15 #Anstatt 100
        self.target_position = 12
        self.starting_position = 8

        self.obstacle_interval = np.arange(9,12)
        self.P_obstacle = None
    
class agent:
    def __init__(self,env_):
        self.N_episodes = 10**2
        #self.tmax_MSD = 100
        
        self.x = None
        self.Q = np.zeros((env_.N_states,3))
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 1.0
        self.target_reward = 10.0
        self.zero_fraction = 0.9

        self.D = 0.125
        self.P_diffstep = 2*self.D
        
        self.x_old = None

        #self.actions_per_episode = []
        #self.actions = None
        #self.actions_min = None


    # Diffusion
    def random_step(self):
        if np.random.rand() < self.P_diffstep:
            if np.random.randint(0,2) == 1:
                self.x += 1
            else:
                self.x -= 1       

    # Anpassung von Epsilon
    def adjust_epsilon(self,episode):
        if episode < (self.N_episodes*self.zero_fraction):
            self.epsilon = 1 - episode/(self.N_episodes*self.zero_fraction)
        else:
            self.epsilon = 0
        

    # Auswahl der Aktion des Agenten
    def choose_action(self):
        max_val = np.max(self.Q[self.x])                # Maximum auslesen
        max_count = (self.Q[self.x] == max_val).sum()   # Anzahl der Maxima auslesen

        if (np.random.rand() < self.epsilon) or (max_count >1):
            self.chosen_action = np.random.randint(0,3)
        else:
            self.chosen_action = np.argmax(self.Q[self.x])


    # Durchführung der Aktion unter periodischen Randbedingungen
    def perform_action(self,env_):
        # Periodizität nach Diffusion sicherstellen
        if self.x > (env_.N_states-1) : self.x -= env_.N_states
        if self.x < 0 : self.x += env_.N_states
        
        self.x_old = self.x        # Zustand vor einer Aktion speichern

        match self.chosen_action:  # Zustand durch Aktion verändern
            case 0: self.x -= 1
            case 1: pass
            case 2: self.x += 1

        # Periodizität nach Aktion sicherstellen
        if self.x > (env_.N_states-1) : self.x -= env_.N_states
        if self.x < 0 : self.x += env_.N_states
     
    
    # Aktualisierung der Einträge der Q-Matrix nach jeder Aktion
    def update_Q(self,env_):
        max_val = np.max(self.Q[self.x])                   # Maximum des Folgezustandes

        if (self.x == self.x_old == env_.target_position):  # Prüfen, ob Agent im Zielzustand verweilt 
            finished = 1
        else:
            finished = 0

        # Neuen Q-Wert je nach Zustand sowie Aktion bestimmen und zuweisen
        self.Q[self.x_old][self.chosen_action] += \
        self.alpha*(finished*self.target_reward + self.gamma*max_val - self.Q[self.x_old][self.chosen_action])
    
    def stoch_obstacle(self,env_):
        if (self.x in env_.obstacle_interval) and (np.random.rand() < env_.P_obstacle):
            self.x -= 1