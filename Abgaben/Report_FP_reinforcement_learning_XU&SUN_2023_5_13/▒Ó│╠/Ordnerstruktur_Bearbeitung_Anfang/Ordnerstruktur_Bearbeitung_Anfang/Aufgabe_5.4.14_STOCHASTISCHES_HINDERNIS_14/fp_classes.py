import numpy as np

class environment:
    def __init__(self):
        self.N_states = 15
        self.target_position = 12
        self.starting_position = 8
        
        self.obstacle_interval = np.arange(9,12)
        self.P_obstacle = 0.7
	
class agent:
    def __init__(self,env_):
        self.N_episodes =1000
        self.tmax_MSD = 100
		
        self.Q = np.zeros((env_.N_states,3))
        self.alpha = 0.999999
        self.gamma = 0.9
        self.epsilon = 1.0
        self.target_reward = 10.0
        self.zero_fraction = 0.9
        
        self.D = 0
        self.P_diffstep = 2*self.D
		
        self.x_old = None
		
        if self.P_diffstep > 1.0:
            print(f"self.P_diffstep = {self.P_diffstep} > 1.0 in agent.__init__(...)")
            print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")

	
    def random_step(self):
                Random_number2 = np.random.randint(0, 2)
                
                if Random_number2 == 0:
                    self.chosen_action=0
                else:
                    self.chosen_action=2
    def adjust_epsilon(self,episode):
        k = 1 / (self.zero_fraction * self.N_episodes)
        if episode < self.zero_fraction * self.N_episodes:
            self.epsilon = 1 - k * episode
        else:
            self.epsilon = 0
    def choose_action(self):
            Random_number3=np.random.rand()
            if Random_number3 < self.epsilon:
                self.chosen_action= np.random.randint(0,3) 
                
            else:
                
                self.chosen_action = np.argmax(self.Q[self.x])
                
                
                """
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
    def perform_action(self,env_):
        self.x_old=self.x
        if self.x_old == 0 and self.chosen_action == 0:
            self.x = env_.N_states-1
        elif self.x == env_.N_states-1 and self.chosen_action == 2:
            self.x = 0
        else:
            
            if  self.chosen_action == 0:
                        self.x=self.x-1
            if  self.chosen_action == 1:
                        self.x=self.x
            if  self.chosen_action == 2:
                        self.x=self.x+1
        if self.epsilon == 0 :   
            """
		Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

	
    def update_Q(self,env_):
        R=0
        if self.chosen_action == 1:
            if self.x ==  env_.target_position :
                R = self.target_reward
            else:
                R = 0
        self.Q[self.x_old,self.chosen_action]  = self.Q[self.x_old,self.chosen_action]+ (self.alpha
        *( R + self.gamma*np.max(self.Q[self.x])-self.Q[self.x_old,self.chosen_action]))
        
        """
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""

	
    def stoch_obstacle(self,env_):
        Random_number4=np.random.rand()
        if self.x in  env_.obstacle_interval:
            if Random_number4<env_.P_obstacle:
            #if Random_number4<P_obstacle_i:
                self.x=self.x-1
            else:
                self.x=self.x
        else:
            self.x=self.x

