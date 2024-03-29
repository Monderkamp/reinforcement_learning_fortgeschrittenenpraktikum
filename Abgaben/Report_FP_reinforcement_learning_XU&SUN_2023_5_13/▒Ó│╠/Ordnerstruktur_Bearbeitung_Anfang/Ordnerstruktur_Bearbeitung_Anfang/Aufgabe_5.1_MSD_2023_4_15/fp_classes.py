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
		self.N_episodes = 20000
		self.tmax_MSD = 100
		
		self.x = 0
		self.Q = np.zeros((env_.N_states,3))
		self.alpha = None
		self.gamma = None
		self.epsilon = 1.0
		self.target_reward = 10.0
		self.zero_fraction = 0.9

		self.D = 0.9
		self.P_diffstep = 2*self.D
		
		self.x_old = None
		
		if self.P_diffstep > 1.0:
			print(f"self.P_diffstep = {self.P_diffstep} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")

	
	def random_step(self):
            Random_number1=np.random.rand()
            if Random_number1<self.P_diffstep:
                Random_number2 = np.random.randint(0, 2)
                if Random_number2 == 1:
                        self.x=self.x + 1
                else:
                        self.x=self.x-1
            else:
                self.x=self.x
	def adjust_epsilon(self,episode):
		pass
		    
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
		pass
	def perform_action(self,env_):
		"""
		Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

		pass
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		pass
	
	def stoch_obstacle(self,env_):
		pass

