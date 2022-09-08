import numpy as np

class environment:
	 def __init__(self):
		 self.N_states = 15
		 self.target_position = 12
		 self.starting_position = 9
		 
		 self.obstacle_interval = np.arange(10,12)
		 self.P_obstacle = 0.10
	
class agent:
	def __init__(self,env_):
		self.N_episodes = 10**4
		self.tmax_MSD = 100

		self.x = env_.starting_position
		self.a = 1
		self.Q = np.zeros((env_.N_states,3))
		self.alpha = .999999
		self.gamma = .9
		self.epsilon = 1.0
		self.target_reward = 10.0
		self.zero_fraction = 0.9
        
		self.chosen_action = None

		self.D = 0.125
		self.D = 0.0
		self.tau = 1
		self.P_diffstep = self.D * 2 * self.tau / (self.a**2)
		
		self.x_old = None
        
		
		if self.P_diffstep > 1.0:
			print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
			exit()
	
	def random_step(self,env_):
		random_nr = np.random.rand()
		step = 0
		if (random_nr <= self.P_diffstep):
			step = np.random.randint(0,2)
			if step == 0:
				step = -1
			else:
				step = 1
		self.x = self.x + step
		if (self.x > env_.N_states - 1):
			self.x = 0
		if (self.x < 0):
			self.x = env_.N_states - 1
		
	def adjust_epsilon(self,episode):
		self.epsilon = 1 - episode * 1 / (self.zero_fraction * self.N_episodes)
		    
	def choose_action(self):
		rand_basic_val = np.random.rand()
		if (rand_basic_val <= self.epsilon):
			self.chosen_action = np.random.randint(0,3)
		else:
			self.chosen_action = np.argmax(self.Q[self.x:self.x + 1]) # füge noch hinzu dass wenn 2 maxima....
			
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
	def perform_action(self,env_):
		self.x_old = self.x
		if (self.chosen_action == 0):
			self.x = self.x - 1
		elif (self.chosen_action == 2):
			self.x = self.x + 1
		if (self.x > env_.N_states - 1):
			self.x = 0
		if (self.x < 0):
			self.x = env_.N_states - 1
            
		"""
		Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

	
	def update_Q(self,env_):
		max_q_i_strich_j_index = np.argmax(self.Q[self.x:self.x + 1])
		mqis = self.Q[self.x,max_q_i_strich_j_index]
		if (self.x == env_.target_position):
			self.Q[self.x_old, self.chosen_action] = self.Q[self.x_old, self.chosen_action] + self.alpha * (self.target_reward + self.gamma * mqis - self.Q[self.x_old, self.chosen_action])
		else:
			self.Q[self.x_old, self.chosen_action] = self.Q[self.x_old, self.chosen_action] + self.alpha * (self.gamma * mqis - self.Q[self.x_old, self.chosen_action])
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
	
	def stoch_obstacle(self,env_):
		if (self.x in env_.obstacle_interval):
			rand_val = np.random.rand()
			if (rand_val <= env_.P_obstacle):
				self.x -= self.x
				if (self.x < 0):
					self.x = env_.N_states - 1
                



