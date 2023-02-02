import numpy as np

class environment:
	def __init__(self):
		self.N_states = 15
		self.target_position = 12
		self.starting_position = 8
		 
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

		#self.output_state = 30

		self.D = 0.125
		self.P_diffstep = 2 * self.D 
		
		self.x_old = None
		
		if self.P_diffstep > 1.0:
			print(f"self.P_step = {self.P_diffstep} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
			exit()
	
	def random_step(self):
		self.rand = np.random.rand()
		if self.rand <= self.P_diffstep:	
			self.step = 2*np.random.randint(0,2)-1
			self.x = self.x + self.step
		else :
			self.x = self.x
		return self.x
		
	def adjust_epsilon(self,episode):
		if episode == 0:
			self.epsilon = 1
		elif episode >= self.zero_fraction * self.N_episodes:
			self.epsilon = 0
		else:
			self.epsilon = 1 - 1/(self.N_episodes*self.zero_fraction)*episode
		return self.epsilon
		    
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
		self.rnd = np.random.rand()
		if (self.rnd <= self.epsilon):
			self.chosen_action = np.random.randint(0,3)
		else:
			self.chosen_action = np.argwhere(self.Q[self.x,:] == np.max(self.Q[self.x,:]))
			if (len(self.chosen_action) == 2):
				self.chosen_action = np.random.randint(0,3)
			else:
				self.chosen_action = self.chosen_action[0][0]
		return self.chosen_action


	def perform_action(self,env_):
		"""
		Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""
		self.xmax = env_.N_states
		self.x = self.x + self.chosen_action - 1

		if self.x >= self.xmax:
			self.x = 0
		elif self.x < 0:
			self.x = self.xmax - 1
		
		return self.x
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		self.x_old = self.x - self.chosen_action + 1
		if self.x_old >= self.xmax:
			self.x_old = 0
		elif self.x_old < 0:
			self.x_old = self.xmax - 1

		self.Q[self.x_old,self.chosen_action] += self.alpha *(self.target_reward*(self.x==env_.target_position) + self.gamma*np.max(self.Q[self.x,:]) - self.Q[self.x_old,self.chosen_action])

	def stoch_obstacle(self,env_):
		for i in env_.obstacle_interval:
			if(i == self.x):
				if(np.random.rand() < env_.P_obstacle):
					self.x = self.x - 1
		return self.x