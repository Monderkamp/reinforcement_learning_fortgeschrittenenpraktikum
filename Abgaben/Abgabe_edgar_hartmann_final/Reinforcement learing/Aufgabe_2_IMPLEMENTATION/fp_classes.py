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

		self.test = 0

		self.D = 1/8
		self.P_diffstep = 2*self.D
		
		self.x_old = None
		
		if self.P_diffstep > 1.0:
			print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
			exit()
	
	def random_step(self,env_):
		i = np.random.rand()
		if i <= self.P_diffstep:
			j = np.random.randint(0,2)
			if j == 0:
				self.x += 1
			else:
				self.x -= 1
		else:
			self.x = self.x
		
		if self.x >= env_.N_states:
			self.x = 0
		elif self.x < 0:
			self.x = env_.N_states - 1
		else:
			self.x = self.x
		
	def adjust_epsilon(self,episode):
		if episode <= self.zero_fraction*self.N_episodes:
			self.epsilon = 1 - episode/self.zero_fraction/self.N_episodes
		else:
			self.epsilon = 0
		    
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
		if np.random.rand() <= self.epsilon:
			self.choosen_action = np.random.randint(0,3)
		else:
			I ,= np.where(self.Q[self.x,:] == np.amax(self.Q[self.x,:]))		#get index of maximum
			if len(I) > 1:
				i = np.random.randint(0,len(I))
				self.choosen_action = I[i]
			else:
				self.choosen_action = I[-1]

	def perform_action(self,env_):
		"""
		Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""
		if self.choosen_action == 0:
			self.x_old = self.x
			self.x -= 1
		elif self.choosen_action ==  1:
			self.x_old = self.x
			self.x = self.x
		elif self.choosen_action ==  2:
			self.x_old = self.x
			self.x += 1
		else:
			print('choosen_action > 2', self.choosen_action)
		
		if self.x >= env_.N_states:
			self.x = 0
		elif self.x < 0:
			self.x = env_.N_states - 1
		else:
			self.x = self.x
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		if (self.x_old == env_.target_position+1 and self.choosen_action == 0) or (self.x_old == env_.target_position and self.choosen_action == 1) or (self.x_old == env_.target_position-1 and self.choosen_action == 2):
			R = self.target_reward
		else:
			R = 0
		
		self.Q[self.x_old, self.choosen_action] += self.alpha*(R + self.gamma*np.amax(self.Q[self.x,:]) - self.Q[self.x_old, self.choosen_action])
	
	def stoch_obstacle(self,env_):
		pass

