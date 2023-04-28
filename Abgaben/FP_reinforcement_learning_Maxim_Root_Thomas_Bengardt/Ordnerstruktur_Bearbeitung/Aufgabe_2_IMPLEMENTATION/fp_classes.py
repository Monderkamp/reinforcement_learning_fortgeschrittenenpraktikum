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

		self.D = 0.125
		self.P_diffstep = 2*self.D

		self.x_old = None

		if self.P_diffstep > 1.0:
			print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
			exit()

	def random_step(self,env_):
		p = np.random.rand() #probability
		p_step = np.random.randint(0,2) #plus or minus step probability
		if p <= self.P_diffstep:
			#Randbedingungen
			if self.x == 0 and p_step == 1:
				self.x = env_.N_states - 1
			elif self.x == (env_.N_states - 1) and p_step == 0:
				self.x = 0
			else:
				if p_step == 0:
					self.x += 1
				else:
					self.x -= 1

	def adjust_epsilon(self,episode):
		slope = -1/(self.N_episodes*self.zero_fraction) #linear decrease
		self.epsilon = 1 + slope * episode

	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
		if np.random.rand() <= self.epsilon:
			self.chosen_action = np.random.randint(0,3)
		else:
			max = np.max(self.Q[self.x])
			if len(np.where(self.Q[self.x] == max)[0]) == 1: #if only one maximum
				self.chosen_action = np.where(self.Q[self.x] == max)[0][0] #index of maximum
				# print(self.chosen_action)
			elif len(np.where(self.Q[self.x] == max)[0]) == 2: #if 2 max
				random_index = np.random.randint(0,2)
				self.chosen_action = np.where(self.Q[self.x] == max)[0][random_index]
			else:
				self.chosen_action = np.random.randint(0,3) #random choice for more than 3 max

	def perform_action(self,env_):
		"""
		Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""
		#x € [0,99]
		self.x_old = self.x # abspeichern für update_Q
		dx = self.chosen_action - 1 #Verschiebung
		#Randbedingungen
		if self.x == 0 and dx == -1:
			self.x = env_.N_states - 1
		elif self.x == (env_.N_states - 1) and dx == 1:
			self.x = 0
		else:
			self.x += dx

	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		# print(np.shape(self.Q))
		# print(self.x_old,self.chosen_action)
		if self.x_old == self.x and self.x == env_.target_position: #falls auf target verblieben
			self.Q[self.x_old,self.chosen_action] += self.alpha*(self.target_reward + self.gamma*np.max(self.Q[self.x]) - self.Q[self.x_old,self.chosen_action]) #mit Belohnung
		else:
			self.Q[self.x_old,self.chosen_action] += self.alpha*(self.gamma*np.max(self.Q[self.x]) - self.Q[self.x_old,self.chosen_action]) #ohne Belohnung

	def stoch_obstacle(self,env_):
		pass
