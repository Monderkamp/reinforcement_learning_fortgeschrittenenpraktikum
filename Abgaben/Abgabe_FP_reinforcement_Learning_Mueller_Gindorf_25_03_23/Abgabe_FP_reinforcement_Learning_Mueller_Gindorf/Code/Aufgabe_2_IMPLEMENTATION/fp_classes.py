import numpy as np
from random import choice

class environment:
	 def __init__(self):
		 self.N_states = 100
		 self.target_position = 8
		 self.starting_position = 30
		 
		 self.obstacle_interval = np.arange(9,12)
		 self.P_obstacle = 0.0
	
class agent:
	def __init__(self,env_):
		self.N_episodes = int(1e4)
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
			#print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
			print(f"self.P_diffstep = {self.P_diffstep} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
			exit()
	
	def random_step(self):
		self.x_old = self.x
		if np.random.rand() > self.P_diffstep:
			return
		if np.random.rand() > 0.5:
			self.x += 1
		else:
			self.x -= 1
		
	#def adjust_epsilon(self,episode):
	def adjust_epsilon(self):
		self.epsilon -= 1./(self.zero_fraction*self.N_episodes)
		if self.epsilon < 0:
			self.epsilon = 0
		    
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
		if np.random.rand() < self.epsilon:
			choices = np.array((0,1,2))
		else:
			choices = np.atleast_1d(np.argmax(self.Q[self.x,:]))
		self.chosen_action = choice(choices)

	def perform_action(self,env_):
		"""
		Hier werden die Aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""
		self.x += (self.chosen_action - 1)
		self.x = self.x % env_.N_states
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		i = self.x_old
		i_prime = self.x
		j = self.chosen_action
		if i != env_.target_position or i_prime != env_.target_position:
			R = 0
		else:
			R = self.target_reward
		self.Q[i,j] += self.alpha*(R + self.gamma*np.amax(self.Q[i_prime,:]) - self.Q[i,j])
	
	def stoch_obstacle(self,env_):
		pass

