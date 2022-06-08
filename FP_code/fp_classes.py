import numpy as np

class environment:
	 def __init__(self):
		 self.N_states = 100
		 self.target_position = 8
		 self.starting_position = 2
	
class agent:
	def __init__(self,env_):
		self.N_episodes = 10000
		self.x = 1
		self.Q = np.zeros((env_.N_states,3))
		self.alpha = 0.01
		self.gamma = 0.9
		self.epsilon = 1.0
		self.target_reward = 10.0
		
	def adjust_epsilon(self,episode):
		self.epsilon = np.maximum(0.0,1.0 - (episode/self.N_episodes))
		    
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die höchsten Q-werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt
		"""
		if (np.random.rand() < self.epsilon) or (np.sum(self.Q[self.x] == np.max(self.Q[self.x])) != 1):  
			self.chosen_action = np.random.randint(3)
		else:	
			self.chosen_action = np.argmax(self.Q[self.x])
			
	def perform_action(self,env_):
		"""
		Hier werden die aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

		self.x += self.chosen_action-1 #aktion 0: geh nach links, aktion 1: bleib, aktion 2: geh nach rechts auf der x-achse		
		self.x = self.x%env_.N_states
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		self.x_old = (self.x-(self.chosen_action-1))%env_.N_states

		self.Q[self.x_old,self.chosen_action] *= (1.0-self.alpha)
		self.Q[self.x_old,self.chosen_action] += self.alpha*self.target_reward*(self.x == env_.target_position)
		self.Q[self.x_old,self.chosen_action] += self.alpha*self.gamma*np.max(self.Q[self.x])



