import numpy as np
class environment:
	 def __init__(self):
		 self.N_states = 100
		 self.target_position = 50
	
class agent:
	def __init__(self,env_):
		self.N_episodes = 1000
		self.x = 5
		self.Q = np.zeros((env_.N_states,3))
		self.alpha = 0.001
		self.gamma = 0.5
		self.epsilon = 1.0
		self.target_reward = 10.0
	def adjust_epsilon(self,episode):
		self.epsilon = np.maximum(0.0,1.0 - (episode/self.N_episodes)/0.9)
		    
	def choose_action(self):
		if (np.random.rand() < self.epsilon) or (np.sum(A[2] == np.max(A[2])) != 1):  #wählt eine Zufallsaktion aus mit 
																					#Wahrscheinlichkeit self.epsilon oder falls zwei aktionen die hoechsten Q-werte haben
			self.chosen_action = np.randint(3)
		else:	
			self.chosen_action = np.argmax(Q[x])
			
	def perform_action(self,env_):
		x += self.chosen_action-1 #aktion 0: geh nach links, aktion 1: bleib, aktion 2: geh nach rechts auf der x-achse		
		x = x%env_.N_states
	
	def update_Q(self,env_):
		#self.Q[self.x-(self.chosen_action-1),self.chosen_action] += self.alpha*(R[self.x,2] + self.gamma*max(self.Q[self.x+1,:]) - self.Q[self.x,2])
		self.Q[self.x-(self.chosen_action-1),self.chosen_action] *= (1.0-self.alpha)
		self.Q[self.x-(self.chosen_action-1),self.chosen_action] += self.alpha*self.target_reward*(self.x == env_.target_position)
		self.Q[self.x-(self.chosen_action-1),self.chosen_action] += self.alpha*self.gamma*np.max(Q[self.x])

seed = np.random.randint(1e4)
np.random.seed(seed)
print(f'seed = {seed}')

env = environment()
learner = agent(env)


if __name__ == "__main__":

	for episode in range(learner.N_episodes):
	#print(f'n = {n}')
	#print(f'agent.gamma = {agent.gamma}')
		learner.adjust_epsilon(episode)
		print(learner.epsilon)
	# ~ agent0.x = np.random.randint(axissize)
	# ~ goal = 0
	# ~ #print(f'start = {agent0.x}.') 
	# ~ stepcounter = 0
	# ~ p_expl = 1-(n/(N_episodes-1))
	# ~ #print(p_expl)
	# ~ while (goal == 0):
		# ~ stepcounter+=1
		# ~ randnr = np.random.rand()
		# ~ what_to_do=0
		
		# ~ if (randnr < p_expl):
			# ~ what_to_do = np.random.randint(3)
			# ~ #print('expl')
		# ~ else:
			# ~ A = maxidx(agent0.Q[agent0.x,:])
			# ~ if (A[1] ==0):
				# ~ what_to_do = A[0]
			# ~ else:
				# ~ what_to_do = np.random.randint(3)
			# ~ #print('det')
			
		# ~ if (what_to_do == 0): 
			# ~ agent0.go_left()
		# ~ if (what_to_do == 1):
			# ~ agent0.stay()
		# ~ if (what_to_do == 2):
			# ~ agent0.go_right()
			
		# ~ if (agent0.x == target):
			# ~ goal = 1
			#print('goal reached')
		#elif (stepcounter > steplimit):
			#break
		#print(f'goal = {goal}')    
		#print(f'agent.Q = {agent.Q}')
	#print(f'taken {stepcounter} steps.')        
    


#print(X)
#print(f'agent0.Q = {agent0.Q}')

"""
for q in agent0.Q:
    print(q)
"""    
#print(f'agent0.R = {agent0.R}')

total_deviation = 0
total_steps = 0

for n in range(N_episodes):
    #print('\n')
    #print(f'agent.gamma = {agent.gamma}')

    agent0.x = np.random.randint(axissize)
    deviation = abs(agent0.x-target)
    #print(f'deviation={deviation}')
    goal = 0
    #print(f'start = {agent0.x}.') 
    stepcounter = 0
    if (deviation==0):
        goal=1
    while (goal == 0):
        stepcounter+=1
        #what_to_do = np.random.randint(3)
        #what_to_do = np.argmax(agent0.Q[agent0.x,:])
        A = maxidx(agent0.Q[agent0.x,:])
        if (A[1] ==0):
            what_to_do = A[0]
        else:
            what_to_do = np.random.randint(3)
        
        
        if (what_to_do == 0):
            agent0.go_left()
        if (what_to_do == 1):
            agent0.stay()
        if (what_to_do == 2):
            agent0.go_right()
            
        if (agent0.x == target):
            goal = 1
            total_steps += stepcounter
            total_deviation += deviation
            #print(f'stepcounter={stepcounter}')
            #print(f'deviation={deviation}')
            if (stepcounter != stepcounter):
                print(f'(stepcounter != stepcounter)')
                print(f'stepcounter={stepcounter}')
                print(f'deviation={deviation}')
                
                
            #print('goal reached')
        #print(f'goal = {goal}')    
        #print(f'agent.Q = {agent.Q}')
    #print(f'taken {stepcounter} steps.')  

"""
for q in agent0.Q:
    print(q)
"""
print(f'total_steps={total_steps}')
print(f'total_deviation={total_deviation}')
    
print(float(total_steps)/total_deviation)

