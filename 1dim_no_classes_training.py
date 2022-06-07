import numpy as np

gamma = 0.8
axissize = 100
target=int(0.75*axissize)
N_actions=3  #go left; stay; go right
N_episodes = int(500)
seed = np.random.randint(1e4)
learning_rate = 0.5

np.random.seed(seed)

print(f'gamma = {gamma}')
print(f'axissize = {axissize}')
print(f'target = {target}')
print(f'seed = {seed}')


X =  np.array(range(axissize))
Q = np.zeros((axissize,N_actions))
R = np.zeros((axissize,N_actions))

R[target-1,2] = 100
R[target,1] = 100
R[target+1,0] = 100
R[0,0] = -100
R[axissize-1,2] = -100

#print(R)
goal = 0

#print(R.shape)


#print(max(R[target,:]))


#print(f'agent.x = {agent.x}')
#print(f'agent.Q = {agent.Q}')
#print(f'agent0.Q.shape = {agent0.Q.shape}')


for n in range(N_episodes):
    
    #print('\n')
    #print(f'agent.gamma = {agent.gamma}')
    x_agent = np.random.randint(axissize)
    goal = 0
    #print(f'start = {x_agent}.') 
    stepcounter = 0
    while (goal == 0):
        stepcounter+=1
        what_to_do = np.random.randint(3)
        if ((x_agent==0) and (what_to_do==0)):
            what_to_do = 1
        if ((x_agent==axissize-1) and (what_to_do==2)):
            what_to_do = 1    
            
        if (what_to_do == 0):
            Q[x_agent,0] += learning_rate*(R[x_agent,0] + gamma*max(Q[x_agent-1,:]) - Q[x_agent,0])
            x_agent-=1
        if (what_to_do == 1):
            Q[x_agent,1] += learning_rate*(R[x_agent,1] + gamma*max(Q[x_agent,:]) - Q[x_agent,1])
        if (what_to_do == 2):
            Q[x_agent,2] += learning_rate*(R[x_agent,2] + gamma*max(Q[x_agent+1,:]) - Q[x_agent,2])
            x_agent+=1
            
            
        if (x_agent == target):
            goal = 1
            #print('goal reached')
        #print(f'goal = {goal}')    
        #print(f'agent.Q = {agent.Q}')
    #print(f'taken {stepcounter} steps.')        
    
print(Q)
"""
#print(X)
print(f'agent0.Q = {agent.Q}')
print(f'agent0.R = {agent.R}')

print('\n')
print(target)

agent1 = agent(49,0,axissize-1,gamma,R,learning_rate)
print(agent1.R)
agent1.go_right()
print(agent1.Q)


print('\n')
D = np.zeros((3,3))
D[1,1] = 1
print(D)

agent1.Q[target-1,2] = 14
print(agent1.Q)
"""