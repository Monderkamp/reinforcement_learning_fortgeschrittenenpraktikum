import numpy as np

def maxidx(A):
    idx_max = 0
    val_max = -1000
    equal=0
    for idx,a in enumerate(A):
        #print(f'a={a}')
        #print(f'idx={idx}')
        if (a>val_max):
            val_max=a
            idx_max=idx
        elif (a==val_max):
            equal=1
            break
    
    
    return (idx_max,equal)

#asd = maxidx([1,2,2])
#print(asd)
#exit()    

class agent:
    def __init__(self,x0,XMIN,XMAX,GAMMA,R,LEARNING_RATE):
        self.x = int(x0)
        self.xmin = int(XMIN)
        self.xmax = int(XMAX)
        self.R = R
        self.Q = np.zeros(R.shape)
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
    def go_right(self):
        if (self.x<self.xmax):
            self.Q[self.x,2] = self.Q[self.x,2] + self.learning_rate*(R[self.x,2] + self.gamma*max(self.Q[self.x+1,:]) - self.Q[self.x,2])
            self.x+=1
        else:
            self.stay()
            
    def go_left(self):
        if (self.x>self.xmin):
            self.Q[self.x,0] = self.Q[self.x,0] + self.learning_rate*(R[self.x,0] + self.gamma*max(self.Q[self.x-1,:]) - self.Q[self.x,0])
            self.x-=1
        else:
            self.stay()            
            
    def stay(self):
        self.Q[self.x,1] = self.Q[self.x,1] + self.learning_rate*(R[self.x,1] + self.gamma*max(self.Q[self.x,:]) - self.Q[self.x,1])
        
    
    x = 0
    xmin = 0
    xmax = 1
    Q = []
    R = []
    gamma = 0.8
    learning_rate = 0.5


#exit()


gamma = 0.99
axissize = 1000
target=int(0.5*axissize)
N_actions=3  #go left; stay; go right
N_episodes = int(3*axissize)
seed = np.random.randint(1e4)
learning_rate = 0.98

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

agent0 = agent(np.random.randint(axissize),0,axissize-1,gamma,R,learning_rate)

#n=0

for n in range(N_episodes):
    #print(f'n = {n}')
    #print(f'agent.gamma = {agent.gamma}')

    agent0.x = np.random.randint(axissize)
    goal = 0
    #print(f'start = {agent0.x}.') 
    stepcounter = 0
    p_expl = 1-(n/(N_episodes-1))
    #print(p_expl)
    while (goal == 0):
        stepcounter+=1
        randnr = np.random.rand()
        what_to_do=0
        
        if (randnr < p_expl):
            what_to_do = np.random.randint(3)
            #print('expl')
        else:
            A = maxidx(agent0.Q[agent0.x,:])
            if (A[1] ==0):
                what_to_do = A[0]
            else:
                what_to_do = np.random.randint(3)
            #print('det')
            
        if (what_to_do == 0): 
            agent0.go_left()
        if (what_to_do == 1):
            agent0.stay()
        if (what_to_do == 2):
            agent0.go_right()
            
        if (agent0.x == target):
            goal = 1
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

