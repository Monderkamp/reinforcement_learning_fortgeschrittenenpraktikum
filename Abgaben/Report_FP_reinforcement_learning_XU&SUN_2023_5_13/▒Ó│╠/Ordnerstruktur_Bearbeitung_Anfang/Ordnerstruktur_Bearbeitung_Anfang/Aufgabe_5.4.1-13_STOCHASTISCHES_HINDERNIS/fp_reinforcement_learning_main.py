import fp_classes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import time as tm

#from celluloid import Camera
from datetime import date
import datetime

now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")

env = fp_classes.environment()
learner = fp_classes.agent(env)

#T_XPOW2_DATAPOINTS = dict()
#T0 = tm.time()

LAST_TRAJECTORY = []
#Episode_Shiritte=dict()
P_obstacle_such=dict()
Accuracy=40
est_P_Max=0.85
est_P_Min=0.70
for i in range(Accuracy):
    learner.Q = np.zeros((env.N_states,3))
    P_obstacle_i=i*(est_P_Max-est_P_Min)*(1/Accuracy)+est_P_Min
    for episode in range(learner.N_episodes):
        #Episode_Shiritte[episode]=[]
        #Anzahl_der_Schritte=0
        if ((episode%(learner.N_episodes//100)) == 0) and episode:
            progress = 1.*episode/learner.N_episodes
            print("------------------------------------------")
            print(f"Fortschritt: {progress}")
        total_action_displacement = 0
        #learner.x=np.random.randint(0,env.N_states)
        learner.x = env.starting_position
        #Min_Schritte = min(abs(env.target_position - learner.x)+1,(env.N_states - learner.x + env.target_position+1))
        #print(env.N_states+1 - learner.x + env.target_position)
        #print( Min_Schritte)
        #learner.x = env.starting_position
        learner.chosen_action = None   
        while (learner.x != env.target_position) or learner.chosen_action != 1:
            if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)
            learner.adjust_epsilon(episode)
            Random_number1=np.random.rand()
            if Random_number1<learner.P_diffstep:
                learner.random_step()
            else:
                learner.choose_action()
                learner.stoch_obstacle(env, P_obstacle_i)
            learner.perform_action(env)
            if learner.epsilon == 0 :
                total_action_displacement += learner.chosen_action-1
            else:
                total_action_displacement= total_action_displacement
            learner.update_Q(env)
        print( P_obstacle_i)
        k=total_action_displacement/(np.abs(total_action_displacement))
        P_obstacle_such[i]=[k]
print(P_obstacle_such.values())
X_3 = list((est_P_Max-est_P_Min)*x/Accuracy+est_P_Min for x in P_obstacle_such.keys())
Y_3 = list(P_obstacle_such.values())

plt.plot(X_3, Y_3)

"""
        #print(learner.x)
        Anzahl_der_Schritte=Anzahl_der_Schritte+1 
    Episode_Shiritte[episode] = [Anzahl_der_Schritte]
    Episode_Shiritte[episode].append(Anzahl_der_Schritte / Min_Schritte )
#print(Episode_Shiritte)

X_1 = list(Episode_Shiritte.keys())
X_2 = list(Episode_Shiritte.keys())[int(0.95*learner.zero_fraction*learner.N_episodes):int(1.15*learner.zero_fraction*learner.N_episodes)]
Y_1 = np.array(list(Episode_Shiritte.values()))[:,0]
Y_2 = np.array(list(Episode_Shiritte.values()))[int(0.95*learner.zero_fraction*learner.N_episodes):int(1.15*learner.zero_fraction*learner.N_episodes),1]
#print(Y_2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)) 
fig.subplots_adjust(wspace=0.5)
ax1.semilogy(X_1, Y_1)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Anzahl_der_Schritte')
ax1.set_title('Schrittanzahl über Episode')

ax2.plot(X_2, Y_2)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Verhältnis')
ax2.set_title('Verhältnis über Episode(teile)')

plt.show()

im_agent = plt.imread("../IMG/Lasse_head.png")
im_treat = plt.imread("../IMG/treat.png")
imagebox = OffsetImage(im_agent, zoom=0.02)
imagebox2 = OffsetImage(im_treat, zoom=0.5)


trajfig,trajax = plt.subplots()
camera = Camera(trajfig)

f = open("Q MATRIX " + now + ".txt","w")
f.write(str(learner.Q))
f.close()

#--------------------------------------------------------------------
#IM FOLGENDEN CODE BLOCK WIRD DIE TRAJEKTORIE ALS GIF AUSGEGEBEN
#WENN DIESER TEIL ZU LANGE DAUERT ODER SIE DIE NÖTIGEN LIBRARIES NICHT INSTALLIEREN KÖNNEN,
#KOMMENTIEREN SIE DIESEN CODE BLOCK AUS
#--------------------------------------------------------------------


LAST_TRAJECTORY_WITHOUT_PBJUMPS = []
for i in range(len(LAST_TRAJECTORY)-1):
	dx = np.abs(LAST_TRAJECTORY[i+1] - LAST_TRAJECTORY[i])
	LAST_TRAJECTORY_WITHOUT_PBJUMPS.append(LAST_TRAJECTORY[i])
	if dx > 0.5*env.N_states:
		LAST_TRAJECTORY_WITHOUT_PBJUMPS.append(None)
		

for i,x in enumerate(LAST_TRAJECTORY_WITHOUT_PBJUMPS):
	trajax.plot(np.array(LAST_TRAJECTORY_WITHOUT_PBJUMPS)[:i],np.linspace(0,0,len(np.array(LAST_TRAJECTORY_WITHOUT_PBJUMPS)[:i])),color="blue")	
	ab = AnnotationBbox(imagebox2, (env.target_position, 0.0),frameon=False)
	trajax.add_artist(ab)
	if x:
		ab = AnnotationBbox(imagebox, (x, 0.0),frameon=False)
		#trajax.add_artist(ab)
	else:
		ab = AnnotationBbox(imagebox, (LAST_TRAJECTORY_WITHOUT_PBJUMPS[i+1], 0.0),frameon=False)
	trajax.add_artist(ab)
			

	trajax.set_xlim(0.0,env.N_states)
	trajax.set_ylim(-1.0,1.0)
	trajax.set_yticks([])
	camera.snap()
animation = camera.animate()		
animation.save("LAST_TRAJECTORY_" + now + ".gif")
"""
