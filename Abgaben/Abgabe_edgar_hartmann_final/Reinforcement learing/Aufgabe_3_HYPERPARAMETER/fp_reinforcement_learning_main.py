import fp_classes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import time as tm

from celluloid import Camera
from datetime import date
import datetime

now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")

env = fp_classes.environment()
learner = fp_classes.agent(env)

T_XPOW2_DATAPOINTS = dict()
T0 = tm.time()


LAST_TRAJECTORY = []
L_3_3_2_3 = []
Q_VALUES_OVER_TIME = np.zeros((learner.N_episodes * learner.tmax_MSD,3))
c = 0
print(Q_VALUES_OVER_TIME.shape)

for episode in range(learner.N_episodes):
	if ((episode%(learner.N_episodes//100)) == 0) and episode:
		progress = 1.*episode/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {progress}")
	count_steps = 0

	learner.x = np.random.randint(0, env.N_states)
	learner.chosen_action = None
	learner.x_old = learner.x
	learner.x_anf = learner.x
	#print(learner.test)
	
	min_actions = abs(learner.x - env.target_position)

	if min_actions > env.N_states/2:
		min_actions = env.N_states - min_actions
	
	if min_actions == 0:	#um division durch 0 zu verhindern
		min_actions = 1
	

	for t in range(learner.tmax_MSD):

		learner.x_old = learner.x

		learner.adjust_epsilon(episode)
		learner.choose_action()
		learner.perform_action(env)
		learner.random_step(env)
		learner.update_Q(env)

		Q_VALUES_OVER_TIME[c,:] = learner.Q[learner.output_state,:]

		c += 1
		
		if episode == learner.N_episodes-1: 
			LAST_TRAJECTORY.append(learner.x)
		
		if learner.x == env.target_position and learner.choosen_action == 1:
			L_3_3_2_3.append([count_steps/min_actions, episode])
			break
		
		if t == learner.tmax_MSD - 1:
			L_3_3_2_3.append([count_steps/min_actions, episode])
		
		count_steps += 1


Q_VALUES_OVER_TIME = Q_VALUES_OVER_TIME[:c,:]

ev = np.arange(start=0, stop=Q_VALUES_OVER_TIME.shape[0])
print(Q_VALUES_OVER_TIME.shape)

plt.figure();plt.title('evolution of Q-Values: state: %i' %learner.output_state)
plt.plot(ev, Q_VALUES_OVER_TIME[:,0])
plt.plot(ev, Q_VALUES_OVER_TIME[:,1])
plt.plot(ev, Q_VALUES_OVER_TIME[:,2])
plt.xlabel('Update step');plt.ylabel('Q-Value')
plt.legend(['left', 'stay', 'right'])
#plt.savefig('evolution of Q-Values '+ now +'.png')

#f = open('Q MATRIX with alpha = 0.01 and N_episodes = 1000 '+ now + '.txt','w')
#f.write(str(learner.Q))
#f.close()

L_3_3_2_3 = np.array(L_3_3_2_3)
print(L_3_3_2_3.shape)
print(L_3_3_2_3)


plt.figure(figsize=(16,9));plt.title('learning curve',fontsize=23)
plt.semilogy(L_3_3_2_3[:,1], L_3_3_2_3[:,0])
plt.ylabel('Anzahl Schritte/min. mög. Schritte',fontsize=17);plt.xlabel('Episode',fontsize=17)
#plt.savefig('learning curve min_actions N=1000'+ now + '.png')


plt.show()

exit()


im_agent = plt.imread("../IMG/Lasse_head.png")
im_treat = plt.imread("../IMG/treat.png")
imagebox = OffsetImage(im_agent, zoom=0.02)
imagebox2 = OffsetImage(im_treat, zoom=0.5)

trajfig,trajax = plt.subplots()
camera = Camera(trajfig)

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
#animation.save("LAST_TRAJECTORY_with alpha = 0.01 and N_episodes = 1000 " + now + ".gif")

