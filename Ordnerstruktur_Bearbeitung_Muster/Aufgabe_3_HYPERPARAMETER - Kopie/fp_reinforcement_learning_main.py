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


#AUFGABE: HIERHIN KOMMT DER CODE ZUR UNTERSUCHUNG DES MSD

T_XPOW2_DATAPOINTS = dict()
T0 = tm.time()
for i in range(learner.N_episodes):
	#print(i)
	if ((i%(learner.N_episodes//100)) == 0) and i:
		progress = 1.*i/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {progress}")
		Trem = (tm.time() - T0)*(1./progress - 1.)
		print(f"Verbleibende Zeit: {Trem} s")
		#print(i)
	learner.x = 0

	for t in range(learner.tmax_MSD):
		if not t in T_XPOW2_DATAPOINTS.keys():
			T_XPOW2_DATAPOINTS[t] = []
		T_XPOW2_DATAPOINTS[t].append(learner.x**2)
		
		#AUFGABE: random_step here	
		learner.random_step()

#exit()

for key in T_XPOW2_DATAPOINTS.keys():
	T_XPOW2_DATAPOINTS[key] =  np.mean(T_XPOW2_DATAPOINTS[key])

X = np.array(list(T_XPOW2_DATAPOINTS.keys()))
Y = np.array(list(T_XPOW2_DATAPOINTS.values()))
plt.rcParams.update({'font.size': 18})

fig,ax = plt.subplots()

ax.plot(X,Y,label="Simulationsdaten")
p = np.polyfit(X,Y,1)
print(p)
ax.plot(X,p[0]*X+p[1],linestyle="dashed",label="fit")
ax.legend()
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\left <x^2(t) \right >$")
plt.show()

#exit()
LAST_TRAJECTORY = []


LEARNING_CURVE = []
Q_VALUES_OVER_TIME = []

for episode in range(learner.N_episodes):
	if ((episode%(learner.N_episodes//100)) == 0) and episode:
		progress = 1.*episode/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {progress}")

	learner.x = np.random.randint(env.N_states)
	learner.chosen_action = None
	stepcounter = 0
	total_dis = env.target_position -  learner.x

	if total_dis < -0.5*env.N_states: total_dis += env.N_states
	elif total_dis >= 0.5*env.N_states: total_dis -= env.N_states
	total_dis = np.abs(total_dis) + 1
	while (learner.x != env.target_position) or learner.chosen_action != 1:
		if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)
			
		learner.adjust_epsilon(episode)
		learner.choose_action()
		
		learner.perform_action(env)
		learner.random_step(env)
		learner.update_Q(env)
		Q_VALUES_OVER_TIME.append(learner.Q[learner.output_state])
		stepcounter += 1
	LEARNING_CURVE.append([episode,(stepcounter)/total_dis])

f = open("Q_MATRIX_" + now + ".txt","w")
f.write(str(learner.Q))
f.close()

LEARNING_CURVE = np.array(LEARNING_CURVE)
fig,ax = plt.subplots()
ax.semilogy(LEARNING_CURVE[:,0],LEARNING_CURVE[:,1])
ax.set_xlabel(r"episode")
ax.set_ylabel(r"T")
fig.savefig("learning_curve_" + now + ".png",bbox_inches="tight")
plt.close(fig)

fig,ax = plt.subplots()
Q_VALUES_OVER_TIME = np.array(Q_VALUES_OVER_TIME)
# ~ for a in Q_VALUES_OVER_TIME:
	# ~ print(a)
ax.loglog(Q_VALUES_OVER_TIME[:,0],label=f"Q_{learner.output_state},0")
ax.loglog(Q_VALUES_OVER_TIME[:,1],label=f"Q_{learner.output_state},1")
ax.loglog(Q_VALUES_OVER_TIME[:,2],label=f"Q_{learner.output_state},2")
ax.set_xlabel(r"episode")
fig.savefig("Q_over_time_" + now + ".png",bbox_inches="tight")
plt.close(fig)


im_agent = plt.imread("../IMG/Lasse_head.png")
im_treat = plt.imread("../IMG/treat.png")
imagebox = OffsetImage(im_agent, zoom=0.02)
imagebox2 = OffsetImage(im_treat, zoom=0.5)

trajfig,trajax = plt.subplots()
camera = Camera(trajfig)


LAST_TRAJECTORY_WITHOUT_PBJUMPS = []
for i in range(len(LAST_TRAJECTORY)-1):
	dx = np.abs(LAST_TRAJECTORY[i+1] - LAST_TRAJECTORY[i])
	LAST_TRAJECTORY_WITHOUT_PBJUMPS.append(LAST_TRAJECTORY[i])
	if dx > 0.5*env.N_states:
		LAST_TRAJECTORY_WITHOUT_PBJUMPS.append(None)
		

for i,x in enumerate(LAST_TRAJECTORY_WITHOUT_PBJUMPS):
	trajax.plot(np.array(LAST_TRAJECTORY_WITHOUT_PBJUMPS)[:i],np.linspace(0,0,len(np.array(LAST_TRAJECTORY_WITHOUT_PBJUMPS)[:i])),color="red")	
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

