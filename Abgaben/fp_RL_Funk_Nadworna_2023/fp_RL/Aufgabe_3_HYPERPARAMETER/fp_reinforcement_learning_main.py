import fp_classes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import time as tm

from celluloid import Camera
from datetime import date
import datetime

from tqdm import tqdm

now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")

env = fp_classes.environment()
learner = fp_classes.agent(env)

"""T_XPOW2_DATAPOINTS = dict()
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
		
		learner.random_step()	




for key in T_XPOW2_DATAPOINTS.keys():
	T_XPOW2_DATAPOINTS[key] =  np.mean(T_XPOW2_DATAPOINTS[key])

X = np.array(list(T_XPOW2_DATAPOINTS.keys()))
Y = np.array(list(T_XPOW2_DATAPOINTS.values()))
plt.rcParams.update({'font.size': 18})

fig,ax = plt.subplots()

ax.plot(X,Y,label="Simulationsdaten")
p = np.polyfit(X,Y,1)
D1 = p[0] * 0.5 #gemessenes D
print(p)
ax.plot(X,p[0]*X+p[1],linestyle="dashed",label="fit")
ax.legend()
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\\left <x^2(t) \right >$")
plt.show()
fig.savefig(f"Aufgabe1_MSD_{learner.D}_{D1}_{now}.png",bbox_inches="tight")"""


LAST_TRAJECTORY = []
#Q_VALUES_OVER_TIME = []

ep_step = np.zeros((learner.N_episodes, 2))
min_count = np.zeros(learner.N_episodes)

for episode in tqdm(range(learner.N_episodes)):
	
	count = 0
	learner.x = np.random.randint(0,env.N_states)
	learner.chosen_action = None

	diff = np.abs(env.target_position - learner.x) + 1
	if(diff > env.N_states/2):
		diff = env.N_states - diff
		min_count[episode] = diff
	else:
		min_count[episode] = diff

	while (learner.x != env.target_position) or learner.chosen_action != 1:
		if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)
		learner.adjust_epsilon(episode)
		learner.choose_action()
		learner.random_step()
		learner.perform_action(env)
		learner.update_Q(env)
		#Q_VALUES_OVER_TIME.append(learner.Q)
		count+=1
	ep_step[episode][0] = episode
	ep_step[episode][1] = count


plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 
plt.figure(figsize=(8, 6), dpi=80)
plt.semilogy(ep_step[:,0], ep_step[:,1]/min_count, color="purple", label="learning curve")  #
plt.xlabel(r"Episoden", fontsize = 20)
plt.ylabel(r"Tatsächliche Schritte / Minimale Schritte", fontsize = 17)
plt.legend(fontsize = 17, frameon = False)
plt.savefig("learning_curve_comparison_min_count_alpha0999999.pdf")
plt.show()


"""im_agent = plt.imread("../IMG/Lasse_head.png")
im_treat = plt.imread("../IMG/treat.png")
imagebox = OffsetImage(im_agent, zoom=0.02)
imagebox2 = OffsetImage(im_treat, zoom=0.5)

trajfig,trajax = plt.subplots()
camera = Camera(trajfig)

f = open("Q_MATRIX_" + now + ".txt","w")
f.write(str(learner.Q))
f.close()"""

#--------------------------------------------------------------------
#IM FOLGENDEN CODE BLOCK WIRD DIE TRAJEKTORIE ALS GIF AUSGEGEBEN
#WENN DIESER TEIL ZU LANGE DAUERT ODER SIE DIE NÖTIGEN LIBRARIES NICHT INSTALLIEREN KÖNNEN,
#KOMMENTIEREN SIE DIESEN CODE BLOCK AUS
#--------------------------------------------------------------------


"""LAST_TRAJECTORY_WITHOUT_PBJUMPS = []
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
animation.save("LAST_TRAJECTORY_" + now + ".gif")"""
