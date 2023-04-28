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

# T_XPOW2_DATAPOINTS = dict()
# T0 = tm.time()
# for i in range(learner.N_episodes):
# 	#print(i)
# 	if ((i%(learner.N_episodes//100)) == 0) and i:
# 		progress = 1.*i/learner.N_episodes
# 		print("------------------------------------------")
# 		print(f"Fortschritt: {progress}")
# 		Trem = (tm.time() - T0)*(1./progress - 1.)
# 		print(f"Verbleibende Zeit: {Trem} s")
# 		#print(i)
# 	learner.x = 0

	# for t in range(learner.tmax_MSD):
	# 	if not t in T_XPOW2_DATAPOINTS.keys():
	# 		T_XPOW2_DATAPOINTS[t] = []
	# 	T_XPOW2_DATAPOINTS[t].append(learner.x**2)
	#
	# 	learner.random_step()
	#

# exit()

# for key in T_XPOW2_DATAPOINTS.keys():
# 	T_XPOW2_DATAPOINTS[key] =  np.mean(T_XPOW2_DATAPOINTS[key])
#
# X = np.array(list(T_XPOW2_DATAPOINTS.keys()))
# Y = np.array(list(T_XPOW2_DATAPOINTS.values()))
# plt.rcParams.update({'font.size': 18})
#
# fig,ax = plt.subplots()
#
# ax.plot(X,Y,label="Simulationsdaten")
# p = np.polyfit(X,Y,1)
# # print(p)
# #-------------------------------
# print('Steigung: ', p[0])
# print('y-intercept: ', p[1])
# D_measured = p[0]/2
# #-------------------------------
#
# ax.plot(X,p[0]*X+p[1],linestyle="dashed",label="fit")
# ax.legend()
# ax.set_xlabel(r"$t$")
# ax.set_ylabel(r"$\left <x^2(t) \right >$")
#
# #-------------------------------
# filename = 'Aufgabe_1_MSD_' + str(learner.D) + '_' + str(D_measured) +'_Maxim_Root_' + now + '.png'
# fig.savefig(filename,bbox_inches='tight')
# #-------------------------------
#
# plt.show()
#
# exit()
LAST_TRAJECTORY = []
counter = 0
COUNTER = []
Q_VALUES_OVER_TIME = []

for episode in range(learner.N_episodes):
	if ((episode%(learner.N_episodes//100)) == 0) and episode:
		progress = 1.*episode/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {progress}")

	learner.x = np.random.randint(0,env.N_states)
	learner.chosen_action = None

	if learner.x <= env.middle:
		minsteps = np.abs(learner.x - env.target_position)
	else:
		minsteps = env.N_states - learner.x + env.target_position

	while (learner.x != env.target_position) or learner.chosen_action != 1:
		if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)
		learner.adjust_epsilon(episode)
		learner.choose_action()
		learner.random_step(env)
		learner.perform_action(env)
		learner.update_Q(env)
		Q_VALUES_OVER_TIME.append(learner.Q[learner.output_state])
		counter += 1

	if minsteps > 0:
		COUNTER.append([episode,counter/minsteps]) #excludes episodes with zerodivision
	# print('Evolution of Q for state ', learner.output_state, ': ',Q_VALUES_OVER_TIME) #trust me, dont
	Q_VALUES_OVER_TIME = []
	counter = 0
# print(COUNTER)

EPISODES = np.array(COUNTER)[:,0]
STEPS = np.array(COUNTER)[:,1]
plt.rcParams['font.size'] = '16'

fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12, 8))

ax1.plot(EPISODES,STEPS)
ax1.set_xlabel('episodes')
ax1.set_ylabel('amount of steps / minimal steps')
ax1.set_title('normal scaling')

ax2.semilogy(EPISODES,STEPS)
ax2.set_xlabel('episodes')
ax2.set_title('y-axis scaled logarithmically')

#-------------------------------
# filename = 'Aufgabe_3_LEARNING_RATE_RANDOM_SCALED_alpha=0.999999_Maxim_Root_' + now + '.png'
# fig.savefig(filename,bbox_inches='tight')
#-------------------------------

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
animation.save("LAST_TRAJECTORY_" + now + ".gif")

f = open('Q_MATRIX_' + now + '.txt', 'w')
f.write(str(learner.Q))
f.close()
