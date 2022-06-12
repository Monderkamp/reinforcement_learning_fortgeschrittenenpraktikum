import fp_classes
import numpy as np
import matplotlib.pyplot as plt
import time as tm

from datetime import date
import datetime

now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")

env = fp_classes.environment()
learner = fp_classes.agent(env)


#AUFGABE: HIERHIN KOMMT DER CODE ZUR UNTERSUCHUNG DES MSD

# ~ T_XPOW2_DATAPOINTS = dict()
# ~ T0 = tm.time()
# ~ for i in range(learner.N_episodes):
	# ~ #print(i)
	# ~ if ((i%(learner.N_episodes//100)) == 0) and i:
		# ~ progress = 1.*i/learner.N_episodes
		# ~ print("------------------------------------------")
		# ~ print(f"Fortschritt: {progress}")
		# ~ Trem = (tm.time() - T0)*(1./progress - 1.)
		# ~ print(f"Verbleibende Zeit: {Trem} s")
		# ~ #print(i)
	# ~ learner.x = 0

	# ~ for t in range(learner.tmax_MSD):
		# ~ if not t in T_XPOW2_DATAPOINTS.keys():
			# ~ T_XPOW2_DATAPOINTS[t] = []
		# ~ T_XPOW2_DATAPOINTS[t].append(learner.x**2)
		
		# ~ #AUFGABE: random_step here	
		# ~ learner.random_step()

# ~ #exit()

# ~ for key in T_XPOW2_DATAPOINTS.keys():
	# ~ T_XPOW2_DATAPOINTS[key] =  np.mean(T_XPOW2_DATAPOINTS[key])

# ~ X = np.array(list(T_XPOW2_DATAPOINTS.keys()))
# ~ Y = np.array(list(T_XPOW2_DATAPOINTS.values()))
# ~ plt.rcParams.update({'font.size': 18})

# ~ fig,ax = plt.subplots()

# ~ ax.plot(X,Y,label="Simulationsdaten")
# ~ p = np.polyfit(X,Y,1)
# ~ print(p)
# ~ ax.plot(X,p[0]*X+p[1],linestyle="dashed",label="fit")
# ~ ax.legend()
# ~ ax.set_xlabel(r"$t$")
# ~ ax.set_ylabel(r"$\left <x^2(t) \right >$")
# ~ plt.show()

#exit()


LEARNING_CURVE = []
Q_VALUES_OVER_TIME = []

total_action_displacement = 0.0


for episode in range(learner.N_episodes):
	if ((episode%(learner.N_episodes//100)) == 0) and episode:
		progress = 1.*episode/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {progress}")

	# ~ learner.x = np.random.randint(env.N_states)
	learner.x = env.starting_position
	learner.chosen_action = None
	stepcounter = 0
	total_dis = env.target_position -  learner.x

	if total_dis < -0.5*env.N_states: total_dis += env.N_states
	elif total_dis >= 0.5*env.N_states: total_dis -= env.N_states
	total_dis = np.abs(total_dis) + 1
	
	while (learner.x != env.target_position) or learner.chosen_action != 1:
		learner.adjust_epsilon(episode)
		learner.choose_action()
		
		learner.x_old = learner.x
		
		learner.perform_action(env)
		if learner.epsilon == 0:
			total_action_displacement += (learner.chosen_action-1)

		
		learner.random_step()
		learner.stoch_obstacle(env)
		learner.update_Q(env)
		Q_VALUES_OVER_TIME.append(learner.Q[learner.output_state])
		stepcounter += 1
	LEARNING_CURVE.append([episode,(stepcounter)/total_dis])
print(f"total_action_displacement/total_action_counter = {total_action_displacement/np.abs(total_action_displacement)}")


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
ax.loglog(Q_VALUES_OVER_TIME[:,0],label=f"Q_{learner.output_state},0")
ax.loglog(Q_VALUES_OVER_TIME[:,1],label=f"Q_{learner.output_state},1")
ax.loglog(Q_VALUES_OVER_TIME[:,2],label=f"Q_{learner.output_state},2")
ax.set_xlabel(r"episode")
fig.savefig("Q_over_time_" + now + ".png",bbox_inches="tight")
plt.close(fig)
