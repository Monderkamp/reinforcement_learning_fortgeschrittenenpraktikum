import fp_classes
import numpy as np
import matplotlib.pyplot as plt
import time as tm

env = fp_classes.environment()
learner = fp_classes.agent(env)


#AUFGABE: HIERHIN KOMMT DER CODE ZUR UNTERSUCHUNG DES MSD

T_XPOW2_DATAPOINTS = []
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
		T_XPOW2_DATAPOINTS.append([t,learner.x**2])
		learner.random_step()



T_XPOW2_DATAPOINTS = np.array(T_XPOW2_DATAPOINTS)
fig,ax = plt.subplots()
ax.scatter(T_XPOW2_DATAPOINTS[:,0],T_XPOW2_DATAPOINTS[:,1],marker=".")

p = np.polyfit(T_XPOW2_DATAPOINTS[:,0],T_XPOW2_DATAPOINTS[:,1],1)
print(p)


plt.show()


# ~ for episode in range(learner.N_episodes):
	# ~ #print(episode)
	# ~ learner.x = env.starting_position
	# ~ while (learner.x != env.target_position) or learner.chosen_action != 1:
		
		# ~ learner.adjust_epsilon(episode)
		# ~ learner.choose_action()
		# ~ learner.perform_action(env)
		# ~ learner.update_Q(env)


# ~ print(learner.Q)
