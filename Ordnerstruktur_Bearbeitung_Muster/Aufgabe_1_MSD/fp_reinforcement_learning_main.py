import fp_classes
import numpy as np
import matplotlib.pyplot as plt
import time as tm

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


# ~ for episode in range(learner.N_episodes):
	# ~ #print(episode)
	# ~ learner.x = env.starting_position
	# ~ while (learner.x != env.target_position) or learner.chosen_action != 1:
		
		# ~ learner.adjust_epsilon(episode)
		# ~ learner.choose_action()
		# ~ learner.perform_action(env)
		# ~ learner.update_Q(env)


# ~ print(learner.Q)



