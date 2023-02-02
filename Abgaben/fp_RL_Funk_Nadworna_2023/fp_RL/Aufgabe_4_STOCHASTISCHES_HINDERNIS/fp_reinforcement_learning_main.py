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

Kokosnusspalme = [np.linspace(0.8,1,5),np.zeros(5)]

for i in range(5):
	env = fp_classes.environment()
	learner = fp_classes.agent(env)

	env.P_obstacle = Kokosnusspalme[0][i]

	LAST_TRAJECTORY = []
	#Q_VALUES_OVER_TIME = []

	total_action_displacement =  0

	for episode in tqdm(range(learner.N_episodes)):
		
		learner.x = env.starting_position
		learner.chosen_action = 0

		while (learner.x != env.target_position) or learner.chosen_action != 1:
			if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)
			learner.adjust_epsilon(episode)
			learner.choose_action()
			learner.random_step()
			learner.stoch_obstacle(env)
			learner.perform_action(env)
			if(learner.epsilon == 0):
				total_action_displacement += learner.chosen_action - 1
			learner.update_Q(env)
			#Q_VALUES_OVER_TIME.append(learner.Q)

	kappa = total_action_displacement / np.abs(total_action_displacement)
	Kokosnusspalme[1][i] = kappa
	print("kappa: %1.1f" % kappa)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(Kokosnusspalme[0][0:5], Kokosnusspalme[1][0:5], color="orange")  #
plt.xlabel(r"Wahrscheinlichkeit $P_{obstacle}$", fontsize = 20)
plt.ylabel(r"Entscheidung $\kappa$", fontsize = 20)
plt.savefig("Stochastisches_Hindernis_kleines_alpha.pdf")
plt.show()