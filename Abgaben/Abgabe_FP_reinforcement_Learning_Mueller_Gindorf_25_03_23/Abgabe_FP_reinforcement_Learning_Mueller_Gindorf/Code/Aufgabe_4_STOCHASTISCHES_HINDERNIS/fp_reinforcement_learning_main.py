import fp_classes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import time as tm

from celluloid import Camera
from datetime import date
import datetime

now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")
names = "Gindorf_Mueller"

ps = np.linspace(0.0,1.0,11)
ks = np.zeros_like(ps)
for iteration in range(11):
	for p_index in range(len(ps)):
		env = fp_classes.environment()
		learner = fp_classes.agent(env)
		env.P_obstacle = ps[p_index]
		total_action_displacement = 0
		for episode in range(learner.N_episodes):
			learner.x = env.starting_position
			learner.chosen_action = None
			while (learner.x != env.target_position) or learner.chosen_action != 1:
				learner.adjust_epsilon(episode)
				learner.choose_action()
				learner.random_step()
				learner.stoch_obstacle(env)
				learner.perform_action(env)
				learner.update_Q(env)
				if learner.epsilon <= 0:
					total_action_displacement += (learner.chosen_action - 1)
		kappa = total_action_displacement/np.abs(total_action_displacement)
		ks[p_index] += kappa

plt.plot(ps, ks, "*")
plt.xlabel("P_obstacle")
plt.ylabel("Kappa")
plt.savefig(f"Kappa-over-p_0-to-1_high_alpha_{names}_{now}.png")
plt.show()
