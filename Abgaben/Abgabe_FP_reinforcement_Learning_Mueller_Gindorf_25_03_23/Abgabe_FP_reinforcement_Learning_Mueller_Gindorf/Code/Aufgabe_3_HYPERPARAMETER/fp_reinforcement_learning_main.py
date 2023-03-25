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
env = fp_classes.environment()
learner = fp_classes.agent(env)

def minimal_steps_needed(target, start, N_states):
	if target <= start:
		return min(start - target, N_states - start + target) + 1
	else:
		return min(target - start, start + N_states - target) + 1

steps = []
for episode in range(learner.N_episodes):
	if ((episode%(learner.N_episodes//100)) == 0) and episode:
		progress = 1.*episode/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {progress}")
	starting_position = np.random.randint(env.N_states)
	env.starting_position = starting_position
	learner.x = env.starting_position
	learner.chosen_action = None
	needed_steps = 0
	minimal_steps = minimal_steps_needed(env.target_position, env.starting_position, env.N_states)
	while (learner.x != env.target_position) or learner.chosen_action != 1:
		learner.adjust_epsilon(episode)
		learner.choose_action()
		learner.random_step()
		learner.perform_action(env)
		learner.update_Q(env)
		needed_steps += 1
	steps.append(1.*needed_steps/minimal_steps)


plt.semilogy(steps)
plt.xlabel("Episode")
plt.ylabel("Benötigte Schritte/Minimale Schrittzahl")
plt.savefig(f"Learning-Curve_N={learner.N_episodes}_alpha={learner.alpha}_{names}_{now}.png")
plt.show()
