# A4
# MODULE
import fp_classes
import numpy as np
import matplotlib.pyplot as plt
import time as tm
import datetime
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from celluloid import Camera
from datetime import date

# Aktuelle Uhrzeit und Datum im ISO-Format
now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")


kappas = []

for i in range(0,3):

	# Initialisierung der Objekte
	env = fp_classes.environment()
	learner = fp_classes.agent(env)
	
	env.P_obstacle = .6+i*0.1
	print("Durchlauf für p=",env.P_obstacle)
	total_action_displacement = 0

	for episode in range(learner.N_episodes):
		"""
		if ((episode%(learner.N_episodes//100)) == 0) and episode:
			progress = 1.*episode/learner.N_episodes
			print("------------------------------------------")
			print(f"Fortschritt: {100*progress} %")
		"""

		learner.x = env.starting_position

		learner.chosen_action = None

		learner.adjust_epsilon(episode)
		
		while (learner.x != env.target_position) or (learner.chosen_action != 1):
			# Aufruf der Funktionen zur "Bewegung" des Agenten
			learner.choose_action()
			learner.random_step()
			learner.stoch_obstacle(env)
			learner.perform_action(env)
			if learner.epsilon == 0:
				total_action_displacement += (learner.chosen_action-1)

			learner.update_Q(env)

	kappa = total_action_displacement / np.abs(total_action_displacement)
	print(kappa)

	kappas.append([env.P_obstacle,kappa])


x = [item[0] for item in kappas]
y = [item[1] for item in kappas]


plt.plot(x,y, 'ro-')
plt.xlabel('P_Obstacle')
plt.ylabel('kappa')
plt.grid(True, which="both")
plt.rcParams.update({'font.size': 18})

filename=f'Aufgabe_4_STOCHASTISCHES_HINDERNIS_HighAlpha_KocksKlett_'+now+'.png' 
#plt.savefig(filename, bbox_inches ="tight")
#plt.show()