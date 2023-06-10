# A3
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

# Initialisierung der Objekte
env = fp_classes.environment()
learner = fp_classes.agent(env)

Q_val_over_time = []

for episode in range(learner.N_episodes):
	"""
	if ((episode%(learner.N_episodes//100)) == 0) and episode:
		progress = 1.*episode/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {100*progress} %")
	"""
	#learner.x = env.starting_position # Gesetzte Startposition
	learner.x = np.random.randint(env.N_states) # Zufällige Startposition
	learner.chosen_action = None

	learner.adjust_epsilon(episode)

	# Tatsächlich benötigte Aktionen. Werden in perform_action() erhöht.
	learner.actions = 0

	# Minimal benötigte Schritte 
	learner.actions_min = np.abs(learner.x - env.target_position)
	if learner.actions_min > (.5*env.N_states): # Berücksichtigung der Periodizität
		learner.actions_min = env.N_states-learner.actions_min
	
	# Minimal benötigte Aktionen (einmal Verweilen ohne Zustandänderung)
	learner.actions_min += 1


	while (learner.x != env.target_position) or (learner.chosen_action != 1):

		# Aufruf der Funktionen zur "Bewegung" des Agenten
		learner.choose_action()
		learner.random_step()
		learner.perform_action(env)
		learner.update_Q(env)
		Q_val_over_time.append(learner.Q[learner.output_state])

	# Abspeichern der Episode und Schritte
	learner.actions_per_episode.append([episode+1,learner.actions,learner.actions_min])
		
"""
# Ausgabe der Entwicklung einer Zeile der Q-Matrix als Textdatei
f = open("Q_DEVEL_" + now + ".txt","w")
for item in Q_val_over_time:
	print(item, file=f)
f.close()
"""


# Plot der learning curve
#plt.title('Number of actions relative to minimal actions possible:')
plt.xlabel('Episode')
plt.ylabel('#Aktionen / #Aktionen_min')
plt.rcParams.update({'font.size': 18})
plt.grid(True, which="both")
plt.semilogy(
	[item[0] for item in learner.actions_per_episode],
	[(item[1]/item[2]) for item in learner.actions_per_episode]
)

filename=f'Aufgabe_3_HYPERPARAMETER_100Episodes_AccuracyPerEpisode_RandomStart_{learner.alpha}_KocksKlett_'+now+'.png'

plt.savefig(filename, bbox_inches ="tight")