# A2
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

LAST_TRAJECTORY = []

for episode in range(learner.N_episodes):
	# Fortschrittsanzeige des Lernalgorithmus
	if ((episode%(learner.N_episodes//100)) == 0) and episode:
		progress = 1.*episode/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {100*progress} %")

	learner.x = env.starting_position
	learner.chosen_action = None

	learner.adjust_epsilon(episode) # Aufrung im for-loop, da sich im while-loop die Episode nicht ändert.
	
	while (learner.x != env.target_position) or (learner.chosen_action != 1):
		if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)
		
		# Aufruf der Funktionen zur "Bewegung" des Agenten
		learner.choose_action()
		learner.random_step()
		learner.perform_action(env)
		learner.update_Q(env)
		
# Ausgabe der Q-Matrix als Textdatei
f = open("Q MATRIX " + now + ".txt","w")
f.write(str(learner.Q))
f.close()


####################################
# Erstellung und Ausgabe eines GIF #								
####################################

im_agent = plt.imread("../IMG/Lasse_head.png")
im_treat = plt.imread("../IMG/treat.png")
imagebox = OffsetImage(im_agent, zoom=0.02)
imagebox2 = OffsetImage(im_treat, zoom=0.5)

trajfig,trajax = plt.subplots()
camera = Camera(trajfig)

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
		#trajax.add_artist(ab)   #(war schon auskommentiert!)
	else:
		ab = AnnotationBbox(imagebox, (LAST_TRAJECTORY_WITHOUT_PBJUMPS[i+1], 0.0),frameon=False)
	trajax.add_artist(ab)
	trajax.set_xlim(0.0,env.N_states)
	trajax.set_ylim(-1.0,1.0)
	trajax.set_yticks([])
	camera.snap()
	
animation = camera.animate()		
animation.save("LAST_TRAJECTORY_" + now + ".gif")