# A1
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


T_XPOW2_DATAPOINTS = dict()
T0 = tm.time()

for i in range(learner.N_episodes):
	if ((i%(learner.N_episodes//100)) == 0) and i:
		progress = 1.*i/learner.N_episodes
		print("------------------------------------------")
		print(f"Fortschritt: {100*progress} %")
		Trem = (tm.time() - T0)*(1./progress - 1.)
		print(f"Verbleibende Zeit: {Trem} s")
	
	learner.x = 0 #Start an dieser Position, warum? (war bereits vorgegeben)

	for t in range(learner.tmax_MSD):
		if not t in T_XPOW2_DATAPOINTS.keys():
			T_XPOW2_DATAPOINTS[t] = []
		T_XPOW2_DATAPOINTS[t].append(learner.x**2)
		
		# Aufruf der Diffusions-Funktion
		learner.random_step()	


for key in T_XPOW2_DATAPOINTS.keys():
	T_XPOW2_DATAPOINTS[key] =  np.mean(T_XPOW2_DATAPOINTS[key])

X = np.array(list(T_XPOW2_DATAPOINTS.keys()))
Y = np.array(list(T_XPOW2_DATAPOINTS.values()))
plt.rcParams.update({'font.size': 18})

fig,ax = plt.subplots()

ax.plot(X,Y,label="Simulationsdaten")
p = np.polyfit(X,Y,1)

ax.plot(X,p[0]*X+p[1],linestyle="dashed",label="fit")
ax.legend()
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\left <x^2(t) \right >$")

# Direktes Abspeichern des Plots als .PNG
filename=f'Aufgabe_1_MSD_{learner.D}_{p[0]/2}_KocksKlett_'+now+'.png' 
fig.savefig(filename, bbox_inches='tight')

plt.show()