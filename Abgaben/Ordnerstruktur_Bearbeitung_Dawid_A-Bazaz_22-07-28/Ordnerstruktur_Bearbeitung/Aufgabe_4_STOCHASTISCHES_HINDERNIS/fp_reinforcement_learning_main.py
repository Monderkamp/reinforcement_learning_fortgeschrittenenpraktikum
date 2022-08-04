import fp_classes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import time as tm

from celluloid import Camera
from datetime import date
import datetime

now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")


'''
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



for key in T_XPOW2_DATAPOINTS.keys():
    T_XPOW2_DATAPOINTS[key] =  np.mean(T_XPOW2_DATAPOINTS[key])

X = np.array(list(T_XPOW2_DATAPOINTS.keys()))
Y = np.array(list(T_XPOW2_DATAPOINTS.values()))
plt.rcParams.update({'font.size': 18})

fig,ax = plt.subplots()

ax.plot(X,Y,label="Simulationsdaten")
p = np.polyfit(X,Y,1)
print(p)
effiD = p[0] / (2)

ax.plot(X,p[0]*X+p[1],linestyle="dashed",label="fit")
ax.legend()
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\left <x^2(t) \right >$")
plt.show()

fig.savefig("AUFGABE_1_MSD_" + str(learner.D) + "_" + "{:1.4f}".format(effiD) + "_Dawid_AlBazaz_" + now + ".png", bbox_inches='tight')
'''



LAST_TRAJECTORY = []

#epi_list = np.arange(0, learner.N_episodes)
count_list = []
opti_count = []

P_amount = 15
P_loop = np.linspace(0, 1, P_amount)
kappa_list =[]
for P in P_loop:
    env = fp_classes.environment()
    learner = fp_classes.agent(env)
    env.P_obstacle = P
    total_action_displacement = 0

    for episode in range(learner.N_episodes):
        if ((episode%(learner.N_episodes//100)) == 0) and episode:
            progress = 1.*episode/learner.N_episodes
            print("------------------------------------------")
            print(f"Fortschritt: {progress}")

        count = 0
        #env.starting_position = np.random.randint(0, env.N_states) # fügt random Position ein
        learner.x = env.starting_position
        learner.chosen_action = None

        # kürzeste Aktionsfolge
        ziel = env.target_position
        Nstate = env.N_states
        i = env.starting_position
        l = 0 #Anzahl schritte nach links
        r = 0 #Anzahl schritte nach rechts
        if i > ziel:
            l = i-ziel
            r = Nstate  -i + ziel
        if i < ziel:
            r = ziel - i
            l = i + Nstate -ziel
        opti_count.append(min(l,r) + 1)

        while (learner.x != env.target_position) or learner.chosen_action != 1:
            if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)

            # eigener Code
            count += 1
            learner.adjust_epsilon(episode)
            learner.x_old = learner.x
            learner.choose_action()
            learner.random_step()
            learner.stoch_obstacle(env)
            learner.perform_action(env)
            if learner.epsilon == 0: total_action_displacement += learner.chosen_action - 1
            learner.update_Q(env)


            pass
        count_list.append(count)

    kappa = total_action_displacement / np.abs(total_action_displacement)
    print("Bei P=", P, " Kappa: ", kappa)
    kappa_list.append(kappa)


fig,ax = plt.subplots()

ax.plot(P_loop, kappa_list, label="Kappa-P-Diagramm")

ax.set_xlabel(r"$P_0$")
ax.set_ylabel(r"$Kappa$")
plt.show()

fig.savefig("AUFGABE_4_"+ "_Dawid_AlBazaz_" + now + ".png", bbox_inches='tight')
'''
ax.semilogy(epi_list, count_list, label="Anzahl Aktionen der Episode")

ax.set_xlabel(r"$Episode$")
ax.set_ylabel(r"$Anzahl Aktionen$")
plt.show()

fig.savefig("AUFGABE_3_ACTION_COUNT_"+ "_Dawid_AlBazaz_" + now + ".png", bbox_inches='tight')
'''
'''
Y = [i/j for i, j in zip(count_list, opti_count)]
ax.semilogy(epi_list, Y, label="Relative Anzahl Aktionen der Episode")

ax.set_xlabel(r"$Episode$")
ax.set_ylabel(r"$Relative Anzahl Aktionen$")
plt.show()

fig.savefig("AUFGABE_3_RELATIV_ACTION_COUNT_"+ "_Dawid_AlBazaz_" + now + ".png", bbox_inches='tight')
'''
exit()


im_agent = plt.imread("../IMG/Lasse_head.png")
im_treat = plt.imread("../IMG/treat.png")
imagebox = OffsetImage(im_agent, zoom=0.02)
imagebox2 = OffsetImage(im_treat, zoom=0.5)

trajfig,trajax = plt.subplots()
camera = Camera(trajfig)

f = open("Q_MATRIX_" + now + ".txt", "w")
f.write(str(learner.Q))
f.close()

#'''
#--------------------------------------------------------------------
#IM FOLGENDEN CODE BLOCK WIRD DIE TRAJEKTORIE ALS GIF AUSGEGEBEN
#WENN DIESER TEIL ZU LANGE DAUERT ODER SIE DIE NÖTIGEN LIBRARIES NICHT INSTALLIEREN KÖNNEN,
#KOMMENTIEREN SIE DIESEN CODE BLOCK AUS
#--------------------------------------------------------------------


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
        #trajax.add_artist(ab)
    else:
        ab = AnnotationBbox(imagebox, (LAST_TRAJECTORY_WITHOUT_PBJUMPS[i+1], 0.0),frameon=False)
    trajax.add_artist(ab)
            

    trajax.set_xlim(0.0,env.N_states)
    trajax.set_ylim(-1.0,1.0)
    trajax.set_yticks([])
    camera.snap()
animation = camera.animate()		
animation.save("LAST_TRAJECTORY_" + now + ".gif")
#'''
