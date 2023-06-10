import numpy as np

states = 15
target = 12
start = 8
interval = np.arange(9,12)
steps_left = start + (states-target) + 1

p_values=[]

for l in range (10):

    for k in range(100):
        p = 0.6+0.001*k # 71% !!!
        step_list = []

        for i in range(100):
            steps_right = 0
            pos = start

            while (pos!=target):
                pos += 1
                if (np.random.rand()<p) and (pos in interval):
                    pos -= 1
                steps_right += 1

            step_list.append(steps_right)

        if np.mean(step_list)>=steps_left:
            p_values.append(p)
            break

print(np.mean(p_values))






































