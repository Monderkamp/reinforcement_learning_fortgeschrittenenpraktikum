import fp_classes

"""
Q[0] t
learning curve
"""

if __name__ == "__main__":
	env = fp_classes.environment()
	learner = fp_classes.agent(env)
	
	for episode in range(learner.N_episodes):
		#print(episode)
		learner.x = env.starting_position
		while (learner.x != env.target_position) or learner.chosen_action != 1:
			
			learner.adjust_epsilon(episode)
			learner.choose_action()
			learner.perform_action(env)
			learner.update_Q(env)


	print(learner.Q)
