# import packages
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial


# constants
NUM_OF_ARMS = 10
NUM_OF_STEPS = 1000
NUM_OF_RUNS = 2000


# environment implementation
class KArmedBandit:

    def __init__(self):
        self.reward = np.random.randn(NUM_OF_ARMS)
        self.optimal_choice = np.argmax(self.reward)

    def step(self, action):
        return self.reward[action] + np.random.randn()
    

# agent implementation
class GreedyAgent:

    def __init__(self, epsilon=0, initial=0, alpha=0.1):
        self.q_estimation = np.zeros(NUM_OF_ARMS) + initial
        self.epsilon = epsilon
        self.alpha = alpha
    
    def act(self):
        if np.random.rand() < self.epsilon: action = np.random.choice(np.arange(NUM_OF_ARMS))
        else: action = np.argmax(self.q_estimation)
        return action
    
    def learn(self, action, reward):
        self.q_estimation[action] += self.alpha * (reward - self.q_estimation[action])


# run tests
def run(agents):
    optimal = np.zeros((len(agents), NUM_OF_STEPS))
    for i, agent in enumerate(agents):
        env = KArmedBandit()
        for step in np.arange(NUM_OF_STEPS):
            action = agent.act()
            reward = env.step(action)
            agent.learn(action, reward)
            optimal[i, step] = (action == env.optimal_choice)
    return optimal


# generate results 
agents = [GreedyAgent(epsilon=0.1), GreedyAgent(initial=5)]
optimal = Parallel(n_jobs=6)(delayed(partial(run, agents))() for _ in range(NUM_OF_RUNS))


# generate figure 2.3
avg_optimal = np.mean(optimal, axis=0)

plt.figure(figsize=(9, 6))

plt.plot(avg_optimal[0], label='e-greedy')
plt.plot(avg_optimal[1], label='optimistic initialization')

plt.xlabel('steps')
plt.ylabel('% optimal action')
plt.legend()

plt.savefig('Figure2-3.png')