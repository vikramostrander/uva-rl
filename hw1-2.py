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
    

# greedy agent implementation
class GreedyAgent:

    def __init__(self, epsilon=0):
        self.q_estimation = np.zeros(NUM_OF_ARMS)
        self.epsilon = epsilon
        self.action_counts = np.zeros(NUM_OF_ARMS)
    
    def act(self):
        if np.random.rand() < self.epsilon: action = np.random.choice(np.arange(NUM_OF_ARMS))
        else: action = np.argmax(self.q_estimation)
        self.action_counts[action] += 1
        return action
    
    def learn(self, action, reward):
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_counts[action]


# ucb agent implementation
class UCBAgent:

    def __init__(self, c=0):
        self.q_estimation = np.zeros(NUM_OF_ARMS)
        self.const = c
        self.time = 0
        self.action_counts = np.zeros(NUM_OF_ARMS)
    
    def act(self):
        action = np.argmax(self.q_estimation + self.const * np.sqrt(np.log(self.time + 1) / (self.action_counts + 1e-5)))
        self.time += 1
        self.action_counts[action] += 1
        return action
    
    def learn(self, action, reward):
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_counts[action]


# run tests
def run(agents):
    rewards = np.zeros((len(agents), NUM_OF_STEPS))
    for i, agent in enumerate(agents):
        env = KArmedBandit()
        for step in np.arange(NUM_OF_STEPS):
            action = agent.act()
            reward = env.step(action)
            agent.learn(action, reward)
            rewards[i, step] = reward
    return rewards


# generate results 
agents = [GreedyAgent(epsilon=0.1), UCBAgent(c=2)]
rewards = Parallel(n_jobs=6)(delayed(partial(run, agents))() for _ in range(NUM_OF_RUNS))


# generate figure 2.4
avg_rewards = np.mean(rewards, axis=0)

plt.figure(figsize=(9, 6))

plt.plot(avg_rewards[0], label='e-greedy (e=0.1)')
plt.plot(avg_rewards[1], label='ucb (c=2)')

plt.xlabel('steps')
plt.ylabel('average reward')
plt.legend()

plt.savefig('figure2-4.png')