import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import Env, spaces
from gym.utils import seeding
import gym
import math


def ind_max(x):
  m = max(x)
  return x.index(m)

class UCB1():
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values
    return

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    return ind_max(ucb_values)

  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return

class QTableAgent():

    def __init__(self,env,num_episodes,nA, nS, reuse_qtable = False ,greedy = True, lr= .8,y=.95, e = 0.93, tau=3.0):

        self.visited_states = []
        self.env = env
        ###Q-learning with q-tables###
        self.e = e
        #Initialize table with all zeros
        self.reuse_qtable = reuse_qtable
        self.tau = tau
        self.greedy = greedy
        self.Q = np.zeros([self.env.observation_space.n,self.env.action_space.n])
        #print "initialized q-table with shape: ", self.Q.shape
        # Set learning parameters
        self.lr = lr
        self.y = y
        self.num_episodes = num_episodes
        #create lists to contain total rewards and steps per episode
        #jList = []
        self.list_of_learn_actions = []

        if self.greedy == 'UCB':

            self.bandits = []
            for i in range(self.env.observation_space.n):
                self.bandits.append(UCB1([],[]))
                self.bandits[i].initialize(self.env.action_space.n)

    def gibbs_choice(self):
        p = np.array([self.Q[(self.env.s,x)]/self.tau for x in range(self.env.action_space.n)])
        prob_actions = np.exp(p) / np.sum(np.exp(p))
        cumulative_probability = 0.0
        choice = random.uniform(0,1)
        for a,pr in enumerate(prob_actions):
            cumulative_probability += pr
            if cumulative_probability > choice:
                return a



    def take_action(self,i):
        # a = np.argmax(self.Q[self.env.s,:] + np.random.randn(1,self.env.action_space.n)*(1./(i+1)))
        if self.greedy == 'egreedy' or self.greedy == False:
            if self.reuse_qtable is True and len(self.list_of_learn_actions)>0:
                print "reusing qtable"
                q = self.Q
                for j in self.list_of_learn_actions:
                    q[:,j] = -1
                if np.random.random()>self.e*(1./(i+1)):
                    a = np.argmax(q[self.env.s,:])
                else:
                    a = random.randrange(self.env.action_space.n)

            if np.random.random()>self.e*(1./(i+1)):
                a = np.argmax(self.Q[self.env.s,:])
            else:
                a = random.randrange(self.env.action_space.n)

        elif self.greedy == 'softmax':
            #print 'gibbs'
            a = self.gibbs_choice()
        elif self.greedy == 'UCB':
            a = self.bandits[self.env.s].select_arm()



        return a
    def update_table(self,s1,a,r):
        if s1 not in self.visited_states:
            self.visited_states.append(s1)
            # print self.visited_states
        self.Q[self.env.s,a] = self.Q[self.env.s,a] + self.lr*(r + self.y*np.max(self.Q[s1,:]) - self.Q[self.env.s,a])
        if self.greedy == 'UCB':
            self.bandits[self.env.s].update(a,self.Q[self.env.s,a])
