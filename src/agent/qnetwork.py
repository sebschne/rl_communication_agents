import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import Env, spaces
from gym.utils import seeding
import gym

class QNetworkAgent():

    def take_action(self,i):
        a,self.allQ = self.sess.run([self.predict,self.Qout],
        feed_dict={self.inputs1:np.identity(self.nS)[self.env.s:self.env.s+1]})
        if self.greedy == 'egreedy':
            if np.random.rand(1) > self.e:
                a[0] = self.env.action_space.sample()

        return a[0]

    def update_table(self,s1,a,r):
            #Obtain the Q' values by feeding the new state through our network
            Q1 = self.sess.run(self.Qout,feed_dict={self.inputs1:np.identity(self.nS)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = self.allQ
            targetQ[0,a] = r + self.y*maxQ1
            #Train our network using target and predicted Q values
            _,self.W1 = self.sess.run([self.updateModel,self.W],feed_dict={self.inputs1:np.identity(self.nS)[self.env.s:self.env.s+1],self.nextQ:targetQ})
            # rAll += r
            # s = s1

    def __init__(self,env, num__episodes, nA, nS, greedy = True, lr= .8, e = 0.93, y = .99):
        self.y = y
        self.e = e
        self.env = env
        self.nS = nS
        self.lr = lr
        self.greedy = greedy
        ###Q-learning with neural networks####
        tf.reset_default_graph()
        #These lines establish the feed-forward part of the network used to choose actions
        self.inputs1 = tf.placeholder(shape=[1,nS],dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([nS,nA],0,0.01))
        self.Qout = tf.matmul(self.inputs1,self.W)
        self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1,nA],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = self.trainer.minimize(self.loss)
        self.init = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(self.init)


#     for i in range(num_episodes):
#         #Reset environment and get first new observation
#         s = env.reset()
#         rAll = 0
#         d = False
#         j = 0
#         #The Q-Network
#         while j < 99:
#             j+=1
#             #Choose an action by greedily (with e chance of random action) from the Q-network
#
#                 #HERE ICUB SHOULD DO SOMETHING
#
#             #Get new state and reward from environment
#             s1,r,d,_ = env.step(a[0])
#
#             if d == True:
#                 #Reduce chance of random action as we train the model.
#                 e = 1./((i/50) + 10)
#                 break
#         jList.append(j)
#         rList.append(rAll)
# print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"
