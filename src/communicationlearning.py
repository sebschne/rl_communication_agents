import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import Env, spaces
from gym.utils import seeding
import gym
from env.communication_game_env_robot import CommunicationPatternRobotEnv
from env.communication_game_env_human import CommunicationPatternHumanEnv
import logging
import time
from agent.qtable_robot import QTableAgent
from agent.qnetwork import QNetworkAgent
import argparse
np.set_printoptions(precision=4)

ROBOT_ACTION_MEANING = {
    0 : "LOOK_0",
    1 : "LOOK_1",
    2 : "LOOK_2",
    3 : "LOOK_H",
    4 : "SOUND_A",
    5 : "SOUND_B",
    6 : "SOUND_C",
    7 : "NOOP",
}

GREATER_STATE_ROBOT_ACTION_MEANING = {
    0 : "LOOK_0",
    1 : "LOOK_1",
    2 : "LOOK_2",
    3 : "LOOK_3",
    4 : "LOOK_4",
    5 : "LOOK_H",
    6 : "SOUND_A",
    7 : "SOUND_B",
    8 : "SOUND_C",
    9 : "SOUND_D",
}

HUMAN_ACTION_MEANING = {
    0 : "PUT_CUP_0",
    1 : "PUT_CUP_1",
    2 : "PUT_CUP_2",
    3 : "NOOP",
}

GREATER_STATE_HUMAN_ACTION_MEANING = {
    0 : "PUT_CUP_1",
    1 : "PUT_CUP_2",
    2 : "PUT_CUP_3",
    3 : "PUT_CUP_4",
    4 : "PUT_CUP_5",
    5 : "NOOP",
}



CUP_LOCATION = {
0 : [ -0.5, 0.5, 0.28 ],
1 : [ -0.3, 0.5, 0.28 ],
2 : [ 0.0, 0.5, 0.28 ],
3 : [ 0.35, 0.5, 0.28 ],
4 : [ 0.5, 0.5, 0.28 ],
}

SOUND = {
6 : "vUl-",
7 : "vAl-",
8 : "vOl-",
9 : "vIl-",
10 : "vEl-"
}

class CommunicationLearning():
    def __init__(self, agent, exp_strategy, fixed_cup_pos, fixed_goal_pos, online_vis, visualisation, num_episodes, num_of_simulation, use_robot, reuse_qtable,  file_name):
        print "using robot: ", use_robot
        use_robot = use_robot
        fixed_cup_pos = fixed_cup_pos
        fixed_goal_pos = fixed_goal_pos
        online_visualisation = online_vis
        visualisation = visualisation
        greedy = exp_strategy
        num_episodes = num_episodes
        num_of_simulation = num_of_simulation
        file_name = file_name
        agent = agent
        reuse_qtable = reuse_qtable



        if use_robot:
            GAZE_LOCATION = {
            0 : RobotGaze(25.0,-15.0,0.0),
            1 : RobotGaze(15.0,-15.0,0.0),
            2 : RobotGaze(0.0,-15.0,0.0),
            3 : RobotGaze(-15.0,-15.0,0.0),
            4 : RobotGaze(-25.0,-15.0,0.0),
            5 : RobotGaze(0.0,0.0,0.0),
            }
            from hlrc_client import *
            import python_simworld_control as psc  # YARP is automatically initalised now

            wc = psc.WorldController() # you will see YARP spitting out debugging messages as RPC client connects to the server
            # Let's create some objects...
            box1= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ 0.45, 0.2, 0.33 ], [ 0.7, 0.3, 0.2 ])
            box2= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ -0.45, 0.2, 0.33 ], [ 0.7, 0.3, 0.2 ])
            box3= wc.create_object('box', [ 2.0, 0.1, 0.9 ], [ 0.0, 0.45, 0.475 ], [ 0.7, 0.3, 0.2 ])

            # Green sphere for normal experiment
            green_sphere1 = wc.create_object('ssph', [ 0.04 ], [ 0.35, 0.5, 0.28 ], [ 0, 1, 0 ])
            red_sphere1 = wc.create_object('ssph', [ 0.04 ], [ 0.35, 0.45, 0.28 ], [ 1, 1, 0 ])

        #green_sphere2 = wc.create_object('ssph', [ 0.04 ], [ 0.17, 0.6, 0.355 ], [ 0, 1, 0 ])
        #green_sphere3 = wc.create_object('ssph', [ 0.04 ], [ -0.02, 0.6, 0.28 ], [ 0, 1, 0 ])


        robot_env = CommunicationPatternRobotEnv(nS=5, nG=5, nA = 10, use_robot = False, fixed_cup_pos=fixed_cup_pos, fixed_goal_pos=fixed_goal_pos)
        human_env = CommunicationPatternHumanEnv(nS=1, nA = 6, nR = 10)

        if visualisation:
            fig1 = plt.figure()
            fig1.suptitle("episode reward", fontsize =20)

            ax1 = fig1.add_subplot(111)

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)

            if fixed_cup_pos is False or fixed_goal_pos is False:
                fig4 = plt.figure()
                fig4.suptitle('robot q table', fontsize=20)
                robot_heatmap = fig4.add_subplot(111)
                y = [i for i in range(0,25)]
                x = [i for i in range(0,10)]

                x_labels = [GREATER_STATE_ROBOT_ACTION_MEANING[i] for i in range(0,10)]
                y_labels = [i for i in range(0,25)]

                plt.xticks(y, y_labels)
                plt.yticks(x, x_labels)

            else:
                fig4 = plt.figure()
                fig4.suptitle('robot q table', fontsize=20)
                robot_heatmap = fig4.add_subplot(111)
                y = [i for i in range(0,5)]
                x = [i for i in range(0,10)]

                x_labels = [GREATER_STATE_ROBOT_ACTION_MEANING[i] for i in range(0,10)]
                y_labels = [i for i in range(0,5)]

                plt.xticks(y, y_labels)
                plt.yticks(x, x_labels)

            onlinefig = plt.figure()
            onlinefig.suptitle("human q table", fontsize =20)
            ax3 = onlinefig.add_subplot(111)
            x = [i for i in range(0,6)]
            y = [i for i in range(0,10)]

            x_labels = [GREATER_STATE_HUMAN_ACTION_MEANING[i] for i in range(0,6)]
            y_labels = [GREATER_STATE_ROBOT_ACTION_MEANING[i] for i in range(0,10)]

            plt.yticks(y, y_labels,fontsize=6)
            plt.xticks(x, x_labels,fontsize=6)

            first_vis = True

        score_list = []
        reward_array = np.zeros([num_of_simulation,num_episodes])

        for k in range(num_of_simulation):
            if fixed_cup_pos:
                if agent == 'qtable':

                    robot = QTableAgent(robot_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 10, nS = 5)
                    human = QTableAgent(human_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 6, nS = 10)
                else:

                    robot = QNetworkAgent(robot_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 10, nS = 5)
                    human = QNetworkAgent(human_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 6, nS = 10)
            else:
                if agent == 'qtable':

                    robot = QTableAgent(robot_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 10, nS = 25)
                    human = QTableAgent(human_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 6, nS = 10)
                else:

                    robot = QNetworkAgent(robot_env,num_episodes,reuse_qtable = reuse_qtable, greedy = greedy, nA = 10, nS = 25)
                    human = QNetworkTableAgent(human_env,num_episodes,reuse_qtable = reuse_qtable, greedy = greedy, nA = 6, nS = 10)


            robot_rList = []
            human_rList = []
            jList = []


            for i in range(num_episodes):

                r_s, cup_pos, cup_goal = robot_env.reset()
            #    print "cup pos: %s cup goal: %s" % (cup_pos, cup_goal)
                h_s = human_env._reset(5,cup_pos)
                robot_rAll = 0
                human_rAll = 0
                d_robot = False
                d_human = False
                j = 0

                if use_robot:
                    icub = RobotController("ROS", "/icub", loglevel=logging.WARNING)
                    # activate a neutral face
                    icub.set_default_emotion(RobotEmotion(RobotEmotion.NEUTRAL))
                    # look straight
                    straight = RobotGaze(0.0,0.0,0.0)
                    icub.set_gaze_target(straight)
                    wc.move_object(green_sphere1,CUP_LOCATION[cup_pos])
                    wc.move_object(red_sphere1,CUP_LOCATION[cup_goal])

                #The Q-Table learning algorithm
                while j < 4:
                    j+=1
                    #first robot makes an action
                    r_a = robot.take_action(i)
                    #print "Robot action", r_a
                    if use_robot:
                        if r_a< 6:
                            print GREATER_STATE_ROBOT_ACTION_MEANING[r_a]
                            icub.set_gaze_target(GAZE_LOCATION[r_a])
                            #time.sleep(0.5)
                        if r_a>= 6:
                            icub.set_speak(SOUND[r_a], blocking=False)
                            print GREATER_STATE_ROBOT_ACTION_MEANING[r_a]

                    #second human observes this actions
                    human_env._update(r_a)
                    #third human makes action
                    h_a = human.take_action(i)

                    if use_robot and h_a != 5:
                        wc.move_object(green_sphere1,CUP_LOCATION[h_a])

                    #fourth robots updates environment based on the action
                    robot_env._update(h_a)


                    #generate rewards and new states
                    r_s1,r_r,r_d,_ = robot_env.step(r_a)
                    robot.update_table(r_s1,r_a,r_r)
                    robot.env.s = r_s1


                    h_s1,h_r,h_d,_ = human_env._step(h_a,r_a,r_d,r_r)
                    # print "updating human", h_s, h_s1, h_r
                    human.update_table(h_s1,h_a,h_r)
                    h_s = h_s1
                    r_s = r_s1
                    robot_rAll += r_r
                    human_rAll += h_r

                    if h_d == True:
                        #print "reward robot", robot_rAll
                        #print "reward human", human_rAll

                        if use_robot:

                            icub.set_gaze_target(GAZE_LOCATION[5])
                            icub.set_default_emotion(RobotEmotion(RobotEmotion.HAPPY))
                            ani = RobotAnimation(RobotAnimation.HEAD_NOD)
                            ani.repetitions = 3
                            icub.set_head_animation(ani, blocking=False)
                            #time.sleep(2)
                        break
                jList.append(j)
                robot_rList.append(robot_rAll)
                human_rList.append(human_rAll)
                reward_array[k,i] = robot_rAll

                # if i % 100 == 0:
                # print "episode: ", i, j , str(sum(robot_rList)/num_episodes)
                if online_visualisation:
                    ax2.cla()
                    ax2.plot(robot_rList)
                    plt.pause(0.0005)

                    ax3.imshow(human.Q, cmap='hot', interpolation='nearest')
                    x = [i for i in range(0,6)]
                    y = [i for i in range(0,10)]
                    x_labels = [GREATER_STATE_HUMAN_ACTION_MEANING[i] for i in range(0,6)]
                    y_labels = [GREATER_STATE_ROBOT_ACTION_MEANING[i] for i in range(0,10)]
                    plt.yticks(y, y_labels)
                    plt.xticks(x, x_labels)
                    # plt.show()
                    plt.pause(0.0005)

                if reuse_qtable == True:
                    indices = np.where(human.Q == human.Q.max())
                    human_x_y_coords =  [indices[0], indices[1]]
                    indices = np.where(robot.Q == robot.Q.max())
                    robot_x_y_coords =  [indices[0], indices[1]]
                    robot.list_of_learn_actions.append(human_x_y_coords[0])
                    print robot.list_of_learn_actions



            #print "Percent of succesful episodes: " + str(sum(robot_rList)/num_episodes) + "%"

            # plt.plot(jList)
            # plt.show()
            #print "Score over time: " +  str(sum(robot_rList)/num_episodes)
            score_list.append(sum(robot_rList)/num_episodes)
            #print "Final Q-Table Values Robot"
            #print robot.Q
            #print "Final Q-Table Values Human"
            #print human.Q
            if visualisation:
                ax1.cla()
                ax1.plot(robot_rList)

                heatmap2 = robot_heatmap.imshow(np.matrix.transpose(robot.Q), cmap='hot', interpolation='nearest',label ='robot q table')


                heatmap = ax3.imshow(human.Q, cmap='hot', interpolation='nearest',label='human q tale')

                plt.pause(1.0)





                if first_vis:
                    plt.colorbar(heatmap)

                    first_vis = False

                    plt.pause(0.0005)


                # plt.show()
            #### Coordinates of the highest action-state pair

            #print x_y_coords
        #print human.Q
                #update q-table for the agens
        if file_name != None:
            np.savetxt(file_name, score_list,fmt='%1.5f', delimiter=',')

        print "Overall score: ", np.mean(score_list)
        print "Overall var: ", np.var(score_list)
        # print robot.Q
        # print reward_array
        string = agent +"_" + str(fixed_cup_pos) + "_" + str(fixed_goal_pos) + ".csv"
        np.savetxt(string, reward_array, fmt='%1.5f', delimiter=',')
if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help='which learning agent: qtable|qnetwork', default = "qtable")
    parser.add_argument('--exp_strategy', help='which exploration strategy: egreedy|softmax', default = "egreedy")
    parser.add_argument('--num_episodes',help="number of learning episodes",
                    type=int, default = 2000)
    parser.add_argument('--num_of_simulation', help="number of simulations",
                    type=int, default=10)
    parser.add_argument('-r','--use_robot', help="use the robot in simulation",
                     action="store_true")
    parser.add_argument('-c','--fixed_cup_pos',help="fixed starting position of cup",
                     action="store_false")
    parser.add_argument("-g",'--fixed_goal_pos',help="fixed goal position of cup",
                     action="store_false")
    parser.add_argument('-o','--online_vis',help="visualize learning online",
                    action = 'store_true')
    parser.add_argument('-v','--visualisation',help="visualize the learning results",
                    action='store_true')
    parser.add_argument('--file_name',help="name of the output file", default = None)
    parser.add_argument('-q','--reuse_qtable',help="visualize the learning results",
                        action='store_true')

    args = parser.parse_args()

    print args
    asdf = CommunicationLearning(agent=args.agent,exp_strategy= args.exp_strategy, fixed_cup_pos=args.fixed_cup_pos,fixed_goal_pos = args.fixed_goal_pos, online_vis = args.online_vis,
     num_episodes = args.num_episodes, reuse_qtable = args.reuse_qtable, num_of_simulation = args.num_of_simulation, visualisation= args.visualisation, file_name=args.file_name, use_robot = args.use_robot)
