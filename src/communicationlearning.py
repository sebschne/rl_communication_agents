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

import rospy

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
0 : [ -0.7, 0.5, 0.48 ],
1 : [ -0.3, 0.5, 0.48 ],
2 : [ 0.0, 0.5, 0.48 ],
3 : [ 0.18, 0.5, 0.48 ],
4 : [ 0.7, 0.5, 0.48 ],
}

SOUND = {
6 : "vUl-",
7 : "vAl-",
8 : "vOl-",
9 : "vIl-",
10 : "vEl-"
}

class CommunicationLearning():

    def callback(self,data):
        #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        intermediate = data.data
        if self.allowChange:
            self.color_detected = intermediate

    def __init__(self, agent, nao_ip , interactive, robot_type, exp_strategy, fixed_cup_pos, fixed_goal_pos, online_vis, visualisation, num_episodes, num_of_simulation, use_robot, reuse_qtable,  file_name):
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
        robot_type = robot_type
        self.color_detected=""
        names = ["HeadPitch", "HeadYaw"]
        interactive = interactive
        robotIP = nao_ip
        PORT = 9559

        self.allowChange = False

        ### We are using the robot so we need to import some stuff
        if use_robot:

            if robot_type == "icub":

                ### using the hlr client to send icub commands

                from hlrc_client import *
                import python_simworld_control as psc  # YARP is automatically initalised now
                ## gaze locations for the robot to look at
                # TODO: load them from a config file if they are provided
                GAZE_LOCATION = {
                0 : RobotGaze(21.0,-38.0,0.0),
                1 : RobotGaze(10.0,-40.0,0.0),
                2 : RobotGaze(0.0,-40.0,0.0),
                3 : RobotGaze(-12.5,-40.0,0.0),
                4 : RobotGaze(-19.0,-40,0.0),
                5 : RobotGaze(0.0,0.0,0.0),
                }

                ### we are using the robot vision to see were the cup position is
                # TODO: change the topcis to soomething more meaningful, create variables
                from std_msgs.msg import String
                rospy.init_node('listener', anonymous=True)
                rospy.Subscriber('chatter', String, self.callback)

                # spin() simply keeps python from exiting until this node is stopped

                #self.rob_vision =  RobotVision()


                wc = psc.WorldController() # you will see YARP spitting out debugging messages as RPC client connects to the server
                # Let's create some objects...
                # box1= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ 0.45, 0.2, 0.33 ], [ 0.7, 0.3, 0.2 ])
                # box2= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ -0.45, 0.2, 0.33 ], [ 0.7, 0.3, 0.2 ])
                # box3= wc.create_object('box', [ 2.0, 0.1, 0.9 ], [ 0.0, 0.45, 0.475 ], [ 0.7, 0.3, 0.2 ])

                box1= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ 0.45, 0.2, 0.33 ], [ 0.0, 0.0, 0.0 ])
                box2= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ -0.45, 0.2, 0.33 ], [ 0.0, 0.0, 0.0 ])
                box3= wc.create_object('box', [ 2.0, 0.1, 0.9 ], [ 0.0, 0.45, 0.475 ], [ 0.0, 0.0, 0.0 ])

                # Green sphere for normal experiment
                green_sphere1 = wc.create_object('ssph', [ 0.04 ], [ 0.35, 0.5, 0.5 ], [ 1, 1, 0 ])
                red_sphere1 = wc.create_object('ssph', [ 0.04 ], [ 0.35, 0.5, 0.5 ], [ 1, 0, 0 ])
            elif robot_type == "nao":
                print "using nao"
                GAZE_LOCATION = {
                0 : [[1.0],[-1.0]],
                1 : [[1.0],[-0.5]],
                2 : [[1.0],[0.0]],
                3 : [[1.0],[0.5]],
                4 : [[1.0],[1.0]],
                5 : [[0.0],[0.0]],
                }
                import naoqi
                from naoqi import ALProxy
                from std_msgs.msg import String
                rospy.init_node('listener', anonymous=True)
                rospy.Subscriber('chatter', String, self.callback)

                import python_simworld_control as psc  # YARP is automatically initalised now

                wc = psc.WorldController() # you will see YARP spitting out debugging messages as RPC client connects to the server

                box1= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ 0.45, 0.2, 0.33 ], [ 0.0, 0.0, 0.0 ])
                box2= wc.create_object('box', [ 0.8, 0.2, 0.6 ], [ -0.45, 0.2, 0.33 ], [ 0.0, 0.0, 0.0 ])
                box3= wc.create_object('box', [ 2.0, 0.1, 0.9 ], [ 0.0, 0.45, 0.475 ], [ 0.0, 0.0, 0.0 ])

                # Green sphere for normal experiment
                green_sphere1 = wc.create_object('ssph', [ 0.04 ], [ 0.35, 0.5, 0.5 ], [ 1, 1, 0 ])
                red_sphere1 = wc.create_object('ssph', [ 0.04 ], [ 0.35, 0.5, 0.5 ], [ 1, 0, 0 ])
                print "object creation ready"
        #green_sphere2 = wc.create_object('ssph', [ 0.04 ], [ 0.17, 0.6, 0.355 ], [ 0, 1, 0 ])
        #green_sphere3 = wc.create_object('ssph', [ 0.04 ], [ -0.02, 0.6, 0.28 ], [ 0, 1, 0 ])


        robot_env = CommunicationPatternRobotEnv(nS=5, nG=5, nA = 10, use_robot = False, fixed_cup_pos=fixed_cup_pos, fixed_goal_pos=fixed_goal_pos)
        human_env = CommunicationPatternHumanEnv(nS=1, nA = 6, nR = 10)



        ### preparing the online visualisation figures
        if visualisation or online_visualisation:
            fig1 = plt.figure()
            fig1.suptitle("episode reward", fontsize =20)

            ax1 = fig1.add_subplot(111)

            fig2 = plt.figure()
            fig2.suptitle('online reward visualisation', fontsize=20)
            ax2 = fig2.add_subplot(111)
            ax2.set_xlabel('episode')
            ax2.set_ylabel('reward')

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

            plt.yticks(y, y_labels)
            plt.xticks(x, x_labels)

            first_vis = True

        score_list = []
        reward_array = np.zeros([num_of_simulation,num_episodes])

        for k in range(num_of_simulation):
            #if we are using a fixed cup starting position we can take the smaller state space
            if fixed_cup_pos:

                if agent == 'qtable': ### using the qtable learning agent

                    robot = QTableAgent(robot_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 10, nS = 5)
                    human = QTableAgent(human_env,num_episodes,reuse_qtable = reuse_qtable,greedy = greedy, nA = 6, nS = 10)
                else: ### using a q-network learning agent, which is actually worse then the table learning agent

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


                ### If we use the robot we need to initalize the robot controller
                if use_robot:


                    if robot_type == "icub":

                        # TODO: add a variable for the icub topic
                        robot_ctrl = RobotController("ROS", "/icub", loglevel=logging.WARNING)
                        # activate a neutral face
                        robot_ctrl.set_default_emotion(RobotEmotion(RobotEmotion.NEUTRAL))
                        # look straight
                        straight = RobotGaze(0.0,0.0,0.0)
                        robot_ctrl.set_gaze_target(straight)
                        current_gaze_pos = 5
                        # TODO sleep so that the robot can reach the target. Is there a better way to do it using the humotion api?
                        time.sleep(1)

                        wc.move_object(green_sphere1,CUP_LOCATION[cup_pos])
                        wc.move_object(red_sphere1,CUP_LOCATION[cup_goal])
                    elif robot_type == "nao":
                        robot_ctrl = ALProxy("ALMotion", robotIP, PORT)
                        nao_tts = ALProxy("ALTextToSpeech" , robotIP, PORT)
                        timeLists   = [[1.0], [ 1.0]]
                        isAbsolute  = True
                        robot_ctrl.angleInterpolation(names, GAZE_LOCATION[5], timeLists, isAbsolute)
                        current_gaze_pos = 5
                        wc.move_object(green_sphere1,CUP_LOCATION[cup_pos])
                        wc.move_object(red_sphere1,CUP_LOCATION[cup_goal])

                #The Q-Table learning algorithm
                #TODO Vary the length of the episode. Currently fixed to 3 to inforce efficient learning and not just trying every position
                while j < 4:
                    j+=1
                    #first robot makes an action
                    r_a = robot.take_action(i)
                    #print "Robot action", r_a
                    if use_robot:
                        ### robot uses a gaze action. Setting the
                        if r_a< 6:
                            current_gaze_pos = r_a
                            print GREATER_STATE_ROBOT_ACTION_MEANING[r_a]
                            if robot_type == 'icub':
                                robot_ctrl.set_gaze_target(GAZE_LOCATION[r_a])
                                # TODO sleep so that the robot can reach the target. Is there a better way to do it using the humotion api?
                                time.sleep(2)
                            elif robot_type == 'nao':
                                robot_ctrl.angleInterpolation(names, GAZE_LOCATION[r_a], timeLists, isAbsolute)


                            #time.sleep(0.5)
                        ### if robot produces sound we do not set the gaze_pos
                        if r_a>= 6 and robot_type != 'nao':
                            robot_ctrl.set_speak(SOUND[r_a], blocking=False)
                            print GREATER_STATE_ROBOT_ACTION_MEANING[r_a]
                        elif r_a>= 6 and robot_type == 'nao':
                            nao_tts.say(SOUND[r_a])
                    #second human observes this actions
                    if not interactive:
                        human_env._update(r_a)

                    #third human makes action

                    # TODO: in interactive mode this needs to  be changed!
                    if interactive:
                        # wait for user action
                        time.sleep(6)
                    else:
                        h_a = human.take_action(i)
                    # h_a = human.take_action(i)


                    #fourth robots updates environment based on the action
                    if not interactive:
                        if use_robot and h_a != 5 :
                            wc.move_object(green_sphere1,CUP_LOCATION[h_a])


                    if use_robot:
                        detected = False
                        ### looking not yet at goal

                        if current_gaze_pos is not cup_goal:
                            self.color_detected=""
                            print "checking position: ", current_gaze_pos
                            ## check position where you are looking at
                            # print "checking the position where i am looking at:", current_gaze_pos
                            self.allowChange = True
                            time.sleep(2)
                            # print self.color_detected
                            if self.color_detected == "green":
                                # if h_a == current_gaze_pos:
                                #     print "IT IS CORRECT"
                                print "found green cup"
                                h_a = current_gaze_pos
                                robot_env._update(h_a)
                                detected = True
                            else:
                                print "did not find green cup"

                            if not detected:
                                # print "not yet detected, lookin at all the other positions except for ", current_gaze_pos
                                for gaze in range(len(GAZE_LOCATION)-1):
                                    print "trying all the other positions, ", gaze, current_gaze_pos
                                    self.color_detected=""
                                    ### Only check the position where we have not started.
                                    if gaze != current_gaze_pos:
                                        print "gaze is not equal current position"
                                        ### We do not want to detect cups while looking around put only when the gaze has reached the target. Dirty hack with time.sleep(). Is there a way to get a call back from humotion and then check the etected color?
                                        self.allowChange = False
                                        if robot_type == "icub":
                                            robot_ctrl.set_gaze_target(GAZE_LOCATION[gaze])
                                            time.sleep(2)
                                        elif robot_type == "nao":
                                            robot_ctrl.angleInterpolation(names, GAZE_LOCATION[gaze], timeLists, isAbsolute)
                                        self.allowChange = True
                                        time.sleep(2)

                                        if not interactive and self.color_detected == "green":
                                            if h_a == gaze:

                                                h_a = gaze
                                                robot_env._update(h_a)
                                                detected = True
                                                break
                                        elif self.color_detected == "green":
                                                h_a = gaze
                                                robot_env._update(h_a)
                                                detected = True
                                                break

                            ### looking at goal
                        else:
                            if self.color_detected == "green":

                                if h_a == gaze:

                                    h_a = gaze
                                    robot_env._update(h_a)
                                    detected = True

                    print "h_a is at: ", h_a

                    robot_env._update(h_a)

                    detected = False



                    #generate rewards and new states
                    r_s1,r_r,r_d,_ = robot_env.step(r_a)
                    print "robot done:", r_d
                    robot.update_table(r_s1,r_a,r_r)
                    robot.env.s = r_s1

                    if not interactive:
                        h_s1,h_r,h_d,_ = human_env._step(h_a,r_a,r_d)
                        # print "updating human", h_s, h_s1, h_r
                        human.update_table(h_s1,h_a,h_r)
                        h_s = h_s1
                        human_rAll += h_r
                    r_s = r_s1
                    robot_rAll += r_r


                    if r_d == True:
                        #print "reward robot", robot_rAll
                        #print "reward human", human_rAll
                        print "YEAH WE ARE DONE"
                        if use_robot:
                            if robot_type == "icub":

                                robot_ctrl.set_gaze_target(GAZE_LOCATION[h_a])
                                time.sleep(1)
                                robot_ctrl.set_gaze_target(GAZE_LOCATION[5])
                                robot_ctrl.set_default_emotion(RobotEmotion(RobotEmotion.HAPPY))
                                ani = RobotAnimation(RobotAnimation.HEAD_NOD)
                                ani.repetitions = 3
                                robot_ctrl.set_head_animation(ani, blocking=True)
                                time.sleep(2)
                            elif robot_type == 'nao':
                                print "NAO SAY SOMETHING"
                                robot_ctrl.angleInterpolation(names, GAZE_LOCATION[5], timeLists, isAbsolute)
                                nao_tts.say('this is corret')
                                nao_tts.say('the cup is at the position were i wanted it')

                        break

                jList.append(j)
                robot_rList.append(robot_rAll)
                if not interactive:
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

                    heatmap2 = robot_heatmap.imshow(np.matrix.transpose(robot.Q), cmap='hot', interpolation='nearest',label ='robot q table')
                    plt.pause(0.005)


                ### TODO this is obsolete... delete?
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
                if not interactive:
                    heatmap = ax3.imshow(human.Q, cmap='hot', interpolation='nearest',label='human q tale')

                plt.pause(1.0)


                if first_vis: # plot the heatmap only for the first time
                    plt.colorbar(heatmap)

                    first_vis = False

                    plt.pause(0.0005)



        #print human.Q

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
    parser.add_argument('--robot_type', help="icub or nao",
                                 default='icub')
    parser.add_argument('--nao_ip', help="naos ip addres",
                                 default='127.0.0.1')
    parser.add_argument('-i','--interactive',help="interactive session. with real user!",
                     action="store_true")


    args = parser.parse_args()

    print args
    asdf = CommunicationLearning(agent=args.agent,nao_ip = args.nao_ip,interactive = args.interactive, robot_type=args.robot_type,exp_strategy= args.exp_strategy, fixed_cup_pos=args.fixed_cup_pos,fixed_goal_pos = args.fixed_goal_pos, online_vis = args.online_vis,
     num_episodes = args.num_episodes, reuse_qtable = args.reuse_qtable, num_of_simulation = args.num_of_simulation, visualisation= args.visualisation, file_name=args.file_name, use_robot = args.use_robot)
