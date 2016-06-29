import tensorflow as tf
import cv2, sys, random, select
sys.path.append("Wrapped Game Code/")

# whichever is imported "as game" will be used
import pong_fun as game

import dummy_game, tetris_fun
import argparse, numpy as np
import matplotlib.pyplot as plt
from collections import deque

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    #VALID means no padding, SAME has same spatial features
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    #VALID means no padding, SAME has same spatial features
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork(s_net):
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, args.action_count])
    b_fc2 = bias_variable([args.action_count])

    # input layer
    #s_net = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers (with weights and biases)
    h_conv1 = tf.nn.relu(conv2d(s_net, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1_net = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout_net = tf.matmul(h_fc1_net, W_fc2) + b_fc2

    return readout_net, h_fc1_net

def getTrainStep( readOut ):
    a_train = tf.placeholder("float", [None, args.action_count])
    y_train = tf.placeholder("float", [None])
    readout_action_train = tf.reduce_sum(tf.mul(readOut, a_train), reduction_indices = 1)
    cost_train = tf.reduce_mean(tf.square(y_train - readout_action_train))
    trainStep = tf.train.AdamOptimizer(1e-5).minimize(cost_train)
    return [trainStep, y_train, a_train]

def ask_ok(prompt, retries=4, complaint='Yes or no, please!'):
    while True:
        ok = input(prompt)
        ok.lower()
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise IOError('uncooperative user')
        print(complaint)

def yes_or_no(question):
    #reply = str(raw_input(question+' (y/n): ')).lower().strip()
    #i, o, e = select.select( [sys.stdin], [], [], 2 )
    print question + " (y/n): "
    i, o, e = select.select( [sys.stdin], [], [], 2 )
    if (i):
        reply = sys.stdin.readline().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False
        else:
            return yes_or_no("Uhhhh... please enter ")
    else:
        return False

def trainNetwork(s, readout_net1,readout_net2, readout_netb1,readout_netb2,sess):
    # define the cost function
    [train_step_net1, y_net1, a_net1] = getTrainStep( readout_net1 )
    [train_step_net2, y_net2, a_net2] = getTrainStep( readout_net2 )
    
    [train_step_netb1, y_netb1, a_netb1] = getTrainStep( readout_netb1 )
    [train_step_netb2, y_netb2, a_netb2] = getTrainStep( readout_netb2 )
    
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()
    
    # printing
    a_file = open("logs_" + args.game_log_name + "/readout.txt", 'w')
    h_file = open("logs_" + args.game_log_name + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(args.action_count)
    do_nothing[0] = 1
    x_t, r_0,r_1, terminal = game_state.frame_step(do_nothing,do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    if(args.figure):
        fig = plt.figure()
        plt.imshow(x_t.T)
        fig.savefig(args.figure_name)

    import time
    #time.sleep(5)
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/")
    print checkpoint.model_checkpoint_path
    #saver.restore(sess, checkpoint.model_checkpoint_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    
    epsilon = args.initial_epsilon
    t = 0
    
    net_flag = 0
    cnt = 0
    while True:
        # choose an action epsilon greedily
        if net_flag == 0:
            readout_t = readout_net1.eval(feed_dict = {s : [s_t]})[0]
            readout_bt = readout_netb1.eval(feed_dict = {s : [s_t]})[0]

        else:
            readout_t = readout_net2.eval(feed_dict = {s : [s_t]})[0]
            readout_bt = readout_netb2.eval(feed_dict = {s : [s_t]})[0]

        a_t = np.zeros([args.action_count])
        a_bt = np.zeros([args.action_count])
        
        action_index = 0
        if random.random() <= epsilon or t <= args.observation_count:
            action_index = random.randrange(args.action_count)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        action_indexb = 0
        if random.random() <= epsilon or t <= args.observation_count:
            action_indexb = random.randrange(args.action_count)
            a_bt[action_indexb] = 1
        else:
            action_indexb = np.argmax(readout_bt)
            a_bt[action_indexb] = 1

        # scale down epsilon
        if epsilon > args.final_epsilon and t > args.observation_count:
            epsilon -= (args.initial_epsilon - args.final_epsilon) / args.explore_frames
        
        for i in range(0, args.k):
            # run the selected action and observe next state and reward
            x_t1_col, r_t,r_bt, terminal = game_state.frame_step(a_t,a_bt)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append( s_t[:,:,1:],x_t1, axis = 2)
            # store the transition in D
            '''
            if t==5:
                fig1 = plt.figure()
                plt.imshow(s_t[:,:,0].T)
                fig1.savefig('trrr_1.png')

                fig2 = plt.figure()
                plt.imshow(s_t[:,:,1].T)
                fig2.savefig('trrr_2.png')

                fig3 = plt.figure()
                plt.imshow(s_t[:,:,2].T)
                fig3.savefig('trrr_3.png')

                fig4 = plt.figure()
                plt.imshow(s_t[:,:,3].T)
                fig4.savefig('trrr_4.png')

                time.sleep(5)
            '''
            D.append((s_t, a_t, r_t,a_bt,r_bt, s_t1, terminal))
            if len(D) > args.replay_memory:
                D.popleft()
        
        # only train if done observing
        if t > args.observation_count:
            
            cnt = cnt+1
            # sample a minibatch to train on
            '''
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]

            ab_batch = [d[3] for d in minibatch]
            rb_batch = [d[4] for d in minibatch]

            s_j1_batch = [d[5] for d in minibatch]

            y_batch = []
            yb_batch = []

            if net_flag == 0:
                readout_j1_batch = readout_net2.eval(feed_dict = {s : s_j1_batch})
                readoutb_j1_batch = readout_netb2.eval(feed_dict = {s : s_j1_batch})
            else:
                readout_j1_batch = readout_net1.eval(feed_dict = {s : s_j1_batch})
                readoutb_j1_batch = readout_netb1.eval(feed_dict = {s : s_j1_batch})

            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][6]:
                    y_batch.append(r_batch[i])
                    yb_batch.append(rb_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    yb_batch.append(rb_batch[i] + GAMMA * np.max(readoutb_j1_batch[i]))

            # perform gradient step
            if net_flag == 0:
                train_step_net2.run( feed_dict = {y_net2 : y_batch,a_net2 : a_batch, s : s_j_batch} )
                train_step_netb2.run( feed_dict = {y_netb2 : yb_batch,a_netb2 : ab_batch, s : s_j_batch} )
            else:
                train_step_net1.run( feed_dict = {y_net1 : y_batch,a_net1 : a_batch, s : s_j_batch} )
                train_step_netb1.run( feed_dict = {y_netb1 : yb_batch,a_netb1 : ab_batch, s : s_j_batch} )
            '''
        # update the old values
            if cnt % args.switch_net == 0:
                if net_flag == 0:
                    net_flag = 1
                else:
                    net_flag = 0
                #print 'SwitchState'
        s_t = s_t1
        t += 1

        # print info
        state = ""
        if t <= args.observation_count:
            state = "observe"
        elif t > args.observation_count and t <= args.observation_count + args.explore_frames:
            state = "explore"
        else:
            state = "train"
        #if r_t != 0:
        #    print "TIMESTEP", t, "/ STATE", state, "/ LINES", game_state.total_lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

        if r_bt != 0:
            print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_indexb, "/ REWARD", r_bt, "/ Q_MAX %e" % np.max(readout_bt)

        if r_t != 0:
            print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''
        if(t and t % args.terminate_prompt == 0):
            print "TIMESTEP = " + str(t)
            if(yes_or_no("Want to terminate the code")):
                return
            else:
                continue

def playGame():
    sess = tf.InteractiveSession()
    s = tf.placeholder("float", [None, 80, 80, 4])
    #Player 1
    readout_net1, h_fc1_net1 = createNetwork(s)
    readout_net2, h_fc1_net2 = createNetwork(s)
    
    #Player 2
    readout_netb1, h_fc1_netb1 = createNetwork(s)
    readout_netb2, h_fc1_netb2 = createNetwork(s)

    trainNetwork(s, readout_net1,readout_net2, readout_netb1,readout_netb2, sess)
    

def main():
    playGame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-game", "--game-log-name", help="Name of the game being played for log files", default='pong')
    parser.add_argument("-actions", "--action-count", help="Number of valid actions", type=int, default=3)
    parser.add_argument("-gamma", "--gamma-value", help="Decay rate of past observations", default=0.99)
    parser.add_argument("-obs", "--observation-count", help="timesteps to observe before training", type=int, default=2500)
    parser.add_argument("-explore", "--explore-frames", help="Number of frames over which to anneal epsilon", type=int, default=220000)
    parser.add_argument("-initialeps", "--initial-epsilon", help="initial value of epsilon", default=0.70)
    parser.add_argument("-decay", "--decay", help="decrease in the epsilon value per timestep", default=0.01)
    parser.add_argument("-finaleps", "--final-epsilon", help="final value of epsilon", default=0.07)
    parser.add_argument("-replay", "--replay-memory", help="number of previous transitions to remember", type=int, default=250000)
    parser.add_argument("-bat", "--mini-batch", help="Size of minibatch", type=int, default=32)
    parser.add_argument("-k", "--k", help="only select an action every Kth frame, repeat prev for others", type=int, default=2)
    parser.add_argument("-switchnet", "--switch-net", help="Set a lower bound for word frequencies", type=int, default=10)
    parser.add_argument("-fig", "--figure", help="save figure", action="store_true")
    parser.add_argument("-figname", "--figure-name", help="figure name", default="trrr.png")
    parser.add_argument("-terminateprompt", "--terminate-prompt", help="terminate step prompt ", type=int, default=1000)
    parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")
    parser.add_argument("-arg", "--arguments", help="print out the arguments entered", action="store_true")
    args = parser.parse_args()
    
    if (args.arguments):
        print "args.game_log_name"      + "\t" + args.game_log_name
        print "args.action_count"       + "\t" + str(args.action_count)
        print "args.gamma_value"        + "\t" + str(args.gamma_value)
        print "args.observation_count"  + "\t" + str(args.observation_count)
        print "args.explore_frames"     + "\t" + str(args.explore_frames)
        print "args.initial_epsilon"    + "\t" + str(args.initial_epsilon)
        print "args.decay"              + "\t" + str(args.decay)
        print "args.final_epsilon"      + "\t" + str(args.final_epsilon)
        print "args.replay_memory"      + "\t" + str(args.replay_memory)
        print "args.mini_batch"         + "\t" + str(args.mini_batch)
        print "args.k"                  + "\t" + str(args.k)
        print "args.switch_net"         + "\t" + str(args.switch_net)
        print "args.figure"             + "\t" + str(args.figure)
        print "args.figure_name"        + "\t" + str(args.figure_name)
        print "args.terminate_prompt"   + "\t" + str(args.terminate_prompt)
        print "args.verbosity"          + "\t" + str(args.verbosity)
        print "args.arguments"          + "\t" + str(args.arguments) + "\n"
    main()
    #numpy.savetxt("logs/"+"W1.csv", W1, delimiter=",")
    #numpy.savetxt("logs/"+"W2.csv", W2, delimiter=",")
