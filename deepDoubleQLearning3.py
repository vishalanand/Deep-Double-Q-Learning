
import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import pong_fun as game# whichever is imported "as game" will be used
import dummy_game
import tetris_fun
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

GAME = 'pong' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 2500 # timesteps to observe before training
EXPLORE = 220000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.07 # final value of epsilon
INITIAL_EPSILON = 0.07 #0.52 starting value of epsilon
REPLAY_MEMORY = 250000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 2 # only select an action every Kth frame, repeat prev for others
SWITCH_NET = 10

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
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

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    #s_net = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
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
    a_train = tf.placeholder("float", [None, ACTIONS])
    y_train = tf.placeholder("float", [None])
    readout_action_train = tf.reduce_sum(tf.mul(readOut, a_train), reduction_indices = 1)
    cost_train = tf.reduce_mean(tf.square(y_train - readout_action_train))
    trainStep = tf.train.AdamOptimizer(1e-5).minimize(cost_train)
    return [trainStep, y_train, a_train]
    
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
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0,r_1, terminal = game_state.frame_step(do_nothing,do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    #fig = plt.figure()
    #plt.imshow(x_t)
    #fig.savefig('trrr.png')

    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    fig = plt.figure()
    plt.imshow(x_t.T)
    fig.savefig('trrr.png')
    import time
    time.sleep(5) 
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

    
    epsilon = INITIAL_EPSILON
    t = 0
    
    net_flag = 0
    cnt = 0
    while "pigs" != "fly":
        # choose an action epsilon greedily
        if net_flag == 0:
            readout_t = readout_net1.eval(feed_dict = {s : [s_t]})[0]
            readout_bt = readout_netb1.eval(feed_dict = {s : [s_t]})[0]

        else:
            readout_t = readout_net2.eval(feed_dict = {s : [s_t]})[0]
            readout_bt = readout_netb2.eval(feed_dict = {s : [s_t]})[0]

        a_t = np.zeros([ACTIONS])
        a_bt = np.zeros([ACTIONS])
        
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        action_indexb = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_indexb = random.randrange(ACTIONS)
            a_bt[action_indexb] = 1
        else:
            action_indexb = np.argmax(readout_bt)
            a_bt[action_indexb] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        
        for i in range(0, K):
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
            if len(D) > REPLAY_MEMORY:
                D.popleft()
        
        # only train if done observing
        if t > OBSERVE:
            
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
			
            if cnt % SWITCH_NET == 0:
                if net_flag == 0:
                    net_flag = 1
                else:
                    net_flag = 0
                #print 'SwitchState'
        s_t = s_t1
        t += 1

        '''
        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-3dqn', global_step = t)
        '''
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
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

def playGame():
    sess = tf.InteractiveSession()
    s = tf.placeholder("float", [None, 80, 80, 4])
    readout_net1, h_fc1_net1 = createNetwork(s)
    readout_net2, h_fc1_net2 = createNetwork(s)
    
    readout_netb1, h_fc1_netb1 = createNetwork(s)
    readout_netb2, h_fc1_netb2 = createNetwork(s)

    trainNetwork(s, readout_net1,readout_net2, readout_netb1,readout_netb2, sess)
    

def main():
    playGame()

if __name__ == "__main__":
    main()
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--deep-neural", help="Deep Neural Network size", type=int, default=50)
    parser.add_argument("-window", "--window-size", help="Window size of words(5)", type=int, default=5)
    parser.add_argument("-word", "--word-size", help="Word vector size(50)", type=int, default=50)
    parser.add_argument("-iter", "--iteration-count", help="iteration count upperbound", type=int, default=5000)
    parser.add_argument("-eps", "--epsilon", help="epsilon value for termination condition", type=int, default=0.001)
    parser.add_argument("-in", "--input-file", help="corpus to read from in order to create the word vector", default="in.txt")
    parser.add_argument("-vocab", "--vocabulary", help="Save the vocabulary file to entered location", default="build_vocab.txt")
    parser.add_argument("-vocabc", "--vocabulary-count", help="Save the vocabulary file with the frequencies to entered location", default="build_vocab_count.txt")
    parser.add_argument("-vec", "--vector-output", help="Save the word vectors to the given file", default="word_vector_output.txt")
    parser.add_argument("-uc", "--uppercase", help="Let words remain in uppercase", action="store_true")
    parser.add_argument("-wlimit", "--word-limit", help="Set a lower bound for word frequencies", type=int, default=0)
    parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")
    parser.add_argument("-arguments", "--arguments", help="print out the arguments entered", action="store_true")
    
    args = parser.parse_args()
    
    if (args.arguments):
        print "args.deep_neural", args.deep_neural
        print "args.window_size", args.window_size
        print "args.word_size", args.word_size
        print "args.iteration_count", args.iteration_count
        print "args.input_file", args.input_file
        print "args.epsilon", args.epsilon
        print "args.input_file", args.input_file
        print "args.vocabulary", args.vocabulary
        print "args.vector_output", args.vector_output
        print "args.uppercase", args.uppercase
        print "args.word_limit", args.word_limit
        print "args.verbosity", args.verbosity, "\n"
    
    weight_init()
    word_vocab_build()
    word_vec_init()
    word_vec_process()
    word_vec_print()
    """
    print "The dimensions of the variable W1 is",
    print W1.__class__
    print W1.shape
    print "The dimensions of the variable W2 is",
    print W2.__class__
    print W2.shape
    """
    numpy.savetxt("logs/"+"W1.csv", W1, delimiter=",")
    numpy.savetxt("logs/"+"W2.csv", W2, delimiter=",")
