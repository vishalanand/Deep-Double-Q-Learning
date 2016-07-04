import tensorflow as tf
import cv2, sys, random, select
import copy
sys.path.append("Wrapped Game Code/")

# whichever is imported "as game" will be used
import pong_fun_DoublePlayer as game
import dummy_game, tetris_fun
import argparse, numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import subprocess
from collections import deque

class DoubleDeepQLearning:
  def __init__(self, argsCallPassed):
    self.argsPassed = copy.deepcopy(argsCallPassed)
    self.playGame()

  def weight_variable(self, shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

  def bias_variable(self, shape):
    return tf.Variable(tf.constant(0.01, shape = shape))

  def conv2d(self, x, W, stride):
    #VALID means no padding, SAME has same spatial features
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

  def max_pool_2x2(self, x):
    #VALID means no padding, SAME has same spatial features
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

  def createNetwork(self, s_net):
    # network weights
    self.W_conv1 = self.weight_variable([8, 8, 4, 32])
    self.b_conv1 = self.bias_variable([32])
    
    self.W_conv2 = self.weight_variable([4, 4, 32, 64])
    self.b_conv2 = self.bias_variable([64])
    
    self.W_conv3 = self.weight_variable([3, 3, 64, 64])
    self.b_conv3 = self.bias_variable([64])
    
    self.W_fc1 = self.weight_variable([1600, 512])
    self.b_fc1 = self.bias_variable([512])
    
    self.W_fc2 = self.weight_variable([512, self.argsPassed.action_count])
    self.b_fc2 = self.bias_variable([self.argsPassed.action_count])

    # input layer
    #s_net = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers (with weights and biases)
    self.h_conv1 = tf.nn.relu(self.conv2d(s_net, self.W_conv1, 4) + self.b_conv1)
    self.h_pool1 = self.max_pool_2x2(self.h_conv1)
    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2)
    #h_pool2 = self.max_pool_2x2(h_conv2)
    self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)
    #h_pool3 = self.max_pool_2x2(h_conv3)
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 1600])
    self.h_fc1_net = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

    # readout layer
    self.readout_net = tf.matmul(self.h_fc1_net, self.W_fc2) + self.b_fc2

    return self.readout_net, self.h_fc1_net

  def getTrainStep(self, readOut):
    a_train = tf.placeholder("float", [None, self.argsPassed.action_count])
    y_train = tf.placeholder("float", [None])
    readout_action_train = tf.reduce_sum(tf.mul(readOut, a_train), reduction_indices = 1)
    cost_train = tf.reduce_mean(tf.square(y_train - readout_action_train))
    trainStep = tf.train.AdamOptimizer(1e-5).minimize(cost_train)
    return [trainStep, y_train, a_train]

  def yes_or_no(self, question):
    print question + " (y/n): "
    i, o, e = select.select( [sys.stdin], [], [], 2 )
    if (i):
      reply = sys.stdin.readline().strip()
      if reply[0] == 'y':
        return True
      if reply[0] == 'n':
        return False
      else:
        return self.yes_or_no("Uhhhh... please enter ")
    else:
      return False

  def trainNetwork(self, s, readout_net1,readout_net2, readout_netb1,readout_netb2,sess):
    # define the cost function
    [train_step_net1, y_net1, a_net1] = self.getTrainStep( readout_net1 )
    [train_step_net2, y_net2, a_net2] = self.getTrainStep( readout_net2 )
    
    [train_step_netb1, y_netb1, a_netb1] = self.getTrainStep( readout_netb1 )
    [train_step_netb2, y_netb2, a_netb2] = self.getTrainStep( readout_netb2 )
    
    # open up a game state to communicate with emulator
    #self.game = copy.deepcopy(game)
    self.game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()
    
    # printing
    a_file = open("logs_" + self.argsPassed.game_log_name + "/readout.txt", 'w')
    h_file = open("logs_" + self.argsPassed.game_log_name + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(self.argsPassed.action_count)
    do_nothing[0] = 1
    x_t, r_0,r_1, terminal = self.game_state.frame_step(do_nothing,do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    '''
    if(self.argsPassed.figure):
      fig = plt.figure()
      plt.imshow(x_t.T)
      fig.savefig(self.argsPassed.figure_name)
    '''

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
    
    epsilon = self.argsPassed.initial_epsilon
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

      a_t = np.zeros([self.argsPassed.action_count])
      a_bt = np.zeros([self.argsPassed.action_count])
      
      action_index = 0
      if random.random() <= epsilon or t <= self.argsPassed.observation_count:
        action_index = random.randrange(self.argsPassed.action_count)
        a_t[action_index] = 1
      else:
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1

      action_indexb = 0
      if random.random() <= epsilon or t <= self.argsPassed.observation_count:
        action_indexb = random.randrange(self.argsPassed.action_count)
        a_bt[action_indexb] = 1
      else:
        action_indexb = np.argmax(readout_bt)
        a_bt[action_indexb] = 1

      # scale down epsilon
      if epsilon > self.argsPassed.final_epsilon and t > self.argsPassed.observation_count:
        epsilon -= (self.argsPassed.initial_epsilon - self.argsPassed.final_epsilon) / self.argsPassed.explore_frames
      
      for i in range(0, self.argsPassed.k):
        # run the selected action and observe next state and reward
        x_t1_col, r_t,r_bt, terminal = self.game_state.frame_step(a_t,a_bt)
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
        if len(D) > self.argsPassed.replay_memory:
          D.popleft()
      
      # only train if done observing
      if t > self.argsPassed.observation_count:
        
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
        if cnt % self.argsPassed.switch_net == 0:
          if net_flag == 0:
            net_flag = 1
          else:
            net_flag = 0
          #print 'SwitchState'
      s_t = s_t1
      t += 1

      # print info
      state = ""
      if t <= self.argsPassed.observation_count:
        state = "observe"
      elif t > self.argsPassed.observation_count and t <= self.argsPassed.observation_count + self.argsPassed.explore_frames:
        state = "explore"
      else:
        state = "train"
      #if r_t != 0:
      #  print "TIMESTEP", t, "/ STATE", state, "/ LINES", self.game_state.total_lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

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
      if(t and t % self.argsPassed.terminate_prompt == 0):
        print "TIMESTEP = " + str(t)
        if(self.yes_or_no("Want to terminate the code")):
          return
        else:
          continue

  def playGame(self):
    self.sess = tf.InteractiveSession()
    self.s = tf.placeholder("float", [None, 80, 80, 4])
    #Player 1
    self.readout_net1, self.h_fc1_net1 = self.createNetwork(self.s)
    self.readout_net2, self.h_fc1_net2 = self.createNetwork(self.s)
    
    #Player 2
    self.readout_netb1, self.h_fc1_netb1 = self.createNetwork(self.s)
    self.readout_netb2, self.h_fc1_netb2 = self.createNetwork(self.s)

    self.trainNetwork(self.s, self.readout_net1,self.readout_net2, self.readout_netb1, self.readout_netb2, self.sess)
