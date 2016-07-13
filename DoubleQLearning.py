import tensorflow as tf
import cv2, sys, random, select
import copy
import threading
from threading import Thread
sys.path.append("WrappedGameCode/")

# whichever is imported "as game" will be used
import pong_fun as game
import dummy_game, tetris_fun
import argparse, numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import subprocess
from collections import deque
import logging
from logging.handlers import TimedRotatingFileHandler

indexing = {}

class DoubleDeepQLearning:
  def __init__(self, argsCallPassed, lock):
    self.argsPassed = copy.deepcopy(argsCallPassed)
    self.lock = lock
    self.t = 0
    indexing[threading.current_thread().name] = self.t
    self.fileName = threading.current_thread().name
    print "This is", self.fileName

    '''
    self.logger.debug('This message should go to the log file')
    self.logger.info('So should this')
    self.logger.warning('And this, too')
    '''
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)

    # create a file handler
    self.handler = logging.FileHandler('logs/' + self.fileName + '.log')
    self.handler.acquire()
    self.handler.setLevel(logging.DEBUG)
    self.handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'))

    # add the handlers to the logger
    self.logger.addHandler(self.handler)
    self.handler.release()
    self.logger.propagate = False
    self.logger.info('Hello baby')
    #self.playGame()

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

  def trainNetwork(self, s, readout_net1, readout_net2, readout_netb1,readout_netb2,sess):
  #def trainNetwork(self, s, readout_net1, readout_netb1, sess):
  #def trainNetwork(self, s, sess):
    # define the cost function
    self.s = s
    self.readout_net1 = readout_net1
    self.readout_net2 = readout_net2
    self.readout_netb1 = readout_netb1
    self.readout_netb2 = readout_netb2
    self.sess = sess

    [self.train_step_net1, self.y_net1, self.a_net1] = self.getTrainStep( self.readout_net1 )
    [self.train_step_net2, self.y_net2, self.a_net2] = self.getTrainStep( self.readout_net2 )
    
    [self.train_step_netb1, self.y_netb1, self.a_netb1] = self.getTrainStep( self.readout_netb1 )
    [self.train_step_netb2, self.y_netb2, self.a_netb2] = self.getTrainStep( self.readout_netb2 )
    
    # open up a game state to communicate with emulator
    #self.game = copy.deepcopy(game)
    self.game_state = game.GameState(threading.current_thread().name, self.lock)
    #self.game_state = game.GameState()

    # store the previous observations in replay memory
    self.D = deque()
    
    # printing
    self.a_file = open("logs_" + self.argsPassed.game_log_name + "/readout.txt", 'w')
    self.h_file = open("logs_" + self.argsPassed.game_log_name + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    self.do_nothing = np.zeros(self.argsPassed.action_count)
    self.do_nothing[0] = 1
    self.x_t, self.r_0, self.terminal = self.game_state.frame_step(self.do_nothing)
    self.x_t = cv2.cvtColor(cv2.resize(self.x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    self.ret, self.x_t = cv2.threshold(self.x_t, 1, 255, cv2.THRESH_BINARY)
    self.s_t = np.stack((self.x_t, self.x_t, self.x_t, self.x_t), axis = 2)
    '''
    if(self.argsPassed.figure):
      fig = plt.figure()
      plt.imshow(self.x_t.T)
      fig.savefig(self.argsPassed.figure_name)
    '''

    import time
    #time.sleep(5)
    # saving and loading networks
    self.saver = tf.train.Saver()
    self.sess.run(tf.initialize_all_variables())
    self.checkpoint = tf.train.get_checkpoint_state("saved_networks/")
    print self.checkpoint.model_checkpoint_path
    #self.saver.restore(sess, checkpoint.model_checkpoint_path)
    if self.checkpoint and self.checkpoint.model_checkpoint_path:
      self.saver.restore(self.sess, self.checkpoint.model_checkpoint_path)
      print "Successfully loaded:", self.checkpoint.model_checkpoint_path
    else:
      print "Could not find old network weights"
    
    self.epsilon = self.argsPassed.initial_epsilon
    
    self.net_flag = 0
    self.cnt = 0
    while True:
      # choose an action epsilon greedily
      if self.net_flag == 0:
        self.readout_t = self.readout_net1.eval(feed_dict = {s : [self.s_t]})[0]
        self.readout_bt = self.readout_netb1.eval(feed_dict = {s : [self.s_t]})[0]
      else:
        self.readout_t = self.readout_net2.eval(feed_dict = {s : [self.s_t]})[0]
        self.readout_bt = self.readout_netb2.eval(feed_dict = {s : [self.s_t]})[0]

      self.a_t = np.zeros([self.argsPassed.action_count])
      self.a_bt = np.zeros([self.argsPassed.action_count])
      
      self.action_index = 0
      if random.random() <= self.epsilon or self.t <= self.argsPassed.observation_count:
        self.action_index = random.randrange(self.argsPassed.action_count)
        self.a_t[self.action_index] = 1
      else:
        self.action_index = np.argmax(self.readout_t)
        self.a_t[self.action_index] = 1

      self.action_indexb = 0
      if random.random() <= self.epsilon or self.t <= self.argsPassed.observation_count:
        self.action_indexb = random.randrange(self.argsPassed.action_count)
        self.a_bt[self.action_indexb] = 1
      else:
        self.action_indexb = np.argmax(self.readout_bt)
        self.a_bt[self.action_indexb] = 1

      # scale down epsilon
      if self.epsilon > self.argsPassed.final_epsilon and self.t > self.argsPassed.observation_count:
        self.epsilon -= (self.argsPassed.initial_epsilon - self.argsPassed.final_epsilon) / self.argsPassed.explore_frames
      
      for i in range(0, self.argsPassed.k):
        # run the selected action and observe next state and reward
        self.x_t1_col, self.r_t, self.terminal = self.game_state.frame_step(self.a_t)
        self.x_t1 = cv2.cvtColor(cv2.resize(self.x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        self.ret, self.x_t1 = cv2.threshold(self.x_t1, 1, 255, cv2.THRESH_BINARY)
        self.x_t1 = np.reshape(self.x_t1, (80, 80, 1))
        self.s_t1 = np.append( self.s_t[:,:,1:], self.x_t1, axis = 2)
        # store the transition in D
        '''
        if t==5:
          self.fig1 = plt.figure()
          plt.imshow(self.s_t[:,:,0].T)
          self.fig1.savefig('trrr_1.png')

          self.fig2 = plt.figure()
          plt.imshow( self.s_t[:,:,1].T)
          self.fig2.savefig('trrr_2.png')

          self.fig3 = plt.figure()
          plt.imshow( self.s_t[:,:,2].T)
          self.fig3.savefig('trrr_3.png')

          self.fig4 = plt.figure()
          plt.imshow( self.s_t[:,:,3].T)
          self.fig4.savefig('trrr_4.png')

          time.sleep(5)
        '''
        #self.D.append((s_t, a_t, r_t,a_bt,r_bt, s_t1, terminal))
        self.D.append(( self.s_t, self.a_t, self.r_t, self.s_t1, self.terminal))
        if len(self.D) > self.argsPassed.replay_memory:
          self.D.popleft()
      
      # only train if done observing
      if self.t > self.argsPassed.observation_count:
        
        self.cnt = self.cnt + 1
        if self.argsPassed.verbosity:
          print "??"
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
        if self.cnt % self.argsPassed.switch_net == 0:
          if self.net_flag == 0:
            self.net_flag = 1
          else:
            self.net_flag = 0
          #print 'SwitchState'
      self.s_t = self.s_t1
      self.t += 1

      # print info
      self.state = ""
      if self.t <= self.argsPassed.observation_count:
        self.state = "observe"
        if self.argsPassed.verbosity:
          print threading.current_thread().name, self.state, self.t
      elif self.t > self.argsPassed.observation_count and self.t <= self.argsPassed.observation_count + self.argsPassed.explore_frames:
        self.state = "explore"
        if self.argsPassed.verbosity:
          print threading.current_thread().name, self.state, self.t
      else:
        self.state = "train"
        if self.argsPassed.verbosity:
          print threading.current_thread().name, self.state, self.t
      indexing[threading.current_thread().name] = self.t
      if self.argsPassed.verbosity:
        print "Assigned to", threading.current_thread().name, "value", indexing[threading.current_thread().name]
      #if self.r_t != 0:
      #  print "TIMESTEP", t, "/ STATE", state, "/ LINES", self.game_state.total_lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

      '''
      if r_bt != 0:
        print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_indexb, "/ REWARD", r_bt, "/ Q_MAX %e" % np.max(readout_bt)
      '''

      if self.r_t != 0:
        print threading.current_thread().name, "TIMESTEP", self.t, "/ STATE", self.state, "/ EPSILON", self.epsilon, "/ ACTION", self.action_index, "/ REWARD", self.r_t, "/ Q_MAX %e" % np.max(self.readout_t)
      # write info to files
      '''
      if t % 10000 <= 100:
        a_file.write(",".join([str(x) for x in readout_t]) + '\n')
        h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
        cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
      '''
      if(self.t and self.t % self.argsPassed.terminate_prompt == 0):
        print "TIMESTEP = " + str(self.t)
        if(self.yes_or_no("Want to terminate the code")):
          return
        else:
          continue

  def playGame(self):
    self.sess = tf.InteractiveSession()
    self.s = tf.placeholder("float", [None, 80, 80, 4])
    #Player 1
    self.readout_net1, self.h_fc1_net1 = self.createNetwork(self.s)
    self.readout_netb1, self.h_fc1_netb1 = self.createNetwork(self.s)
    
    #Player 2
    self.readout_net2, self.h_fc1_net2 = self.createNetwork(self.s)
    self.readout_netb2, self.h_fc1_netb2 = self.createNetwork(self.s)

    self.trainNetwork(self.s, self.readout_net1, self.readout_net2, self.readout_netb1, self.readout_netb2, self.sess)
    #self.trainNetwork(self.s, self.sess)
    #self.trainNetwork(self.s, self.readout_net1, self.readout_netb1, self.sess)
