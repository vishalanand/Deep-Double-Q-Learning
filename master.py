import tensorflow as tf
import cv2, sys, random, select
sys.path.append("Wrapped Game Code/")

# whichever is imported "as game" will be used
import pong_fun as game

import dummy_game, tetris_fun
import argparse, numpy as np
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread
import subprocess
import deepDoubleQLearning

def main():
  t1 = Thread(target=deepDoubleQLearning.main(), args=args)
  t2 = Thread(target=deepDoubleQLearning, args=args)
  t3 = Thread(target=deepDoubleQLearning, args=args)
  t4 = Thread(target=deepDoubleQLearning, args=args)
  t5 = Thread(target=deepDoubleQLearning, args=args)
  t6 = Thread(target=deepDoubleQLearning, args=args)
  t7 = Thread(target=deepDoubleQLearning, args=args)
  t8 = Thread(target=deepDoubleQLearning, args=args)


  t1.start()
  t2.start()
  t3.start()
  t4.start()
  t5.start()
  t6.start()
  t7.start()
  t8.start()

  t1.join()
  t2.join()
  t3.join()
  t4.join()
  t5.join()
  t6.join()
  t7.join()
  t8.join()

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