#import tensorflow as tf
import cv2
import sys
import numpy, argparse, os, re, random, math, copy
sys.path.append("Wrapped Game Code/")
import pong_fun as game# whichever is imported "as game" will be used
import dummy_game
import tetris_fun
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def trainNetwork():
    print "The game is "
    #print args.game_log_name

def playGame():
    print "Again"
    trainNetwork()
    
def main():
    #print args.game_log_name
    playGame()

if __name__ == "__main__":
    main()
    parser = argparse.ArgumentParser()
    parser.add_argument("-game", "--game-log-name", help="Name of the game being played for log files", default='pong')
    parser.add_argument("-actions", "--action-count", help="Number of valid actions", type=int, default=3)
    parser.add_argument("-gamma", "--gamma-value", help="Decay rate of past observations", default=0.99)
    parser.add_argument("-obs", "--observation-count", help="timesteps to observe before training", type=int, default=2500)
    parser.add_argument("-explore", "--explore-frames", help="Number of frames over which to anneal epsilon", type=int, default=220000)
    parser.add_argument("-finaleps", "--final-epsilon", help="final value of epsilon", default=0.07)
    parser.add_argument("-initialeps", "--initial-epsilon", help="initial value of epsilon", default=0.07)
    parser.add_argument("-replay", "--replay-memory", help="number of previous transitions to remember", type=int, default=250000)
    parser.add_argument("-bat", "--mini-batch", help="Size of minibatch", type=int, default=32)
    parser.add_argument("-k", "--k", help="only select an action every Kth frame, repeat prev for others", type=int, default=2)
    parser.add_argument("-switchnet", "--switch-net", help="Set a lower bound for word frequencies", type=int, default=10)
    parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")
    parser.add_argument("-arg", "--arguments", help="print out the arguments entered", action="store_true")
    args = parser.parse_args()
    
    if (args.arguments):
        print "args.game_log_name", args.game_log_name
        print "args.action_count", args.action_count
        print "args.gamma_value", args.gamma_value
        print "args.observation_count", args.observation_count
        print "args.explore_frames", args.explore_frames
        print "args.initial_epsilon", args.initial_epsilon
        print "args.final_epsilon", args.final_epsilon
        print "args.replay_memory", args.replay_memory
        print "args.mini_batch", args.mini_batch
        print "args.k", args.k
        print "args.switch_net", args.switch_net
        print "args.verbosity", args.verbosity
        print "args.arguments", args.arguments, "\n"
    