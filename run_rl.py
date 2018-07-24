import argparse
import os
import random
import shutil
import time
from collections import deque
from itertools import count

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import dqn
import game.flappy_bird as flappy
from utils.average_meter import AverageMeter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='DL Benchmark based on Pytorch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-p", "--play", action="store_true",
                    help="Play game if set(need to use with -m), otherwise train model")
parser.add_argument("--display", action="store_true",
                    help="Display game screen (True if -p is specified)")
parser.add_argument("--mem", default=5000, type=int,
                    help="Maximum size of experience repaly memory")
parser.add_argument('--init_e', default=0.1, type=float,
                    help='Initial epsilon for epsilon-greedy exploration')
parser.add_argument('--final_e', default=0.0001, type=float,
                    help='Final epsilon for epsilon-greedy exploration')
parser.add_argument('--observation', default=5000, type=int,
                    help='Number of observation before training')
parser.add_argument('--exploration', default=1000000, type=int,
                    help='Number of exploration over which to anneal epsilon')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='Decay rate of past observations')

parser.add_argument('--start-episode', default=0, type=int,
                    help='Manual episode to start training')
parser.add_argument('--episodes', default=30000, type=int,
                    help='Number of total episodes to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='Mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='Learning rate')
parser.add_argument('--weight-decay', default=0, type=float,
                    help='Weight decay')

parser.add_argument('--print-freq', default=100, type=int,
                    help='Print frequency')
parser.add_argument('--save-freq', default=20000, type=int,
                    help='Time step interval to save model')
parser.add_argument('-m', '--model', default='', type=str,
                    help='Load model at PATH')
parser.add_argument('--save-dir', default='./saves', type=str,
                    help='The directory used to save the trained models')
parser.add_argument('--data-dir', default='./data', type=str,
                    help='Data directory')

INPUT_SIZE = (80, 80, 4)
ACTIONS = 2


def getCriterion():
    """Criterion (Loss function) for training and validation"""
    return nn.MSELoss()


def getOptimizer(model, lr, weight_decay):
    """Optimizer for training"""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def getRandomAction(actions):
    action = [0] * actions
    # action_index = random.randrange(actions)
    action_index = 0 if random.random() < 0.9 else 1
    action[action_index] = 1
    return action


def getModelAction(model, state, actions):
    with torch.no_grad():
        state = state.unsqueeze(0).cuda()
        out = model(state)
        action_index = torch.argmax(out).item()
        action = [0] * actions
        action[action_index] = 1
        return action


def getAction(model, state, actions, epsilon):
    if random.random() < epsilon:
        return getRandomAction(actions)
    return getModelAction(model, state, actions)


def train(game, model, criterion, optimizer):
    """Train DQN using Q-Learning"""
    # Switch to train mode
    model.train()

    replay = dqn.ReplayMem(args.mem, INPUT_SIZE)
    state = replay.getCurrentState()

    # Observation
    for t in range(args.observation):
        action = getRandomAction(ACTIONS)
        frame, reward, terminal = game.frame_step(action)
        replay.storeTransition(frame, action, reward, terminal)

        # Print Training Information
        if t % args.print_freq == 0:
            print("Timestep [{0} (observe)]".format(t))

    epsilon = args.init_e
    episode = args.start_episode
    for t in count(0, 1):
        action = getAction(model, state, ACTIONS, epsilon)

        # Do one step, get reward
        frame, reward, terminal = game.frame_step(action)
        replay.storeTransition(frame, action, reward, terminal)
        state = replay.getCurrentState()

        # Train model
        current_states, actions, next_states, rewards, terminals = replay.getBatch(
            args.batch_size)
        current_states, actions, next_states = current_states.cuda(
        ), actions.cuda(), next_states.cuda()

        with torch.no_grad():
            next_q = model(next_states)
            max_next_q = torch.max(next_q, dim=1)[0].cpu()

            y = torch.tensor(rewards, dtype=torch.float)
            for i in range(args.batch_size):
                if not terminals[i]:
                    y[i] += args.gamma * max_next_q[i].item()
            y = y.cuda()

        q = model(current_states)
        q = torch.sum(torch.mul(actions, q), dim=1)

        loss = criterion(q, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print Training Information
        if t % args.print_freq == 0:
            phase = "explore"
            if t > args.observation + args.exploration:
                phase = "train"

            print("Timestep [{0} ({1})]: "
                  "Episode {episode}\t"
                  "Epsilon {epsilon:.5f}".format(
                      t, phase,
                      episode=episode, epsilon=epsilon
                  ))

        # scale down epsilon
        if epsilon > args.final_e:
            epsilon -= (args.init_e - args.final_e) / args.exploration

        if t % args.save_freq == 0:
            saveCheckpoint({
                'episode': episode,
                'epsilon': epsilon,
                'state_dict': model.state_dict(),
            }, False, filename=os.path.join(args.save_dir, 'model_{}_{}.pth.tar'.format(episode, t)))

        if terminal:
            episode += 1

        if episode > args.episodes:
            break


def play(game, model):
    model.eval()

    replay = dqn.ReplayMem(0, INPUT_SIZE)
    state = replay.getCurrentState()

    terminal = False
    while not terminal:
        action = getModelAction(model, state, ACTIONS)
        frame, reward, terminal = game.frame_step(action)
        replay.storeTransition(frame, action, reward, terminal)
        state = replay.getCurrentState()


def saveCheckpoint(state, is_better, filename='model.pth.tar'):
    """Save the training model"""
    torch.save(state, filename)
    dirname = os.path.dirname(filename)
    if is_better:
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Ensure model is provided if to play game (not to train)
    if args.play and not args.model:
        raise ValueError("No model provided to play game")

    model = dqn.DQN()
    model.cuda()

    # Load model or initialize weights
    if args.model:
        if os.path.isfile(args.model):
            checkpoint = torch.load(args.model)
            args.start_episode = checkpoint['episode']
            args.ini_e = checkpoint['epsilon']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Load model '{}'".format(args.model))
        else:
            raise ValueError("=> No model found at '{}'".format(args.model))
    else:
        model.initializeWeights()

    # Criterion and optimizer
    criterion = getCriterion().cuda()
    optimizer = getOptimizer(model, args.lr, args.weight_decay)

    flappy.init(display=(args.play or args.display))
    game = flappy.Flappy()

    if args.play:
        play(game, model)
    else:
        train(game, model, criterion, optimizer)
