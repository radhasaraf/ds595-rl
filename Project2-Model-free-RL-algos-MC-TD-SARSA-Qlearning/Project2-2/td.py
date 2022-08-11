#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np

"""
    Temporal Difference
    In this problem, you will implement an AI player for cliff-walking.
    The main goal of this problem is to get familiar with temporal difference algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v td_test.py' in the terminal.
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
"""


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """

    greedy_index = np.argmax(Q[state])
    probability = np.ones(nA)*epsilon/nA  # exploration
    probability[greedy_index] += 1 - epsilon  # exploitation

    return np.random.choice(np.arange(len(Q[state])), p=probability)


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """

    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.array(nA)
    action_count = env.action_space.n
    Q = defaultdict(lambda: np.zeros(action_count))

    for i in range(n_episodes):
        epsilon = 0.99*epsilon  # define decaying epsilon
        curr_state = env.reset()  # initialize the environment
        curr_action = epsilon_greedy(Q, curr_state, action_count, epsilon)  # get an action from policy

        done = False
        while not done:  # loop for each step of episode until terminal state
            next_state, reward, done, info = env.step(curr_action)  # return new state, reward and done
            next_action = epsilon_greedy(Q, next_state, action_count, epsilon)  # get next action

            # TD update
            td_target = reward + gamma*Q[next_state][next_action]  # td_target
            td_error = td_target - Q[curr_state][curr_action]  # td_error

            Q[curr_state][curr_action] = Q[curr_state][curr_action] + alpha*td_error  # new Q

            curr_state = next_state  # update state
            curr_action = next_action  # update action

    return Q


def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.array(nA)
    action_count = env.action_space.n
    Q = defaultdict(lambda: np.zeros(action_count))

    # loop n_episodes
    for i in range(n_episodes):
        epsilon = 0.99*epsilon  # define decaying epsilon
        curr_state = env.reset()  # initialize the environment

        done = False
        while not done:  # loop for each step of episode until terminal state
            curr_action = epsilon_greedy(Q, curr_state, action_count, epsilon)  # get an action from policy
            next_state, reward, done, info = env.step(curr_action)  # return a new state, reward and done

            # TD update
            best_action = np.argmax(Q[next_state])
            td_target = reward + gamma*Q[next_state][best_action]  # td_target with best Q
            td_error = td_target - Q[curr_state][curr_action]  # td_error

            Q[curr_state][curr_action] += alpha*td_error  # new Q

            curr_state = next_state  # update state

    return Q
