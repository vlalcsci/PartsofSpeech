
#
####
# Put your report here!!
#
# (1) Description
#
# [Creating initial, transition, emission probability] First I calculated initial,
#  transition, and emission probabilities. In this procedure, I first calculated
#  initial probability. Now I have the total different types of the hidden states.
#  Using this information, I next initialized transition probabilities with all
#  possible combinations and set it to 0. Then I calculated transition probabilities.
#  Lastly, when creating emission probability, whenever a new word appears, I added
#  that word as the emission probability for all the hidden states and set it to 0.
#  The reason for the initialization is because, in the calculation of Viterbi algorithm,
#  I need all the information of the emission probabilities for all the words appeared
#  in training set.
#
# [Simplified]
#  For Simplified, for every word, I maximized over P(POS)*P(word | POS) 
#  In case the test data set had new unseen word, a very small value of probability was
#  taken as emission
#
# [Variable Elimination]
# For variable elimination, I used Dynamic Programming to calculate forward propagation and backward propagation 
# values. Forward Propagation = sum of (P(State-i | State-j)*P(State-j)* P(W-i | State-i)) across all states of j position,
# where j = i - 1. For i = 1, we have P(State)*(P(Word1 | State) for each state
# Backward propagation for any i = sum of (P(State-j | State-i)*P(Word-j | State-j)) for all states in j position,
# where j = i + 1
# For i = n (last word), back propagation = 1
# We get the probabilities for VE by multiplying forward scores with corresponding backward score
# VE = forward_propagation * backward_propagation
#
# [Viterbi] I used `score` and `trace` matrices. `score` matrix contains the scores
#  calculated during the Viterbi algorithm. `trace` is used to trace back the
#  hidden states. During the traceback, I appended the states in the list `hidden`,
#  then returned the reverse order of `hidden` that returns the list of predicted
#  hidden states from the beginning of the given sentence.
#
##############################################################
# (2) Description of how the program works
#
# The program `label.py` takes in a training file and a test file, then applies
# three different algorithms to mark every word in a sentence with its part of speech.
# In `solver.train`, function `train` in class `Solver` is called. As described above,
# the train function creates `initial`, `transition`, and `emission` probabilities.
# Then using these information, the program tests the test data on three algorithm
# we implemented in the class `Solver`.
#
#
#
##############################################################
# (3) Disscussion of problems, assumptions, simplification, and design decisions we made
#
# There were words in test file that are not trained in training file. In this case,
# there's no emission probabilities, hence the Viterbi algorithm raised error.
# I tried two different approach on this problem. First approach was simply set
# the score to 0 whenever unknown word appeared. Following is the result of the
# approach for the test set bc.test:
#
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       18.60%                0.00%
#         2. HMM VE:       18.60%                0.00%
#        3. HMM MAP:       62.21%               30.15%
#
# The result was poor. Hence, I tried another approach: reproduce the algorithm
# by only removing the emission probability in calculation (Using the previous
# score and transition probability only). Following is the result of the approach
# for the test set bc.test:
#
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       18.60%                0.00%
#         2. HMM VE:       18.60%                0.00%
#        3. HMM MAP:       89.78%               31.55%
#
# There was tremendous improvement on the accuracy.
#
# We took the log probability in the Viterbi algorithm. Also, by now the other algorithm
# had been implemented. This is the final accuracy. For the words not in the training set,
# we set the emission probability to a very small number
# 
# ==> So far scored 2000 sentences with 29442 words.
#                 Words correct:     Sentences correct:
# 0. Ground truth:      100.00%              100.00%
#   1. Simplified:       93.92%               47.45%
#       2. HMM VE:       94.94%               53.60%
#      3. HMM MAP:       94.88%               53.55%
#
##############################################################
# (4) Answers to any questions asked below in the assignment
#
#
#
####

from __future__ import division
import random
from math import log
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        self.SMALL_PROB = 1/10**6
        self.pos = {}
        self.initial = {}
        self.transition = {}
        self.emission = {}
#        self.words_in_training = []

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        ans = self.initial[label[0]]
        for i, (st, obs) in enumerate(zip(label, sentence)):
            if i == len(label)-1:
                ans *= self.emission[st].get(obs, self.SMALL_PROB)
                break
            ans *= self.transition[st].get(label[i+1], self.SMALL_PROB) * self.emission[st].get(obs, self.SMALL_PROB)
        return log(ans)

    # Do the training!
    #
    def train(self, data):
        ##############################################################
        # Initial Probability
        ##############################################################
        for line in data:
            ##### considering only first word
            self.initial[line[1][0]] = self.initial.get(line[1][0], 0) + 1
            ##### considering all words
            for S in line[1]:
                 self.pos[S] = self.pos.get(S, 0) + 1

        ##############################################################
        # Transition Probability
        ##############################################################
        states = list(self.initial.keys())

        for line in data:
            for S, S_prime in zip(line[1], line[1][1:]):
                try:
                    self.transition[S][S_prime] = self.transition[S].get(S_prime, 0) + 1
                except:
                    self.transition[S] = {}
                    self.transition[S][S_prime] = self.transition[S].get(S_prime, 0) + 1

        ##############################################################
        # Emission Probability
        ##############################################################
        for S in states:
            self.emission[S] = {}

        for line in data:
            for W, S in zip(line[0], line[1]):
                self.emission[S][W] = self.emission[S].get(W, 0) + 1
#                self.words_in_training.append(W) # for unseen words

        # Convert Counts to probabilities
        S_total = sum(self.pos.values())
        for S in self.pos:
            self.pos[S] /= S_total
        S_total = sum(self.initial.values())
        for S in self.initial:
            self.initial[S] /= S_total
        for S in self.transition:
            S_total = sum(self.transition[S].values())
            for S_prime in self.transition[S]:
                self.transition[S][S_prime] /= S_total
        for S in self.emission:
            S_total = sum(self.emission[S].values())
            for W in self.emission[S]:
                self.emission[S][W] /= S_total

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        ##### P(S | W) = P(W | S) * P(S) / P(W)
        states = list(self.pos.keys())
        predicted_states = []
        for word in sentence: # ignore unseen and return noun?
            most_prob_state = max([ (st, self.emission[st].get(word, self.SMALL_PROB) * self.pos[st]) \
                                        for st in states ], key = lambda x: x[1])
            predicted_states.append(most_prob_state[0])
#            max_prob, most_prob_state = 0, ''
#            for state in states:
#                P_state_given_word = self.emission[state].get(word, 1) * self.initial[state]
#                if P_state_given_word > max_prob:
#                    max_prob, most_prob_state = P_state_given_word, state
#            sentence_states.append(most_prob_state)
        return predicted_states

    def hmm_ve(self, sentence):
        states = list(self.pos.keys())
        observed = sentence
        # observed = [word for word in sentence if word in self.words_in_training] # ignore unseen words
#        score = np.zeros([len(states), len(observed)])
        forward = np.zeros([len(states), len(observed)])
        backward = np.zeros([len(states), len(observed)])
        predicted_states = []

        for i, obs in zip(range(len(observed)-1, -1, -1), observed[::-1]):
            for j, st in enumerate(states):
                if i == len(observed) - 1:
                    p = 1
                else:
                    p = sum( [ backward[k][i+1] * self.transition[st].get(key, self.SMALL_PROB) * self.emission[key].get(observed[i+1], self.SMALL_PROB) \
                                for k, key in enumerate(self.transition)] )
                backward[j][i] = p

        for i, obs in enumerate(observed):
            for j, st in enumerate(states):
                if i == 0:
                    p = self.initial[st]
                else:
                    p = sum( [forward[k][i-1] * self.transition[key].get(st, self.SMALL_PROB) \
                                for k, key in enumerate(self.transition)] )
                forward[j][i] = p * self.emission[st].get(obs, self.SMALL_PROB)

#                if forward[j][i] > max_value:
#                    max_value, max_state = score[j][i], st
#            predicted_states.append(max_state)
        self.ve = np.multiply(forward, backward)

        for i in range(len(observed)):
            z = np.argmax(self.ve[:, i])
            predicted_states.append(states[z])

        return predicted_states

    def hmm_viterbi(self, sentence):
        states = list(self.pos.keys())
        observed = sentence
        # observed = [word for word in sentence if word in self.words_in_training] # ignore unseen words
        self.viterbi = np.zeros([len(states), len(observed)])
        trace = np.zeros([len(states), len(observed)], dtype=int)

        for i, obs in enumerate(observed):
            for j, st in enumerate(states):
                if i == 0:
                    self.viterbi[j][i], trace[j][i] = log(self.initial[st]) + log(self.emission[st].get(obs, self.SMALL_PROB)), 0
                    #print score[j][i]
                else:
                    max_k, max_p = max([ (k, self.viterbi[k][i-1] + log(self.transition[key].get(st, self.SMALL_PROB))) \
                                           for k, key in enumerate(self.transition)], key = lambda x: x[1])
                    self.viterbi[j][i], trace[j][i] = max_p + log(self.emission[st].get(obs, self.SMALL_PROB)), max_k
        # trace back
        z = np.argmax(self.viterbi[:,-1])
        hidden = [states[z]]
        for i in range(len(observed)-1, 0, -1):
            z = trace[z,i]
            hidden.append(states[z])

        # return REVERSED traceback sequence
        return hidden[::-1]

    # This solve() method is called by label.py, so you should keep the interface
    #  the same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print ("Unknown algo!")



