import itertools as it
import pickle,os
from random import sample, randint, random
from time import time
import cv2
import numpy as np
import theano
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, MaxPool2DLayer, LSTMLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from theano import tensor
from tqdm import *
from time import sleep

# Replay memory:
class ReplayMemory:
    def __init__(self, capacity, channels, downsampled_x, downsampled_y):

        state_shape = (capacity, channels, downsampled_y, downsampled_x)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.nonterminal = np.zeros(capacity, dtype=np.bool_)

        self.size = 0
        self.capacity = capacity
        self.oldest_index = 0

    def add_transition(self, s1, action, s2, reward):
        self.s1[self.oldest_index] = s1
        if s2 is None:
            self.nonterminal[self.oldest_index] = False
        else:
            self.s2[self.oldest_index] = s2
            self.nonterminal[self.oldest_index] = True
        self.a[self.oldest_index] = action
        self.r[self.oldest_index] = reward

        self.oldest_index = (self.oldest_index + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.s2[i], self.a[i], self.r[i], self.nonterminal[i]

# Recurrent memory:
class RecurrentMemory:
    def __init__(self, capacity, channels, downsampled_x, downsampled_y):

        state_shape = (capacity, channels, downsampled_y, downsampled_x)
        self.s = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.nonterminal = np.zeros(capacity, dtype=np.bool_)

        self.size = 0
        self.capacity = capacity
        self.oldest_index = 0

    def add_transition(self, s1, action, reward):
        self.s[self.oldest_index] = s1
        if s1 is None:
            self.nonterminal[self.oldest_index] = False
        else:
            self.nonterminal[self.oldest_index] = True
        self.a[self.oldest_index] = action
        self.r[self.oldest_index] = reward

        self.oldest_index = (self.oldest_index + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = randint(0, self.size-sample_size)
        return self.s1[i:i+sample_size], self.a[i:i+sample_size], self.r[i:i+sample_size], self.nonterminal[i:i+sample_size]
        
# DRLBot
class DRLBot:
    # Q-learning settings:
    replay_memory_size = 10000
    discount_factor = 0.99
    start_epsilon = float(1.0)
    end_epsilon = float(0.1)
    epsilon = start_epsilon
    epsilon = 1.0
    static_epsilon_steps = 5000
    epsilon_decay_steps = 20000
    epsilon_decay_stride = (start_epsilon - end_epsilon) / epsilon_decay_steps

    # Max reward is about 100 (for killing) so it'll be normalized
    reward_scale = 0.01

    # Some of the network's and learning settings:
    learning_rate = 0.00001
    batch_size = 32
    epochs = 20
    training_steps_per_epoch = 5000
    test_episodes_per_epoch = 100

    # Other parameters
    skiprate = 7
    channels = 3
    downsampled_x = 192
    downsampled_y = downsampled_x
    episodes_to_watch = 10

    # Where to save and load network's weights.
    #params_savefile = "basic_params"
    params_savefile = "basic_params"
    params_loadfile = "ivomi_params" #"basic_params"

    ############
    #Initialization function
    #gets the required get_state_f and make_action_f
    #   get_state_f
    #   - Does not take input
    #   - Returns current image/state or None if game finished or -1 if waiting for next frame
    #   make_action_f
    #   - Takes action as integer
    #   - Returns resulting reward from action
    #   available_actions_num
    #   - an integer describing how many actions are available to the dqn
    #
    ############
    def __init__(self,name="Darryl",get_state_f=None,make_action_f=None,available_actions_num=0,recurrent=False):
        self.name = name
        self.get_state_f = get_state_f
        self.make_action_f = make_action_f
        self.recurrent = recurrent
        self.available_actions_num = available_actions_num
        self.memory = ReplayMemory(capacity=self.replay_memory_size, channels=self.channels, downsampled_x=self.downsampled_x, downsampled_y=self.downsampled_y)
        self.dqn, self.learn, self.get_q_values, self.get_best_action = self.create_network(self.available_actions_num)
    
    def load_model(self,params_loadfile="save_params"):
        if os.path.isfile(params_loadfile):
            params = pickle.load(open(params_loadfile, "r"))
            set_all_param_values(self.dqn, params)
        else:
            print params_loadfile + " not found"
    
    def save_model(self,params_savefile="save_params"):
        pickle.dump(get_all_param_values(self.dqn), open(params_savefile, "w"))
    
    def convert(self,img):
        img = img.astype(np.float32) / 255.0
        #img = cv2.resize(img, (self.channels, self.downsampled_x, self.downsampled_y))
        return img

    # Creates the network:
    def create_network(self,available_actions_num = None):
        if available_actions_num == None:
            available_actions_num = self.available_actions_num
        # Creates the input variables
        s1 = tensor.tensor4("States")
        a = tensor.vector("Actions", dtype="int32")
        q2 = tensor.vector("Next State best Q-Value")
        r = tensor.vector("Rewards")
        nonterminal = tensor.vector("Nonterminal", dtype="int8")

        # Creates the input layer of the network.
        
        dqn = InputLayer(shape=[self.batch_size, self.channels, self.downsampled_y, self.downsampled_x], input_var=s1)
        
        
        # Adds 3 convolutional layers, each followed by a max pooling layer.
        dqn = Conv2DLayer(dqn, num_filters=32, filter_size=[8, 8],
                          nonlinearity=rectify, W=GlorotUniform("relu"),
                          b=Constant(.1))
        dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
        dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[4, 4],
                          nonlinearity=rectify, W=GlorotUniform("relu"),
                          b=Constant(.1))
        dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
        dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[3, 3],
                          nonlinearity=rectify, W=GlorotUniform("relu"),
                          b=Constant(.1))
        dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
        # Adds a single fully connected layer.
        #dqn = DenseLayer(dqn, num_units=768, nonlinearity=rectify, W=GlorotUniform("relu"),
        #                 b=Constant(.1))
        dqn = DenseLayer(dqn, num_units=512, nonlinearity=rectify, W=GlorotUniform("relu"),
                         b=Constant(.1))
        dqn = DenseLayer(dqn, num_units=384, nonlinearity=rectify, W=GlorotUniform("relu"),
                         b=Constant(.1))
        dqn = DenseLayer(dqn, num_units=256, nonlinearity=rectify, W=GlorotUniform("relu"),
                         b=Constant(.1))
        # Adds a single fully connected layer which is the output layer.
        # (no nonlinearity as it is for approximating an arbitrary real function)
        dqn = DenseLayer(dqn, num_units=self.available_actions_num, nonlinearity=None)

        # Theano stuff
        q = get_output(dqn)
        # Only q for the chosen actions is updated more or less according to following formula:
        # target Q(s,a,t) = r + gamma * max Q(s2,_,t+1)
        target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + self.discount_factor * nonterminal * q2)
        loss = squared_error(q, target_q).mean()

        # Updates the parameters according to the computed gradient using rmsprop.
        params = get_all_params(dqn, trainable=True)
        updates = rmsprop(loss, params, self.learning_rate)

        # Compiles theano functions
        print "Compiling the network ..."
        function_learn = theano.function([s1, q2, a, r, nonterminal], loss, updates=updates, name="learn_fn")
        function_get_q_values = theano.function([s1], q, name="eval_fn")
        function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
        print "Network compiled."

        # Returns Theano objects for the net and functions.
        # We wouldn't need the net anymore but it is nice to save your model.
        return dqn, function_learn, function_get_q_values, function_get_best_action
 
    # Makes an action according to epsilon greedy policy and performs a single backpropagation on the network.
    def perform_play_step(self):
        # Checks the state and downsamples it.
        curr_state = self.get_state_f()
        if curr_state is None:
            return None
        s1 = self.convert(curr_state)

        # With probability epsilon makes a random action.
        if random() <= self.epsilon:
            a = randint(0, self.available_actions_num - 1)
        else:
            # Chooses the best action according to the network.
            a = self.get_best_action(s1.reshape([1, self.channels, self.downsampled_y, self.downsampled_x]))
        reward = self.make_action_f(a)
        reward *= self.reward_scale
        curr_state = self.get_state_f()
        if curr_state is None:
            s2 = None
        else:
            s2 = self.convert(curr_state)
        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, reward)
        return reward/self.reward_scale
        
    # Makes an action according to epsilon greedy policy and performs a single backpropagation on the network.
    def perform_play_step_no_storage(self):
        # Checks the state and downsamples it.
        curr_state = self.get_state_f()
        if curr_state is None:
            return -1
        s1 = self.convert(curr_state)


            # Chooses the best action according to the network.
        a = self.get_best_action(s1.reshape([1, self.channels, self.downsampled_y, self.downsampled_x]))
        reward = self.make_action_f(a)
        reward *= self.reward_scale
        curr_state = self.get_state_f()
        return reward/self.reward_scale
    
    # Makes an action according to epsilon greedy policy and performs a single backpropagation on the network.
    def perform_learning_step(self):
        if self.memory.size > self.batch_size:
            s1, s2, a, reward, nonterminal = self.memory.get_sample(self.batch_size)
            q2 = np.max(self.get_q_values(s2), axis=1)
            loss = self.learn(s1, q2, a, reward, nonterminal)
        else:
            loss = 0
        return loss
    
    #gathers state-transition/reward pairs to be learned later
    #RETURN - average reward
    def gain_experience(self,num_steps):
        if num_steps < 1:
            return 0
        total_reward = 0
        for i in xrange(min(num_steps,self.memory.capacity)):
            result = self.perform_play_step() 
            if result == None:
                epsilon = max(0.1,self.epsilon-0.01)
                return total_reward
            else:
                total_reward += result
        return total_reward
    
    def learn_from_experience(self,num_steps):
        for i in tqdm(xrange(num_steps)):
            self.perform_learning_step()