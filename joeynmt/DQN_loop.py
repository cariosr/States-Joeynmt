"""
DQN modules
"""
import math
from logging import Logger
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from joeynmt.helpers import bpe_postprocess, load_config, make_logger, get_latest_checkpoint, \
    load_checkpoint, set_seed, log_cfg, symlink_update
from joeynmt.data import load_data, make_data_iter
from joeynmt.model import build_model
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.batch import Batch
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
import random
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

from scipy.stats import entropy

import queue

def freeze_model(model):
    """
    Function to freeze a given model.
    :param model: model to be frozen
    :return:
    """
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def unfreeze_model(model):
    """
    Function to unfreeze a given model.
    :param model: model to be unfrozen
    :return:
    """
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def funcMapSta_Ac(N_layers, state_size, actions_size):
    """
    Function to create a list with the sizes of the layers.
    :param N_layers: Number of layers to be created in the DQN. (Including the output layer).
    :param state_size: Size of the state for the DQN (it's the initial input of the DQN).
    :param actions_size: Number of actions for the DQN (it's the very last output of the DQN).
    :return:
        -ns: list with the sizes for the layers to be used in the DQN (given the parameter N_layers).
             These sizes are related linearly, being the state_size the
             maximum(first element) and action_size the minumum (the last element).
        """
    
    n_layers = np.arange(1,N_layers+1)
    
    ns = ((state_size-actions_size)/(1-N_layers)) * (n_layers -1) + state_size
    
    return ns.astype(int)

class Net(nn.Module):
    """Creates a DQN (MLP) with random initialization."""
    def __init__(self, N_STATES, N_ACTIONS, N_LAYERS):
        """
        :param N_STATES: Size of the state (input layer).
        :param N_ACTIONS: Number of actions (output layer).
        :param N_layers: Number of layers to be created in the DQN. (Including the output layer).
        """
        super(Net, self).__init__()
        
        net_size = funcMapSta_Ac(N_LAYERS, N_STATES, N_ACTIONS)
        

        self.hidden = []
        for k in range(len(net_size)-2):
            self.hidden.append(nn.Linear(net_size[k], net_size[k+1]))
            self.hidden[k].weight.data.normal_(0, 0.1)   # initialization


        # Output layer
        self.out = nn.Linear(net_size[-2], net_size[-1])
        self.out.weight.data.normal_(0, 0.1)   # initialization


    def forward(self, x):
        """
        Applies a MLP to estimate the Q-value given a state
        for all the actions. It uses relu activation as DQN from
        the Mnih's paper.
        :param x: state that represents the current sequence.
            (batch_size, state_size)
        :return: actions_value: Q-values for each action.
            (batch_size, actions_size)
        """

        for layer in self.hidden:
            x = F.relu(layer(x))
        actions_value = self.out(x)

        return actions_value

# init a seed for the randomness
random.seed(10)

class QManager(object):
    """ Manages Q-learning loop, aggregate the .yaml parameters.
        Initializes the model. Loads the training, dev and test data.
    """

    def __init__(self, cfg_file,
         ckpt: str,
         output_path: str = None,
         logger: Logger = None) -> None:

        """
        Recovers a saved JoeyNMT model, specified in configuration file.
        :param cfg_file: path for the configuration file.
        :param ckpt: path for checkpoint to load.
        :param output_path: path to save output model.
        :param logger: logger to trace (creates new logger if it's not set).
        """

        cfg = load_config(cfg_file)

        if "test" not in cfg["data"].keys():
            raise ValueError("Test data must be specified in config.")


        if "dqn" not in cfg.keys():
            raise ValueError("dqn data must be specified in config.")

        self.model_dir = cfg["training"]["model_dir"]

        # when checkpoint is not specified, take latest (best) from model dir
        if ckpt is None:
            model_dir = cfg["training"]["model_dir"]
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                raise FileNotFoundError("No checkpoint found in directory {}."
                                        .format(model_dir))
            try:
                step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
            except IndexError:
                step = "best"

        # load JoeyNMT hyperparameters
        set_seed(seed=cfg["training"].get("random_seed", 42))
        self.batch_size = cfg["dqn"]["batch_size"]
        self.batch_type = cfg["training"].get(
            "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
        self.use_cuda = cfg["training"].get("use_cuda", False)
        self.level = cfg["data"]["level"]
        self.eval_metric = cfg["training"]["eval_metric"]
        self.max_output_length = cfg["training"].get("max_output_length", None)

        # lists to track bleu and entropy
        self.bleu_list = []
        self.entro_list = []

        # variables for the stopping criteria
        self.stop = False
        self.non_stop = cfg["dqn"]["non_stop"]
        self.stop_reason = ""

        # load the data
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])
        #Load the DQN parameters:
        self.sample_size = cfg["dqn"]["sample_size"]
        self.lr = cfg["dqn"].get("lr", 0.01)
        self.egreed_max = cfg["dqn"].get("egreed_max", 0.9)
        self.egreed_min = cfg["dqn"].get("egreed_min", 0.01)
        self.gamma = cfg["dqn"].get("gamma", 0.9)
        self.nu_iter = cfg["dqn"]["nu_iter"]
        
        self.mem_cap = cfg["dqn"]["mem_cap"]
        self.beam_min = cfg["dqn"]["beam_min"]
        self.beam_max = cfg["dqn"]["beam_max"]
        self.state_type = cfg["dqn"]["state_type"]
        self.nu_pretrain = cfg["dqn"]["nu_pretrain"]
        self.reward_type = cfg["dqn"]["reward_type"]
        self.count_post_pre_train = 0
        self.hidden_size = cfg["model"]["encoder"]["hidden_size"]
        # get the state_size depending on the state__type
        if self.state_type == 'hidden' and cfg["model"]["encoder"]["bidirectional"]:
            self.state_size = self.hidden_size*2
        else:
            self.state_size = self.hidden_size
        self.N_layers = cfg["dqn"]["N_layers"]
        # compute the actions_size based on the length of the vocabulary
        self.actions_size = len(src_vocab)

        # the bigger the batch the more epochs we can perform
        # this ratio computes that and modify the batch size accordingly
        epoch_batch_size_ratio = self.batch_size/32
        self.epochs = int(cfg["dqn"]["epochs"]*epoch_batch_size_ratio)

        # Init the Q-eval and Q-target
        self.eval_net = Net(self.state_size, self.actions_size, self.N_layers)
        self.target_net = Net(self.state_size, self.actions_size, self.N_layers)

        # Init the Q-target as the algorithm 5 in the Yaser's paper suggests.
        # Yaser's paper: https://arxiv.org/pdf/1805.09461v1.pdf.
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = 0
        # size of second dim of the memory based on the transition tuple
        # (explained in store_transition function)
        # transition = (state, a, state_, a_, finished)
        self.size_memory1 = self.state_size * 2 + 2 + 2
        # memory to store the experiences of transitions
        self.memory = np.zeros((self.mem_cap, self.size_memory1 ))
        # use Adam optimizer as the  Mnih's paper suggests.
        self.optimizer = torch.optim.Adam(self.eval_net.parameters()
                                          , lr=self.lr )
        # loss to optimize according to the DQN-learning algorithm (Mnih's paper).
        self.loss_func = nn.MSELoss()
        # define special tokens for joeyNMT
        self.bos_index = trg_vocab.stoi[BOS_TOKEN]
        self.eos_index = trg_vocab.stoi[EOS_TOKEN]
        self.pad_index = trg_vocab.stoi[PAD_TOKEN]

        # specify the datasets
        self.data_to_train_dqn = {"train": train_data}
        self.data_to_dev = {"train": train_data}
        # self.data_to_dev = {"dev": dev_data}

        # load model state from the checkpoint
        model_checkpoint = load_checkpoint(ckpt, use_cuda=self.use_cuda)
        # build model and load parameters into it
        self.model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
        self.model.load_state_dict(model_checkpoint["model_state"])
        

        self.best_ckpt_score = -np.inf
        # comparison function to get the model that performs best
        self.is_best = lambda score: score > self.best_ckpt_score
        self.ckpt_queue = queue.Queue(maxsize=2)

        # change the model to cuda if using cuda
        if self.use_cuda:
            self.model.cuda()
            self.eval_net.cuda()
            self.target_net.cuda()
            self.loss_func.cuda()

        # set the reward_type
        if self.reward_type == "bleu_batch":
            print("You select the reward based on the bleu_batch fin")
        elif self.reward_type == "hc_batch":
            print("You select the reward based on the Hand Crafted Reward ")

        # get the current date to write the folder for tensorboard
        time_stamp = time.time()
        date = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
        # get the hyperparameters more relevant
        # add as many hyperparameters as desired in the following manner:
        # -write the name of the hyperparameter at front then the value
        # -write underscore after each hyperparameter, except for the latest one
        relevant_hyp = "gamma=" + str(self.gamma) + "_" + "batch_size=" + str(self.batch_size)  \
                    + "_" + "lr=" + str(self.lr) \
                    + "_" + "reward_type=" + str(self.reward_type) + cfg["dqn"]["other_descrip"]

        # construct the name of the folder for tensorboard for the test given the date and the relevant_hyp
        path_tensroboard = self.model_dir + "/tensorboard_DQN/" + date + "/" + relevant_hyp + "/"
        self.tb_writer = SummaryWriter( log_dir=path_tensroboard , purge_step=0)
        if not os.path.exists(self.model_dir + "/logs/" + date + "/"):
            os.makedirs(self.model_dir + "/logs/" + date + "/")
        path_logger = self.model_dir + "/logs/" + date + "/" + relevant_hyp + ".log"
        self.logger = make_logger(path_logger)

        # counters and accumulators
        self.dev_network_count = 0
        self.r_optimal_total = 0
        self.index_mem = 0

        # set up the logger
        log_cfg(cfg, self.logger)
        self.logger.info("The number of epoch proportional to the batch size is %d", self.epochs)
        self.logger.info("We are using the reward named: %s", self.reward_type)
        self.logger.info("As states we set the : %s", self.state_type)


    def Collecting_experiences(self)-> None:
        """
        Main function.
        This function performs the main loop of the training.
        It calls the functions to collect experiences and the
        learning part when sufficient experiences are accumulated.
        It iterates in a nested ways as follows:
         1. First per epoch
         2. Then per data set
         3. Then per batch
         4. Finally for each time t along the sequences that are being generated.
        """

        for epoch_no in range(self.epochs):
            print("EPOCH %d", epoch_no + 1)
            self.logger.info("EPOCH %d", epoch_no + 1)

            # stop if any of the stopping criteria is satisfied
            if self.stop and not self.non_stop:
                print(self.stop_reason)
                break

            self.tb_writer.add_scalar("parameters/gamma",
                                              self.gamma, epoch_no)
            self.logger.info('gamma:  %.2f', self.gamma)
            print('gamma: ', self.gamma)

            for _, data_set in self.data_to_train_dqn.items():
                valid_iter = make_data_iter(
                    dataset=data_set, batch_size=self.batch_size, batch_type=self.batch_type,
                    shuffle=True, train=True)

                batch_i = 0

                for valid_batch in iter(valid_iter):
                    freeze_model(self.model)
                    batch = Batch(valid_batch
                    , self.pad_index, use_cuda=self.use_cuda)
                    encoder_output, encoder_hidden = self.model.encode(
                        batch.src, batch.src_lengths,
                        batch.src_mask)

                    # if maximum output length is not globally specified, adapt to src len
                    if self.max_output_length is None:
                        self.max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

                    batch_size = batch.src_mask.size(0)
                    prev_y = batch.src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                            dtype=torch.long) #JoeyNMT parameter

                    output = [] # output for output sequences
                    hidden = self.model.decoder._init_hidden(encoder_hidden) #JoeyNMT parameter
                    prev_att_vector = None #JoeyNMT parameter
                    # flag to check if the current iterator has reached the final of a sequence
                    finished = batch.src_mask.new_zeros((batch_size, 1)).byte()

                    # list for experiences (transitions)
                    exp_list = []

                    prev_att_vector = encoder_output.new_zeros([batch_size, 1, self.hidden_size]) #JoeyNMT parameter
                    output.append(prev_y.squeeze(1).detach().cpu().numpy())
                    for t in range(self.max_output_length):
                        # get the state_type accordingly to the config file
                        if self.state_type == 'hidden':
                            state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]#(batch_size, state_size)
                        else:
                            state = prev_att_vector.squeeze(1).detach().cpu().numpy()  # (batch_size, state_size)

                        # decode one single step
                        logits, hidden, att_probs, prev_att_vector = self.model.decoder(
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask,
                            trg_embed=self.model.trg_embed(prev_y),
                            hidden=hidden,
                            prev_att_vector=prev_att_vector,
                            unroll_steps=1)

                        if self.state_type == 'hidden':
                            state_ = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0] # (batch_size, state_size)
                        else:
                            state_ = prev_att_vector.squeeze(1).detach().cpu().numpy()          # (batch_size,state_size)

                        next_word = torch.argmax(logits, dim=-1) # batch x time=1
                        a = prev_y.squeeze(1).detach().cpu().numpy() # (batch_size ,1)
                        a_ = next_word.squeeze(1).detach().cpu().numpy() # (batch_size ,1)
                        output.append(next_word.squeeze(1).detach().cpu().numpy())
                        prev_y = next_word
                        # check if it is the last element of the sequence (EOS_TOKEN)
                        is_eos = torch.eq(next_word, self.eos_index)        # (batch_size ,1)
                        finished += is_eos
                        if t > 0:
                            finished_aux = finished.clone()
                            tup = (state, a, state_, a_, finished_aux)
                            exp_list.append(tup)

                        if (finished >= 1).sum() == batch_size:
                            finished_aux = finished.clone() + 1
                            tup = (state_, self.eos_index*torch.ones(batch_size, 1 ) , torch.zeros(size=(batch_size, self.state_size )),
                                   self.pad_index * torch.ones(size=(batch_size, 1)), finished_aux)
                            exp_list.append(tup)
                            break

                        if t == self.max_output_length-1:
                            print("reach the max output")
                            break

                    #Collect rewards
                    hyp = np.stack(output, axis=1)  # batch, time

                    if epoch_no == 0 and batch_i == 0:
                        # Tracing to check if the rewards were well assigned for the first time
                        r = self.Reward_batch(batch.trg_input, hyp, show=True)  # (batch, length seq)
                        print ("Function Collecting_experiences")

                    else:
                        r = self.Reward_batch(batch.trg_input, hyp, show=False)  # (batch, length seq)

                    # Store the transitions with included rewards
                    self.store_transition(exp_list, r)

                    batch_i += 1
                    # Call the learning if there are enough transitions
                    if self.index_mem > self.mem_cap:
                        self.learn()

        self.tb_writer.close()       

    def learn(self):
        """
        Selects experiences based on the reward value.
        Computes the bellman equation and the ACC model.
        The learning part is primarily based on the Algorithm 5 from the Yaser's  paper (without the actor updating).
            -link: https://arxiv.org/pdf/1805.09461v1.pdf
        """
        # Update of the Q-target network
        if self.learn_step_counter % self.nu_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

            if self.learn_step_counter == 0:
                self.logger.info(
                    'As a referece this first test on dev data is made with the Q networks, initialized randomly: ')
                print('As a referece this first test on dev data is made with the Q networks, initialized randomly: ')

            else:
                self.logger.info(" Copying the Q-Val to Q-tar in step: %d ", self.dev_network_count)
                print("\nLets copy the parameters of Q-value net to Q-target net!. Then test the performance on the dev data: ")

            # Get the bleu score to evaluate the performance so far
            current_bleu = self.dev_network()
            print("Current bleu score is: ", current_bleu)

            # update the bleu_list
            self.bleu_list += [current_bleu]
            print("Current bleu_list: {}\n".format(self.bleu_list))

            # Save the best model if that's the case
            if self.is_best(current_bleu):
                self.best_ckpt_score = current_bleu
                self.logger.info(
                    'Hooray! New best validation result [%f]!', current_bleu)
                print('Hooray! New best validation result :', current_bleu)
                self._save_checkpoint()

        self.learn_step_counter += 1
        long_Batch = self.sample_size*10

        # Select the highest rewards values
        b_memory_big = self.memory[np.argsort(-self.memory[:, self.state_size+1])][:long_Batch]

        # Sample from the memory with the highest rewards values
        sample_index = np.random.choice(long_Batch, self.sample_size)

        # Get the batch transitions from the sample
        b_memory = b_memory_big[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_size+2: self.state_size+2 + self.state_size])
        b_is_eos = torch.FloatTensor(b_memory[:, -1]).view(self.sample_size, 1)

        # Unfreeze the Q-eval to perform the learning
        unfreeze_model(self.eval_net)

        if self.use_cuda:
            b_s = b_s.cuda()
            b_s_ = b_s_.cuda()
            b_a = b_a.cuda()

        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate the on the Q-target

        if self.learn_step_counter % 50 == 1:
            print ("learn step counter: ", self.learn_step_counter)
            print ("dev_network_count: ", self.dev_network_count )

        # Take the most likely action.
        if self.learn_step_counter < self.nu_pretrain: # Use the hyperparameter nu_pretrain to take the true action
            if self.learn_step_counter == 1:
                print ("Using pretraining...")
            b_is_eos = ~torch.eq(b_a, self.eos_index)
            b_a_ = torch.LongTensor(b_memory[:, self.state_size+2 + self.state_size]).view(self.sample_size, 1)

        else: # or take the max from the output of the Q-target
            if self.learn_step_counter == self.nu_pretrain:
                print ("Starting using Q target net....")
            b_a_ = torch.LongTensor(q_next.max(1)[1].view(self.sample_size, 1).long())

        if self.use_cuda:
            b_a_ = b_a_.cuda()

        q_eval_next = self.eval_net(b_s_).gather(1, b_a_) # (batch, 1)

        # Bellman equation
        # If eos q_target = reward
        q_target = b_r + self.gamma * b_is_eos* q_eval_next.view(self.sample_size, 1)   # shape (batch, 1)

        # code to manage the entropy of the Q-eval net
        soft_func = torch.nn.Softmax(dim = -1)
        q_eval_all = self.eval_net(b_s).detach()
        q_eval_max = torch.LongTensor(q_eval_all.max(1)[1].view(self.sample_size, 1).long())
        entro = entropy(soft_func(q_eval_all).T, base=self.actions_size)
        aver_entro = entro.sum()/self.sample_size
        self.tb_writer.add_scalar("learn/q_eval_entropy",
                        aver_entro, self.learn_step_counter)

        if self.use_cuda:
            q_eval = q_eval.cuda()
            q_target = q_target.cuda()

        # compute the loss based on the Q-eval and the Q-target
        loss = self.loss_func(q_eval, q_target)
        self.tb_writer.add_scalar("learn/learn_batch_loss",
                                              loss.data, self.learn_step_counter)

        # Stopping criterion for loss
        if loss < (1.5 * (10 ** (-4))):
            self.stop_reason = "Stopped because loss shrinking too slowly"
            self.stop = True

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Freeze the eval net after performing the learning
        freeze_model(self.eval_net)

    def store_transition(self, exp_list, rew, show = False):

        """
        Fills/ or refills the memory with experiences.
        :param exp_list: List of experiences (transitions).
                            -tuples  = (state, a, state_, a_, finished)
        :param rew: list of rewards for every experience.
        """

        if len(exp_list) != len(rew[0]):
            print('length of exp_list: ', len(exp_list))
            print('length of rew: ', len(rew[0]))

        # these two lists should have the same length
        assert (len(exp_list) == len(rew[0]))

        # Iterate over the experience list
        for i, ele in enumerate(exp_list):
            # Get the values from the tuple
            state, a, state_, a_, finished  = ele
            r = rew[:,i]   # (batch_size, 1)
            batch_size_aux = len(rew)

            # Iterate over the batch
            for idx_batch in np.arange(batch_size_aux):
                state_idx = state[idx_batch]
                a_idx = a[idx_batch]
                r_idx = r[idx_batch]#.numpy()
                state_idx_ = state_[idx_batch]
                a_idx_ = a_[idx_batch]

                # Check if it's a final transition
                if finished[idx_batch] < 2:
                    transition = np.hstack((state_idx, [a_idx, r_idx], state_idx_, a_idx_, 1))
                else:
                    # correct line according to the paper
                    # transition = np.hstack((state_idx, [a_idx, r_idx], state_idx_, a_idx_, 0))

                    # line that showed better performance
                    transition = np.hstack((state_idx, [a_idx, r_idx], state_idx_, a_idx_, 1))

                # store the last transitions into the memory
                index = self.index_mem  % self.mem_cap
                self.memory[index, :] = transition
                self.index_mem += 1

    def Reward_batch(self, trg, hyp, show = False):
        """
        Computes the rewards per batch based on the bleu of
        the final translation for the full sequence or based
        the hand crafted reward explained in the report.
        :param trg: target sequence.
        :param hyp: hypothesis sequence.
        :return:
            -rew_distributed: array with the rewards per batch given trg and hyp.
        """
        batch_size_aux = len(hyp)
        extra_col = self.eos_index * torch.ones(batch_size_aux, dtype=int).view(batch_size_aux, -1)
        if self.use_cuda:
            extra_col = extra_col.cuda()
        trg = torch.cat([trg, extra_col], dim=1).detach()
        rew_distributed = np.zeros([batch_size_aux, len(hyp[0]) -1 ])

        # Iterates over the batch to assign rewards
        for i in np.arange(batch_size_aux):
            
            hyp_i = hyp[i]
            trg_i = trg[i].detach().cpu().numpy()

            decoded_valid_hyp = self.model.trg_vocab.arrays_to_sentences(arrays=hyp_i.reshape([1,-1]),
                                                        cut_at_eos=True)
            decoded_valid_tar = self.model.trg_vocab.arrays_to_sentences(arrays=trg_i.reshape([1,-1]),
                                                        cut_at_eos=True)

            # evaluate with metric on each target and hypothesis
            join_char = " " if self.level in ["word", "bpe"] else ""
            valid_references = [join_char.join(t) for t in decoded_valid_tar]
            valid_hypotheses = [join_char.join(t) for t in decoded_valid_hyp]

            # post-process
            if self.level == "bpe":
                valid_references = [bpe_postprocess(v)
                                    for v in valid_references]
                valid_hypotheses = [bpe_postprocess(v) for
                                    v in valid_hypotheses]

            # if references are given, evaluate against them
            assert len(valid_hypotheses) == len(valid_references)

            if self.reward_type == "bleu_batch" or self.reward_type == "bleu_fin":
                blue_batch_score = bleu(valid_hypotheses, valid_references)
                rew_distributed[i, -1] =  blue_batch_score
            else: # case of hand crafted reward (self.reward_type == "hc_batch")
                rew_temp = self.Reward_hc(trg_i, hyp_i)
                rew_distributed[i] =  rew_temp

        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target reward: ", decoded_valid_tar)
            print("Eval  reward: ", decoded_valid_hyp)
            print("Target reward: ", trg)
            print("Eval  reward: ", hyp)
            print("Rew vector reward: ", rew_distributed)

        return rew_distributed


    def Reward_hc(self, trg, hyp, show = False):

        """
        Computes a handcrafted reward given a target sequence and an hypothesis sequence.
        The explanation of this handcrafted reward is in the report.
        :param trg: target sequence.
        :param hyp: hypothesis sequence.
        :param show: flag to trace or not.
        :return:
            -final_rew: array with the rewards of the size of the hypothesis sequence.
        """

        trg_b = trg 
        if len(trg_b) != len(hyp):
            trg_c = np.ones([len(hyp)], dtype = int)*(self.actions_size+1)
            if len(trg_b) > len(hyp):
                lon = len(hyp)
            else:
                lon = len(trg_b)
            trg_c[:lon] = trg_b[:lon]
            trg_b = trg_c

        bolles = trg_b[:] == hyp[:]
        bolles = bolles*np.ones(len(trg_b)) + np.zeros(len(trg_b))

        # Punishment increases as the length increases.
        final_rew = bolles*np.arange(1,len(trg_b)+1)
        final_rew = np.diff(final_rew) -np.arange(len(hyp)-1)*0.2

        # Punish the wrong decisions, but the last one (to avoid the large hyp).
        for i in np.arange(1,len(final_rew)-1):
            final_rew[i] = (final_rew[i]+final_rew[i-1])/2.0

        # Penalize the second token when it goes wrong
        if len(hyp) > 2:
            if trg_b[1] != hyp[1] and trg_b[2] == hyp[2]:
                final_rew[1] = final_rew[1]*0.5

        # tracing
        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target: ", trg_b)
            print("Eval  : ", hyp)
            print("Reward: ", final_rew, "\n")
            print(trg, trg_a)

        return final_rew


    def dev_network(self):
        """
        Shows how is the current performance over the dev (or training) data set,
        by means of the total reward and the bleu score.
        :return:
            -current_valid_score: bleu score for the entire dev (or training) dataset.
        """

        freeze_model(self.eval_net)
        for data_set_name, data_set in self.data_to_dev.items():

            valid_iter = make_data_iter(
                dataset=data_set, batch_size=self.batch_size, batch_type=self.batch_type,
                shuffle=False, train=False)
            valid_sources_raw = data_set.src

            # Don't track gradients during validation
            r_total = 0
            roptimal_total = 0
            all_outputs = []
            all_outputs_to_bleu = []
            batch_i = 0
            aver_entro_list = []

            for valid_batch in iter(valid_iter):

                batch = Batch(valid_batch, self.pad_index, use_cuda=self.use_cuda)
                # sort batch now by src length to evaluate/train easier
                sort_reverse_index = batch.sort_by_src_lengths()
                encoder_output, encoder_hidden = self.model.encode(
                    batch.src, batch.src_lengths,
                    batch.src_mask)

                # if maximum output length is not globally specified, adapt it to src length
                if self.max_output_length is None:
                    self.max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

                batch_size = batch.src_mask.size(0)
                prev_y = batch.src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                        dtype=torch.long)
                output = []
                output_to_blue = []
                hidden = self.model.decoder._init_hidden(encoder_hidden)
                prev_att_vector = None
                finished = batch.src_mask.new_zeros((batch_size, 1)).byte()
                output.append(prev_y.squeeze(1).detach().cpu().numpy())

                for t in range(self.max_output_length):

                    # decode one single step
                    logits, hidden, att_probs, prev_att_vector = self.model.decoder(
                        encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=batch.src_mask,
                        trg_embed=self.model.trg_embed(prev_y),
                        hidden=hidden,
                        prev_att_vector=prev_att_vector,
                        unroll_steps=1)

                    if self.state_type == 'hidden':
                        state = torch.cat(hidden, dim=2).squeeze(1).detach()[0]
                        if self.use_cuda:
                            state = state.cuda()
                    else:
                        state = torch.FloatTensor(prev_att_vector.squeeze(1).detach())
                        if self.use_cuda:
                            state = state.cuda()

                    # Compute the Q values with the help of the Q-eval net.
                    logits = self.eval_net(state)
                    batch_size_aux =  len(logits)

                    # Code to compute entropy
                    soft_func = torch.nn.Softmax(dim = -1)
                    entro = entropy(soft_func(logits.detach()).T, base=self.actions_size)
                    aver_entro = entro.sum()/batch_size_aux
                    aver_entro_list.append(aver_entro)

                    logits = logits.reshape([batch_size_aux, 1, -1])
                    next_word = torch.argmax(logits, dim=-1)                        
                    output.append(next_word.squeeze(1).detach().cpu().numpy())
                    output_to_blue.append(next_word.squeeze(1).detach().cpu().numpy())
                    prev_y = next_word

                    # check if it is the last element of the sequence (EOS_TOKEN)
                    is_eos = torch.eq(next_word, self.eos_index)
                    finished += is_eos

                    # stop predicting if <eos> reached for all elements in batch
                    if (finished >= 1).sum() == batch_size:
                        break

                stacked_output = np.stack(output, axis=1)  # batch, time
                stacked_output_to_bleu = np.stack(output_to_blue, axis=1)  # batch, time
                hyp = stacked_output
                if batch_i == 0:
                    self.logger.info("Currently on the prediction (dev network function): ")
                    r = self.Reward_batch(batch.trg_input, hyp , show = True)
                else:
                    r = self.Reward_batch(batch.trg_input, hyp , show = False)

                r_total += np.sum(r)

                # compute the optimal reward just the first time, then just use it
                if self.dev_network_count == 0:
                    extra_col = self.eos_index *torch.ones(len(batch.trg_input) , dtype=int).view(len(batch.trg_input),-1)
                    if self.use_cuda:
                        extra_col = extra_col.cuda()
                    trg_extra_col = torch.cat([batch.trg_input,extra_col ], dim= 1).detach()
                    roptimal = self.Reward_batch(batch.trg_input, trg_extra_col , show = False)
                    roptimal_total += np.sum(roptimal)

                # collect the output sequences
                all_outputs.extend(stacked_output)
                all_outputs_to_bleu.extend(stacked_output_to_bleu)
                batch_i += 1

            # get average of the entropy
            aver_entro = sum(aver_entro_list)/len(aver_entro_list)

            # Trace the optimal reward just in the first iteration
            if self.dev_network_count == 0:
                self.logger.info("Optimal reward is: %.2f", roptimal_total)
                print('Optimal reward is: ', roptimal_total)
                self.r_optimal_total = roptimal_total

            assert len(all_outputs) == len(data_set)

            # decode back to symbols
            decoded_valid = self.model.trg_vocab.arrays_to_sentences(arrays=all_outputs_to_bleu,
                                                                cut_at_eos=True)

           # evaluate with metric in the full data set
            join_char = " " if self.level in ["word", "bpe"] else ""
            valid_sources = [join_char.join(s) for s in data_set.src]
            valid_references = [join_char.join(t) for t in data_set.trg]
            valid_hypotheses = [join_char.join(t) for t in decoded_valid]


            self.logger.info('On %s', data_set_name)
            self.logger.info('valid_references \t vs \t predicted_hypotheses')

            # Trace the first ten examples of references and hypotheses
            for i in np.arange(10):
                self.logger.info( ' %s\t vs %s\t', valid_references[i],  valid_hypotheses[i])
                print(valid_references[i], '\t vs \t',  valid_hypotheses[i])

            # post-process
            if self.level == "bpe":
                valid_sources = [bpe_postprocess(s) for s in valid_sources]
                valid_references = [bpe_postprocess(v)
                                    for v in valid_references]
                valid_hypotheses = [bpe_postprocess(v) for
                                    v in valid_hypotheses]

           # if references are given, evaluate against them
            if valid_references:
                assert len(valid_hypotheses) == len(valid_references)
                current_valid_score = bleu(valid_hypotheses, valid_references)
            else:
                current_valid_score = -1

            self.dev_network_count += 1
            self.tb_writer.add_scalar("dev/"+data_set_name+"_reward",
                                            r_total, self.dev_network_count)
            self.tb_writer.add_scalar("dev/"+data_set_name+"_bleu",
                                            current_valid_score, self.dev_network_count)
            self.tb_writer.add_scalar("dev/regret",
                                            self.r_optimal_total - r_total, self.dev_network_count)
            self.tb_writer.add_scalar("dev/Entropy",
                                            aver_entro, self.dev_network_count)

            # code for stopping criteria
            self.entro_list += [aver_entro]
            if len(self.entro_list) > 20:
                prev = sum(self.entro_list[-20:-11])
                curr = sum(self.entro_list[-10:-1])
                if curr > prev:
                    self.stop_reason = "Stopped because entropy no longer decreasing"
                    self.stop = True

            if len(self.bleu_list) > 10:
                prev = sum(self.bleu_list[-10:-6])
                curr = sum(self.bleu_list[-5:-1])
                if curr < prev:
                    self.stop_reason = "Stopped because dev_bleu decreasing"
                    self.stop = True

            if self.r_optimal_total - r_total < 0:
                self.stop_reason = "Stopped because regret fell below 0"
                self.stop = True


            print(self.dev_network_count ,' r_total and score: ', r_total , current_valid_score)
            
            self.logger.info("r_total: %.2f ", r_total)
            self.logger.info("bleu score: %.2f", current_valid_score)

            unfreeze_model(self.eval_net)

        return current_valid_score

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters (DQN) to a checkpoint.
        This function was taken from joeyNMT and was modified
        according to the needs of the Deep Q Network implementation.
        """

        if not os.path.exists(self.model_dir + "/dqn_model/"):
            os.makedirs(self.model_dir + "/dqn_model/")

        model_path = "{}/dqn_model/{}_dqn.ckpt".format(self.model_dir, self.dev_network_count)
        state = {
            "best_ckpt_score": self.best_ckpt_score,
            "model_state": self.eval_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(state, model_path)
        
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning("Wanted to delete old checkpoint %s but "
                                    "file does not exist.", to_delete)

        self.ckpt_queue.put(model_path)

        best_path = "{}/dqn_model/best_dqn.ckpt".format(self.model_dir)
        try:
            # create/modify symbolic link for best checkpoint
            symlink_update("{}_dqn.ckpt".format(self.dev_network_count), best_path)
        except OSError:
            # overwrite best.ckpt
            torch.save(state, best_path)


def dqn_train(cfg_file, ckpt: str, output_path: str = None) -> None:
    """
    Main wrapper function to execute all the training/dev of the DQN.
    It can be called from the console.
    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to save the model
    """
    MDQN = QManager(cfg_file, ckpt)
    MDQN.Collecting_experiences()
