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
import sacrebleu
import time
import datetime

from scipy.stats import entropy

import queue

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def distribute_reward(trg, hyp, blue_batch_score, eos_index):
    hyp_aux = np.array(hyp, copy=True)
    hyp = torch.from_numpy(hyp_aux).type(torch.LongTensor)
    batch_size_aux = len(hyp)
    # print("::trg: \n", trg.shape)
    # print("::hyp: ", hyp.shape)

    for i in range(len(hyp)):
        first_eos = np.where(hyp[i] == eos_index)
        if len(first_eos[0]) > 0:
            first_eos = first_eos[0][0]
            idx = np.arange(first_eos, len(hyp[0]))
            hyp[i][idx] = eos_index

    extra_col = eos_index * torch.ones(batch_size_aux, dtype=int).view(batch_size_aux, -1)
    trg = torch.cat([trg, extra_col], dim=1)

    if torch.sum(torch.sum(hyp == eos_index, dim=1) == 0) > 0:
        hyp = torch.cat([hyp, extra_col], dim=1)

    len_batch = (torch.sum(~torch.eq(hyp, eos_index), dim=1)).view(batch_size_aux, -1)
    hyp = hyp[:, 1:]
    trg = trg[:, 1:]
    mat_aux = -1 * torch.ones((batch_size_aux, max(len(hyp[0]), len(trg[0]))))

    if (len(hyp[0]) > len(trg[0])):
        mat_aux[:, :len(trg[0])] = trg
        matches = torch.eq(hyp, mat_aux)
    elif (len(hyp[0]) < len(trg[0])):
        mat_aux[:, :len(hyp[0])] = hyp
        matches = torch.eq(mat_aux, trg)
    else:
        is_same_len = True
        matches = torch.eq(hyp, trg)

    less_than = torch.tensor(np.arange(torch.max(len_batch))) * torch.ones(batch_size_aux, torch.max(len_batch))
    aux_counts = less_than < len_batch
    aux_counts_exc = -1 * (less_than >= len_batch)
    aux_counts_fit = aux_counts[:, :]
    matches_fit = matches[:, :len(hyp[0])]
    rew_one = torch.div(blue_batch_score * matches_fit, len_batch) * aux_counts_fit
    final_rew = rew_one + aux_counts_exc

    return final_rew


def funcMapSta_Ac(N_layers, state_size, actions_size):
    
    n_layers = np.arange(1,N_layers+1)
    
    ns = ((state_size-actions_size)/(1-N_layers)) * (n_layers -1) + state_size
    
    return ns.astype(int)

class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS, N_LAYERS):
        super(Net, self).__init__()
        
        net_size = funcMapSta_Ac(N_LAYERS, N_STATES, N_ACTIONS)
        
        print("The Net values are:")
        self.hidden = []
        for k in range(len(net_size)-2):
            self.hidden.append(nn.Linear(net_size[k], net_size[k+1]))
            self.hidden[k].weight.data.normal_(0, 0.1)   # initialization
            print(net_size[k], net_size[k+1])

        # Output layer
        self.out = nn.Linear(net_size[-2], net_size[-1])
        self.out.weight.data.normal_(0, 0.1)   # initialization
        print(net_size[-2], net_size[-1])

    def forward(self, x):
        
        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        actions_value = self.out(x)

        return actions_value


random.seed(10)

class QManager(object):
    """ Manages Q-learning loop, Agregate the .yalm parameters, 
        Initiate the model, test, Dev and Targets data.           
    """

    def __init__(self, cfg_file,
         ckpt: str,
         output_path: str = None,
         logger: Logger = None) -> None:

        """
        Recover the saved model, specified as in configuration.
        :param cfg_file: path to configuration file
        :param ckpt: path to checkpoint to load
        :param output_path: path to output
        :param logger: log output to this logger (creates new logger if not set)
        """

        cfg = load_config(cfg_file)

        if "test" not in cfg["data"].keys():
            raise ValueError("Test data must be specified in config.")

        #print(cfg.keys())
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

        set_seed(seed=cfg["training"].get("random_seed", 42))
        self.batch_size = cfg["training"]["batch_size"]
        self.batch_type = cfg["training"].get(
            "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
        self.use_cuda = cfg["training"].get("use_cuda", False)
        self.level = cfg["data"]["level"]
        self.eval_metric = cfg["training"]["eval_metric"]
        self.max_output_length = cfg["training"].get("max_output_length", None)
        # load the data
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])
         #Loading the DQN parameters:
        self.sample_size = cfg["dqn"]["sample_size"]
        self.lr = cfg["dqn"].get("lr", 0.01)
        self.egreed_max = cfg["dqn"].get("egreed_max", 0.9)
        self.egreed_min = cfg["dqn"].get("egreed_min", 0.01)
        self.gamma_max = cfg["dqn"].get("gamma_max", 0.9)
        self.gamma_min = cfg["dqn"].get("gamma_min", 0.5)
        self.nu_iter = cfg["dqn"]["nu_iter"]
        self.mem_cap = cfg["dqn"]["mem_cap"]
        self.beam_min = cfg["dqn"]["beam_min"]
        self.beam_max = cfg["dqn"]["beam_max"]
        self.state_type = cfg["dqn"]["state_type"]
        self.nu_pretrain = cfg["dqn"]["nu_pretrain"]
        self.reward_type = cfg["dqn"]["reward_type"]
        self.count_post_pre_train = 0
        self.hidden_size = cfg["model"]["encoder"]["hidden_size"]
        if self.state_type == 'hidden' and cfg["model"]["encoder"]["bidirectional"]:
            self.state_size = self.hidden_size*2
        else:
            self.state_size = self.hidden_size

        self.N_layers = cfg["dqn"]["N_layers"]

        self.actions_size = len(src_vocab)
        self.gamma = None
        # print("Sample size: ", self.sample_size )
        # print("State size: ", self.state_size)
        # print("Action size: ", self.actions_size)
        self.epochs = cfg["dqn"]["epochs"]
        # Inii the Qnet and Qnet2
        self.eval_net = Net(self.state_size, self.actions_size, self.N_layers)
        self.target_net = Net(self.state_size, self.actions_size, self.N_layers)
        #Following the algorithm
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = 0
        #self.memory_counter = 0
        self.size_memory1 = self.state_size * 2 + 2 + 2
        self.memory = np.zeros((self.mem_cap, self.size_memory1 ))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters()
                                          , lr=self.lr )
        self.loss_func = nn.MSELoss()
        #others parameters
        self.bos_index = trg_vocab.stoi[BOS_TOKEN]
        self.eos_index = trg_vocab.stoi[EOS_TOKEN]
        self.pad_index = trg_vocab.stoi[PAD_TOKEN]
        self.data_to_train_dqn = {"train": train_data}
        #self.data_to_train_dqn = {"test": test_data}
        self.data_to_dev = {"train": train_data}
        #self.data_to_dev = {"dev": dev_data, "train": train_data}
        #self.data_to_train_dqn = {"train": train_data
        #                          ,"dev": dev_data, "test": test_data}
        # load model state from disk
        model_checkpoint = load_checkpoint(ckpt, use_cuda=self.use_cuda)
        # build model and load parameters into it
        self.model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
        self.model.load_state_dict(model_checkpoint["model_state"])
        

        self.best_ckpt_score = -np.inf
        # comparison function for scores
        self.is_best = lambda score: score > self.best_ckpt_score
        self.ckpt_queue = queue.Queue(maxsize=2)
        

        if self.use_cuda:
            self.model.cuda()
            self.eval_net.cuda()
            self.target_net.cuda()
            self.loss_func.cuda()



        # whether to use beam search for decoding, 0: greedy decoding
        beam_size = 1
        beam_alpha = -1
        # get the current date to write the folder for tensorboard
        time_stamp = time.time()
        date = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
        # get the hyperparameters more relevant
        # add as many hyperparameters as desired in the following manner:
        # -write the name of the hyperparameter at front then the value
        # -write underscore after each hyperparameter, except for the latest one
        relevant_hyp = "nu_pretrain=" + str(self.nu_pretrain) + "_" + "reward_type=" + str(self.reward_type) + cfg["dqn"]["other_descrip"]
        #others not important parameters
        self.index_fin = None
        # construct the name of the folder for tensorboard for the test given the date and the relevant_hyp
        path_tensroboard = self.model_dir + "/tensorboard_DQN/" + date + "/" + relevant_hyp + "/"
        self.tb_writer = SummaryWriter( log_dir=path_tensroboard , purge_step=0)
        if not os.path.exists(self.model_dir + "/logs/" + date + "/"):
            os.makedirs(self.model_dir + "/logs/" + date + "/")

        path_logger = self.model_dir + "/logs/" + date + "/" + relevant_hyp + ".log"
        self.logger = make_logger(path_logger) 
        self.dev_network_count = 0
        self.r_optimal_total = 0
        log_cfg(cfg, self.logger)
        self.index_mem = 0
        self.logger.info("We are using the reward named: %s", self.reward_type)
        #Reward funtion related:
        self.logger.info("As states we set the : %s", self.state_type)
        
        if self.reward_type == "bleu_batch":
            print("You select the reward based on the bleu_batch ")
            self.Reward = self.Reward_batch


    def Collecting_experiences(self)-> None:
        """
        Main funtion. Compute all the process.
        :param exp_list: List of experineces. Tuples (memory_counter, state, a, state_, is_eos[0,0])
        :param rew: rewards for every experince. Of lenght of the hypotesis
        """

        for epoch_no in range(self.epochs):
            print("EPOCH %d", epoch_no + 1)
            self.logger.info("EPOCH %d", epoch_no + 1)
            #beam_dqn = self.beam_min + int(self.beam_max * epoch_no/self.epochs)
            #egreed = self.egreed_max*(1 - epoch_no/(1.1*self.epochs))
            #self.gamma = self.gamma_max*(1 - epoch_no/(2*self.epochs))
            # keep the beam_dqn = 1, otherwise is harmfull to the learning
            beam_dqn = 1
            self.gamma = 0.99
            if self.learn_step_counter < self.nu_pretrain:
                # print("On the pretrain of the Q target network. The beam_dqn =1.")
                beam_dqn = 1
                egreed = self.egreed_max
                self.gamma = (self.gamma_min + self.gamma_max)/2
            else:
                self.count_post_pre_train += 1
                # beam_dqn = int(beam_dqn*math.pow(1.001,self.count_post_pre_train))
                egreed = self.egreed_max*math.pow(0.999, self.count_post_pre_train)
                # self.gamma = self.gamma_max*math.pow(0.999, self.count_post_pre_train)

                if egreed < self.egreed_min:
                    egreed = self.egreed_min

                if self.gamma < self.gamma_min:
                    self.gamma = self.gamma_min

                if beam_dqn > self.actions_size:
                    print("The beam_dqn cannot exceed the action size!")
                    print("then the beam_dqn = action size")
                    beam_dqn = self.actions_size - 1

            self.tb_writer.add_scalar("parameters/beam_dqn",
                                              beam_dqn, epoch_no)
            self.tb_writer.add_scalar("parameters/egreed",
                                              egreed, epoch_no)
            self.tb_writer.add_scalar("parameters/gamma",
                                              self.gamma, epoch_no)
            self.logger.info(' beam_dqn : %d, egreed: %.2f, gamma:  %.2f', beam_dqn, egreed, self.gamma)
            print(' beam_dqn, egreed, gamma: ', beam_dqn, egreed, self.gamma)

            for _, data_set in self.data_to_train_dqn.items():
                valid_iter = make_data_iter(
                    dataset=data_set, batch_size=self.batch_size, batch_type=self.batch_type,
                    shuffle=True, train=True)
                #valid_sources_raw = data_set.src
                # disable dropout
                #self.model.eval()
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
                                            dtype=torch.long)
                    #self.logger.info("self.bos_index: %d", self.bos_index)
                    
                    output = []
                    hidden = self.model.decoder._init_hidden(encoder_hidden)
                    prev_att_vector = None
                    finished = batch.src_mask.new_zeros((batch_size, 1)).byte()
                    #print("Source_raw: ", batch.src)
                    #print("Target_raw: ", batch.trg_input)
                    # get the raw vector in order to use it
                    # later in the learning for the true actions
                    #trg_ = batch.trg.cpu().detach().numpy().squeeze()
                    # batch.trg [0,5,3]  -> 5 7
                    # batch.trg_imp [2,0,5,3]  -> <bos> 5 7
                    # #print ("trg numpy: ", trg_input_np)
                    # print("y0: ", prev_y)
                    exp_list = []
                    # pylint: disable=unused-variable

                    # * Defining it as zeros:____------------------------------------
                    prev_att_vector = encoder_output.new_zeros([batch_size, 1, self.hidden_size])
                    #----------------------------------------------------------------
                    #We can try 2 options (using a state 0 from attention):
                    # * Generating a previus one:------------------------------------
                    # use new (top) decoder layer as attention query
                    # if isinstance(hidden, tuple):
                    #     query = hidden[0][-1].unsqueeze(1)
                    # else:
                    #     query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
                    # # compute context vector using attention mechanism
                    # # only use last layer for attention mechanism
                    # # key projections are pre-computed
                    # context, att_probs = self.attention(
                    #     query=query, values=encoder_output, mask=batch.src_mask)
                    # # return attention vector (Luong)
                    # # combine context with decoder hidden state before prediction
                    # att_vector_input = torch.cat([query, context], dim=2)
                    # # batch x 1 x 2*enc_size+hidden_size
                    # att_vector_input = self.hidden_dropout(att_vector_input)
                    # prev_att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

                    # * Generating a previus one:------------------------------------
                    flag_comp = False
                    output.append(prev_y.squeeze(1).detach().cpu().numpy())
                    for t in range(self.max_output_length):
                        if self.state_type == 'hidden':
                            #state = hidden[0].squeeze(1).detach().cpu().numpy()
                            #state = hidden[1].squeeze(1).detach().cpu().numpy()
                            state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]
                            #print(" state shape: ", state.shape) # (batch_size,size_state)

                        else:
                            state = prev_att_vector.squeeze(1).detach().cpu().numpy()  # (batch_size,size_state)
                            #print(" state shape: ", state.shape) # (batch_size,size_state)

                        # decode one single step
                        logits, hidden, att_probs, prev_att_vector = self.model.decoder(
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask,
                            trg_embed=self.model.trg_embed(prev_y),
                            hidden=hidden,
                            prev_att_vector=prev_att_vector,
                            unroll_steps=1)
                        # logits: batch x time=1 x vocab (logits)

                        if self.state_type == 'hidden':
                            state_ = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0] # (batch_size,size_state)
                            #print(" state_ shape: ", state_.shape) 
                        else:
                            state_ = prev_att_vector.squeeze(1).detach().cpu().numpy()          # (batch_size,size_state)
                            #print(" state_ shape: ", state_.shape) 

                        # Checar que si sea el 3, despues del ultimo
                        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
                        # a = prev_y.squeeze(1).detach().cpu().numpy()[0]
                        a = prev_y.squeeze(1).detach().cpu().numpy()        # (batch_size ,1)
                        
                        a_ = next_word.squeeze(1).detach().cpu().numpy()        # (batch_size ,1)
                        # if t < trg_.size:
                        #     a_ = trg_[t]
                        # else:
                        #     a_ = self.eos_index
                        
                        if not flag_comp:
                            output.append(next_word.squeeze(1).detach().cpu().numpy())

                        prev_y = next_word
                        # check if previous symbol was <eos>
                        is_eos = torch.eq(next_word, self.eos_index)        # (batch_size ,1)
                        finished += is_eos
                        #print('finished main func:... ',t, finished)
                        # for i, ele in enumerate(is_eos):
                        #     if ele == True:
                        #         tup = (self.memory_counter, state[i], a[i], state_[i], 0, 0)
                        #         exp_list.append(tup)
                        #         self.memory_counter += 1

                        #Lo comentamos, para solo considerar la ultima accion.
                        if t > 0:
                            #print ("a = ", a)
                            finished_aux = finished.clone()
                            #if flag_comp:
                            #    tup = (self.memory_counter, state, a, torch.zeros(size=(batch_size, 1)), torch.zeros(size=(batch_size, 1)), finished_aux)
                            #else:
                            tup = (state, a, state_, a_, finished_aux)
                            exp_list.append(tup)
                            #self.memory_counter += 1
                        #print(t)
                        # ** A cambiar para  cuando se usan Batches
                        # stop predicting if <eos> reached for all elements in batch

                        # if flag_comp:
                        #     #self.logger("Break with flag complete.")
                        #     print("break with flag")
                        #     break

                        if (finished >= 1).sum() == batch_size:
                            # flag_comp = True
                            finished_aux = finished.clone() + 1
                            #if flag_comp:
                            #    tup = (self.memory_counter, state, a, torch.zeros(size=(batch_size, 1)), torch.zeros(size=(batch_size, 1)), finished_aux)
                            #else:
                            tup = (state_, self.eos_index*torch.ones(batch_size, 1 ) , torch.zeros(size=(batch_size, self.state_size )),
                                   self.pad_index * torch.ones(size=(batch_size, 1)), finished_aux)
                            exp_list.append(tup)
                            
                            break

                        if t == self.max_output_length-1:
                            print("reach the max output")
                            break
                            # a = next_word.squeeze(1).detach().cpu().numpy()[0]
                            # tup = (self.memory_counter, state_, a, -1*np.ones([self.state_size]), 0, 0)
                            # exp_list.append(tup)
                            # self.memory_counter += 1

                    #Collecting rewards
                    hyp = np.stack(output, axis=1)  # batch, time
                    # if epoch_no == 0 and batch_i < 6:
                    #     r = self.Reward(batch.trg_input, hyp, show=True)  # 1 , time-1    # batch, time-1
                    # else:
                    #     r = self.Reward(batch.trg_input, hyp, show=False)  # 1 , time -1 

                    if epoch_no == 0 and batch_i == 0:
                        r = self.Reward(batch.trg_input, hyp, show=True)  # 1 , time-1    # batch, time-1
                        print ("Function Collecting_experiences")

                    else:
                        r = self.Reward(batch.trg_input, hyp, show=False)  # 1 , time -1 

                    if epoch_no == 0 and batch_i == 0:
                        self.store_transition(exp_list, r, show =True)

                    else:
                        self.store_transition(exp_list, r)

                    batch_i += 1

                    #Learning.....
                    if self.index_mem > self.mem_cap:
                        #if self.dev_network_count < 13:
                        self.learn()
                    # else:
                    #         freeze_model(self.eval_net)
        self.tb_writer.close()       



    def learn(self):
        """
        Select experinces based on the reward. And compute the bellman eqution and the ACC model.
        Algoritm 5. from paper https://arxiv.org/pdf/1805.09461v1.pdf like. Without the actor updating.
        :param exp_list: List of experineces. Tuples (memory_counter, state, a, state_, is_eos[0,0])
        :param rew: rewards for every experince. Of lenght of the hypotesis
        """
        # target parameter update
                # target parameter update
        if self.learn_step_counter % self.nu_iter == 0:
            #if self.dev_network_count < 13:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            #testing the preformace of the network

            if self.learn_step_counter == 0:
                self.logger.info(' As referece this first test on dev data. Is maded with the Q networks, initialized randomly : ')
                print('As referece this first test on dev data. Is maded with the Q networks, initialized randomly : ' )

            else:
                self.logger.info(" Copying the Q-Val on Q-tar on step: %d ", self.dev_network_count )
                print("\n Lets copy the Q-value Net in to Q-target net!. And test the performace on the dev data: ")

            current_bleu = self.dev_network()
            print("Current Bleu score is: ", current_bleu)

            if self.is_best(current_bleu):
                self.best_ckpt_score = current_bleu
                self.logger.info(
                    'Hooray! New best validation result [%f]!', current_bleu)
                print('Hooray! New best validation result :', current_bleu)
                self._save_checkpoint()

        self.learn_step_counter += 1
        long_Batch = self.sample_size*10

        # Sampling the higgest rewards values
        b_memory_big = self.memory[np.argsort(-self.memory[:, self.state_size+1])][:long_Batch]
        # print("b_memory_big: ", b_memory_big)
        sample_index = np.random.choice(long_Batch, self.sample_size)
        b_memory = b_memory_big[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_size+2: self.state_size+2 + self.state_size])
        b_is_eos = torch.FloatTensor(b_memory[:, -1]).view(self.sample_size, 1)

        # print("b_s",b_s.shape, "\n", b_s[:5], )
        # print("b_s_",b_s_.shape, "\n", b_s_[:5], )
        # print("b_a: ", b_a, b_a.shape)
        # print("b_r: ", b_r, b_r.shape)
        # print("b_is_eos: ", b_is_eos, b_is_eos.shape)
        
        #Activate the eval_net
        unfreeze_model(self.eval_net)
       # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate

        # taking the most likely action.
        # use the hyperparameter nu_pretrain to take the true action
        # or the one take from the one computed from the q_target

        if self.learn_step_counter % 50 == 1:
            print ("learn step counter: ", self.learn_step_counter)
            print ("dev_network_count: ", self.dev_network_count )

        if self.learn_step_counter < self.nu_pretrain:
            if self.learn_step_counter == 1:
                print ("Using pretraining...")
            b_is_eos = ~torch.eq(b_a, self.eos_index)
            
            b_a_ = torch.LongTensor(b_memory[:, self.state_size+2 + self.state_size]).view(self.sample_size, 1)

        else:
            if self.learn_step_counter == self.nu_pretrain:
                print ("Starting using Q target net....")
            b_a_ = torch.LongTensor(q_next.max(1)[1].view(self.sample_size, 1).long())

        #b_a_ = q_next.max(1)[0].view(self.sample_size, 1).long()   # shape (batch, 1)
        q_eval_next = self.eval_net(b_s_).gather(1, b_a_)   # shape (batch, 1)
        #If eos q_target = reward. 
        q_target = b_r + self.gamma * b_is_eos* q_eval_next.view(self.sample_size, 1)   # shape (batch, 1)

 

        soft_func = torch.nn.Softmax(dim = -1)
        #print(soft_func(a))
        
        q_eval_all = self.eval_net(b_s).detach()

        q_eval_max = torch.LongTensor(q_eval_all.max(1)[1].view(self.sample_size, 1).long())

        print ("q_eval_all: ")
        print (q_eval_all[:10])
        print ("q_eval_max: ")
        print (q_eval_max[:10])
        print ("b_a:")
        print (b_a[:10]) 

        entro = entropy(soft_func(q_eval_all).T, base=self.actions_size)
        #print("entropy: ", entro)
        aver_entro = entro.sum()/self.sample_size
        self.tb_writer.add_scalar("learn/q_eval_entropy",
                        aver_entro, self.learn_step_counter)


        loss = self.loss_func(q_eval, q_target)
        
        self.tb_writer.add_scalar("learn/learn_batch_loss",
                                              loss.data, self.learn_step_counter)

        if loss < (1.5 * (10 ** (-4))):
            self.stop_reason = "Stopped because loss shrinking too slowly"
            self.stop = True

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #desctivate the eval_net
        freeze_model(self.eval_net)

    def store_transition(self, exp_list, rew, show = False):
        """
        Fill/ or refill the memory with experiences.
        :param exp_list: List of experineces. Tuples (memory_counter, state, a, state_, is_eos[0,0])
        :param rew: rewards for every experince. Of lenght of the hypotesis        
        """

        if len(exp_list) != len(rew[0]):
            print(len(exp_list) , len(rew[0]))
            print(' exp_list: ', exp_list)
            print(' rew: ', rew)

        assert (len(exp_list) == len(rew[0]))
        # tup = (self.memory_counter, state, a, state_, torch.zeros(size=(batch_size, 1)), finished)
        #rew (size_batch, t)
        #self.index_t = 0

        for i, ele in enumerate(exp_list):
            state, a, state_, a_, finished  = ele
            r = rew[:,i]   # (size_batch, t)
            # print ("a = ",a)
            # print ("a_ = ",a_)
            # print ("reward = ", r)
            # print ("finished =  ", finished)

            batch_size_aux = len(rew)

            for idx_batch in np.arange(batch_size_aux):
                # if show == True:
                #print(' ... s[:3]: ', state[:3], ' ... s_[:5]: ', state_[:3], )
                state_idx = state[idx_batch]
                a_idx = a[idx_batch]
                r_idx = r[idx_batch]#.numpy()
                state_idx_ = state_[idx_batch]
                a_idx_ = a_[idx_batch]
            
                if finished[idx_batch] < 2:
                    transition = np.hstack((state_idx, [a_idx, r_idx], state_idx_, a_idx_, 1))
                else:
                    transition = np.hstack((state_idx, [a_idx, r_idx], state_idx_, a_idx_, 1))
                
                # if finished[idx_batch] <= 2:
                #     # print("Debugg idx_batch = ", idx_batch)
                #     # print( a_idx, r_idx ,a_idx_, finished[idx_batch], "\n")
                index = self.index_mem  % self.mem_cap
                self.memory[index, :] = transition
                self.index_mem += 1

    def Reward_batch(self, trg, hyp, show = False):
       #is_eos_index = False

        batch_size_aux = len(hyp)
        extra_col = self.eos_index * torch.ones(batch_size_aux, dtype=int).view(batch_size_aux, -1)
        trg = torch.cat([trg, extra_col], dim=1)
        # for i in range (len(hyp)):
        #     first_eos = np.where (hyp[i] == self.eos_index)

        #     if len(first_eos[0]) > 0:
        #         first_eos = first_eos[0][0]
        #         idx = np.arange (first_eos, len(hyp[0]))
        #         hyp[i][idx] = self.eos_index
        #         #is_eos_index = True

        rew_distributed = np.zeros([batch_size_aux, len(hyp[0]) -1 ])

        # print("hyp",type(hyp),"\n", hyp)
        # print("trg",type(trg),"\n", trg)

        for i in np.arange(batch_size_aux):
            
            hyp_i = hyp[i]
            trg_i = np.asarray(trg[i])
            # trg_i = trg[i]
            
            # print("hyp_i: ", type(hyp_i), hyp_i.shape, '\n', hyp_i.reshape([1,-1]))
            # print("trg_i: ", type(trg_i), trg_i.shape, '\n', trg_i.reshape([1,-1]))

            
            decoded_valid_hyp = self.model.trg_vocab.arrays_to_sentences(arrays=hyp_i.reshape([1,-1]),
                                                        cut_at_eos=True)
            decoded_valid_tar = self.model.trg_vocab.arrays_to_sentences(arrays=trg_i.reshape([1,-1]),
                                                        cut_at_eos=True)

            # evaluate with metric on each src, tar, and hypotesis
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

            blue_batch_score = bleu(valid_hypotheses, valid_references)
            rew_distributed[i, -1] =  blue_batch_score


            # rew_distributed = distribute_reward(trg, hyp, blue_batch_score, self.eos_index)

        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target reward: ", decoded_valid_tar)
            print("Eval  reward: ", decoded_valid_hyp)
            print("Target reward: ", trg)
            print("Eval  reward: ", hyp)
            print("Bleu batch reward: ", blue_batch_score)
            print("Rew vector reward: ", rew_distributed)

        return rew_distributed


    def Reward_seq(self, trg, hyp, show = False):
        
        print("trg, hyp")
        print(trg.shape, hyp.shape)
        
        trg_a = np.asarray(trg[0])
        trg_b = np.zeros([len(trg_a)+1], dtype = int)
        trg_b[:len(trg_a)] = trg_a
        trg_b[-1] = 3

        if len(trg_b) != len(hyp[0]):
            trg_c = np.ones([len(hyp[0])], dtype = int)*(self.actions_size+1)
            if len(trg_b) > len(hyp[0]):
                lon = len(hyp[0])
            else:
                lon = len(trg_b)
            trg_c[:lon] = trg_b[:lon] 
            #print(trg_c[:] == hyp[0,:])
            trg_b = trg_c

        bolles = trg_b[:] == hyp[0,:]

        bolles = bolles*np.ones(len(trg_b)) + np.zeros(len(trg_b))
        
        # Punisment to longer or shorter hyp than trg
        def fac_long_punish(x, len_tran):
            return -np.abs(-(x/len_tran) +1)

        #*4*original* Punish increas as the len increase. The best so far!
        #To fit on the diff computation idea.
        final_rew = bolles*np.arange(1,len(trg_b)+1)
        final_rew = np.diff(final_rew) -np.arange(len(hyp[0])-1)*0.2

        #To punish the wrong desitions, but the last one (to avoid the large hyp)
        for i in np.arange(1,len(final_rew)-1):
            final_rew[i] = (final_rew[i]+final_rew[i-1])/2.0

        # #*4_1* Include the original ** Best so far (12/04/20). Better regret.
        # # Penalizing the second token when goes worng:
        if len(hyp[0]) > 2:
            if trg_b[1] != hyp[0,1] and trg_b[2] == hyp[0,2]:
                final_rew[1] = final_rew[1]*0.5
  
        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target: ", trg_b)
            print("Eval  : ", hyp)
            print("Reward: ", final_rew, "\n")
            print(trg, trg_a)
        return final_rew


    def dev_network(self):
        """
        Show how is the current performace over the dev dataset, by mean of the
        total reward and the belu score.
        :return: current Bleu score
        """

        freeze_model(self.eval_net)
        for data_set_name, data_set in self.data_to_dev.items():
            #print(data_set_name)
            valid_iter = make_data_iter(
                dataset=data_set, batch_size=self.batch_size, batch_type=self.batch_type,
                shuffle=False, train=False)
            valid_sources_raw = data_set.src

            # don't track gradients during validation
            r_total = 0
            roptimal_total = 0
            all_outputs = []
            all_outputs_to_bleu = []
            batch_i = 0

            aver_entro_list = []

            for valid_batch in iter(valid_iter):
                # run as during training to get validation loss (e.g. xent)
                batch = Batch(valid_batch, self.pad_index, use_cuda=self.use_cuda)
                # sort batch now by src length and keep track of order
                sort_reverse_index = batch.sort_by_src_lengths()
                encoder_output, encoder_hidden = self.model.encode(
                    batch.src, batch.src_lengths,
                    batch.src_mask)

                # if maximum output length is 
                # not globally specified, adapt to src len

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

                #prev_att_vector = encoder_output.new_zeros([batch_size, 1, self.state_size])
                output.append(prev_y.squeeze(1).detach().cpu().numpy())
                # pylint: disable=unused-variable

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

                    # greedy decoding: choose arg max over vocabulary in each step with egreedy porbability
                    if self.state_type == 'hidden':
                        state = torch.cat(hidden, dim=2).squeeze(1).detach()[0]

                    else:
                        state = torch.FloatTensor(prev_att_vector.squeeze(1).detach().cpu().numpy())

                    if batch_i == 0:
                        print('So far: ', output, ' the state[:3] is: ', state[:3])

                    logits = self.eval_net(state)
                    batch_size_aux =  len(logits)


                    soft_func = torch.nn.Softmax(dim = -1)

                    entro = entropy(soft_func(logits.detach()).T, base=self.actions_size)
                    aver_entro = entro.sum()/batch_size_aux
                    aver_entro_list.append(aver_entro)

                    logits = logits.reshape([batch_size_aux, 1, -1])
                    #print(type(logits), logits.shape, logits)
                    next_word = torch.argmax(logits, dim=-1)                        
                    output.append(next_word.squeeze(1).detach().cpu().numpy())
                    output_to_blue.append(next_word.squeeze(1).detach().cpu().numpy())
                    prev_y = next_word
                    # check if previous symbol was <eos>
                    is_eos = torch.eq(next_word, self.eos_index)
                    finished += is_eos
                    # stop predicting if <eos> reached for all elements in batch

                    if (finished >= 1).sum() == batch_size:
                        break

                stacked_output = np.stack(output, axis=1)  # batch, time
                stacked_output_to_bleu = np.stack(output_to_blue, axis=1)  # batch, time
                #decode back to symbols
                decoded_valid_in = self.model.trg_vocab.arrays_to_sentences(arrays=batch.src,
                                            cut_at_eos=True)
                decoded_valid_out_trg = self.model.trg_vocab.arrays_to_sentences(arrays=batch.trg,
                                                                    cut_at_eos=True)
                decoded_valid_out = self.model.trg_vocab.arrays_to_sentences(arrays=stacked_output,
                                            cut_at_eos=True)
                hyp = stacked_output

                if batch_i == 0:
                    self.logger.info("Currently on the prediction.")
                    r = self.Reward(batch.trg_input, hyp , show = True)
                    print ("Function dev_network")

                else:
                    r = self.Reward(batch.trg_input, hyp , show = False)

                r_total += np.sum(r)

                if self.dev_network_count == 0:
                    extra_col = self.eos_index *torch.ones(len(batch.trg_input) , dtype=int).view(len(batch.trg_input),-1) 
                    trg_extra_col = torch.cat([batch.trg_input,extra_col ], dim= 1)
                    roptimal = self.Reward(batch.trg_input, trg_extra_col , show = False)
                    #print('roptimal: ', roptimal)
                    roptimal_total += np.sum(roptimal)
                    #print('roptimal_total: ', roptimal_total)

                all_outputs.extend(stacked_output)
                all_outputs_to_bleu.extend(stacked_output_to_bleu)

                batch_i += 1

            aver_entro = sum(aver_entro_list)/len(aver_entro_list)


            if self.dev_network_count == 0:
                self.logger.info("Optimal reward is: %.2f", roptimal_total)
                print('The optimal reward is: ', roptimal_total)
                self.r_optimal_total = roptimal_total

            assert len(all_outputs) == len(data_set)

            # decode back to symbols
            decoded_valid = self.model.trg_vocab.arrays_to_sentences(arrays=all_outputs_to_bleu,
                                                                cut_at_eos=True)

           # evaluate with metric on full dataset
            join_char = " " if self.level in ["word", "bpe"] else ""
            valid_sources = [join_char.join(s) for s in data_set.src]
            valid_references = [join_char.join(t) for t in data_set.trg]
            valid_hypotheses = [join_char.join(t) for t in decoded_valid]

            #print('On dataset: ', data_set.trg[:10])
            self.logger.info('On %s', data_set_name)
            self.logger.info('valid_references \t vs \t predicted_hypotheses')

            for i in np.arange(10):
                self.logger.info( ' %s\t vs %s\t', valid_references[i],  valid_hypotheses[i])

            for i in np.arange(10):
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

                current_valid_score = 0
                if self.eval_metric.lower() == 'bleu':
                    # this version does not use any tokenization
                    current_valid_score = bleu(valid_hypotheses, valid_references)

                elif self.eval_metric.lower() == 'chrf':
                    current_valid_score = chrf(valid_hypotheses, valid_references)

                elif self.eval_metric.lower() == 'token_accuracy':
                    current_valid_score = token_accuracy(
                        valid_hypotheses, valid_references, level=self.level)

                elif self.eval_metric.lower() == 'sequence_accuracy':
                    current_valid_score = sequence_accuracy(
                        valid_hypotheses, valid_references)

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


            print(self.dev_network_count ,' r_total and score: ', r_total , current_valid_score)
            
            self.logger.info("r_total: %.2f ", r_total)
            self.logger.info("bleu score: %.2f", current_valid_score)

            unfreeze_model(self.eval_net)

        return current_valid_score

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

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
    Interactive state function.  Just to understand how looks like the states
    Loads model from checkpoint and show states either the stdin input or
    asks for input to get states interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.
    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    """

    #cfg_file = "./reverse_model/config.yaml"
    #cfg_file = "./configs/reverse.yaml" 
    #MDQN = QManager(cfg_file, "./reverse_model/best.ckpt")
    MDQN = QManager(cfg_file, ckpt)
    MDQN.Collecting_experiences()