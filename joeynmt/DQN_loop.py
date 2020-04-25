import math
from logging import Logger
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from joeynmt.helpers import bpe_postprocess, load_config, make_logger, get_latest_checkpoint, load_checkpoint, set_seed, log_cfg
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



class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, int(N_STATES/2))
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(int(N_STATES/2), int(N_STATES/2))
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(int(N_STATES/2), N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
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
        if self.state_type == 'hidden':
            self.state_size = cfg["model"]["encoder"]["hidden_size"]*2
        else:
            self.state_size = cfg["model"]["encoder"]["hidden_size"]
        self.actions_size = len(src_vocab)
        self.gamma = None
        # print("Sample size: ", self.sample_size )
        # print("State size: ", self.state_size)
        # print("Action size: ", self.actions_size)
        self.epochs = cfg["dqn"]["epochs"]
        # Inii the Qnet and Qnet2
        self.eval_net = Net(self.state_size, self.actions_size)
        self.target_net = Net(self.state_size, self.actions_size)
        #Following the algorithm
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = 0
        self.memory_counter = 0
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
        if self.use_cuda:
            self.model.cuda()

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

        if self.reward_type == "bleu_diff" :
            print("You select the reward based on the Bleu score differences")
            self.Reward = self.Reward_bleu_diff

        elif self.reward_type == "bleu_lin" :
            print("You select the reward based on the linear Bleu socres, and several punishments")
            self.Reward = self.Reward_lin

        elif self.reward_type == "bleu_fin" :
            print("You select the reward based on the final score on the last state ")
            self.Reward = self.Reward_bleu_fin

        elif self.reward_type == "bleu_hand" :
            print("You select the reward based on the improved version of _seq ")
            self.Reward = self.Reward_hand

        elif self.reward_type == "bleu_batch" :
            print("You select the reward based on the bleu_batch ")
            self.Reward = self.Reward_batch

        else:
            print("You select the reward based on the sequence accuaracy bleu_seq")
            self.Reward = self.Reward_seq

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
            self.gamma = 0.84
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
                    shuffle=False, train=False)
                #valid_sources_raw = data_set.src
                # disable dropout
                #self.model.eval()
                batch_i = 0

                for valid_batch in iter(valid_iter):
                    freeze_model(self.model)
                    batch = Batch(valid_batch
                    , self.pad_index, use_cuda=self.use_cuda)
                    # sort batch now by src length and keep track of order
                    sort_reverse_index = batch.sort_by_src_lengths()
                    encoder_output, encoder_hidden = self.model.encode(
                        batch.src, batch.src_lengths,
                        batch.src_mask)
                    # if maximum output length is not globally specified, adapt to src len

                    if self.max_output_length is None:
                        self.max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

                    batch_size = batch.src_mask.size(0)
                    prev_y = batch.src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                            dtype=torch.long)
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
                    prev_att_vector = encoder_output.new_zeros([batch_size, 1, self.state_size])
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
                            #state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]
                            state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()   # (batch_size,1,size_state)

                        else:
                            #state = prev_att_vector.squeeze(1).detach().cpu().numpy()[0]  # (1,size_state)
                            state = prev_att_vector.squeeze(1).detach().cpu().numpy()  # (batch_size,1,size_state)
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
                            # state_ = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]
                            state_ = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy() # (batch_size,1,size_state)

                        else:
                            # state_ = prev_att_vector.squeeze(1).detach().cpu().numpy()[0]
                            state_ = prev_att_vector.squeeze(1).detach().cpu().numpy()          # (batch_size,1,size_state)

                        # Checar que si sea el 3, despues del ultimo
                        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
                        # a = prev_y.squeeze(1).detach().cpu().numpy()[0]
                        a = prev_y.squeeze(1).detach().cpu().numpy()        # (batch_size ,1)
                        # if t < trg_.size:
                        #     a_ = trg_[t]
                        # else:
                        #     a_ = self.eos_index
                        a_ = 0

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
                            tup = (self.memory_counter, state, a, state_, torch.zeros(size=(batch_size, 1)), finished_aux)
                            exp_list.append(tup)
                            self.memory_counter += 1
                        #print(t)
                        # ** A cambiar para  cuando se usan Batches
                        # stop predicting if <eos> reached for all elements in batch

                        if flag_comp:
                            print("break with flag")
                            break

                        if (finished >= 1).sum() == batch_size:
                            flag_comp = True

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

                    if batch_i == 0:
                        self.store_transition(exp_list, r, sort_reverse_index, show =True)

                    else:
                        self.store_transition(exp_list, r, sort_reverse_index)

                    batch_i += 1

                    #Learning.....
                    if self.memory_counter > self.mem_cap:
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
                self.logger.info(' As referece this first test on dev data. Is maded with the Q networks, initialized randomly : ' )
                print('As referece this first test on dev data. Is maded with the Q networks, initialized randomly : ' )

            else:
                self.logger.info(" Copying the Q-Val on Q-tar on step: %d ", self.dev_network_count )
                print("\n Lets copy the Q-value Net in to Q-target net!. And test the performace on the dev data: ")

            current_bleu = self.dev_network()
            print("Current Bleu score is: ", current_bleu)

        self.learn_step_counter += 1
        long_Batch = self.sample_size*1

        # Sampling the higgest rewards values
        b_memory_big = self.memory[np.argsort(-self.memory[:-self.max_output_length, self.state_size+1])][:long_Batch]
        print("b_memory_big: ", b_memory_big)
        sample_index = np.random.choice(long_Batch, self.sample_size)
        b_memory = b_memory_big[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_size+2: self.state_size+2 + self.state_size])
        b_is_eos = torch.FloatTensor(b_memory[:, -1]).view(self.sample_size, 1)
        #print(b_a, b_a.size)
        #print(b_is_eos)
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
            b_a_ = torch.LongTensor(b_memory[:, self.state_size+2 + self.state_size]).view(self.sample_size, 1)

        else:
            if self.learn_step_counter == self.nu_pretrain:
                print ("Starting using Q target net....")
            b_a_ = torch.LongTensor(q_next.max(1)[1].view(self.sample_size, 1).long())

        #b_a_ = q_next.max(1)[0].view(self.sample_size, 1).long()   # shape (batch, 1)
        q_eval_next = self.eval_net(b_s_).gather(1, b_a_)   # shape (batch, 1)
        #If eos q_target = reward. 
        q_target = b_r + self.gamma * b_is_eos* q_eval_next.view(self.sample_size, 1)   # shape (batch, 1)

        #version 0
        #q_target = b_r + self.gamma * q_next.max(1)[0].view(self.sample_size, 1)   # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        self.tb_writer.add_scalar("learn/learn_batch_loss",
                                              loss.data, self.learn_step_counter)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #desctivate the eval_net
        freeze_model(self.eval_net)

    def store_transition(self, exp_list, rew, sort_reverse_index, show = False):
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
            index, state, a, state_, a_, finished  = ele
            #r = rew[sort_reverse_index,i]   # (size_batch, t)
            r = rew[:,i]   # (size_batch, t)
            #state = state[sort_reverse_index]
            #state_ = state_[sort_reverse_index]
            #a = a[sort_reverse_index]
            #finished = finished[sort_reverse_index]
            # print ("a = ",a)
            # print ("reward = ", r)
            # print ("finished =  ", finished)

            batch_size_aux = len(rew)

            for idx_batch in np.arange(batch_size_aux):
                # if show == True:
                #print(' ... s[:3]: ', state[:3], ' ... s_[:5]: ', state_[:3], )
                state_idx = state[idx_batch]
                a_idx = a[idx_batch]
                r_idx = r[idx_batch].numpy()
                state_idx_ = state_[idx_batch]
                a_idx_ = a_[idx_batch].numpy()

                if finished[idx_batch] < 2:
                    transition = np.hstack((state_idx, [a_idx, r_idx], state_idx_, a_idx_, 1))

                elif finished[idx_batch] == 2:
                    transition = np.hstack((state_idx, [a_idx, r_idx], state_idx_, a_idx_, 0))

                if finished[idx_batch] <= 2:
                    print("Debugg idx_batch = ", idx_batch)
                    print(index, a_idx, r_idx ,a_idx_, finished[idx_batch], "\n")
                    index = self.index_mem  % self.mem_cap
                    self.memory[index, :] = transition
                    self.index_mem += 1

    def Reward_batch(self, trg, hyp, show = False):
       #is_eos_index = False
        for i in range (len(hyp)):
            first_eos = np.where (hyp[i] == self.eos_index)

            if len(first_eos[0]) > 0:
                first_eos = first_eos[0][0]
                idx = np.arange (first_eos, len(hyp[0]))
                hyp[i][idx] = self.eos_index
                #is_eos_index = True

        decoded_valid_hyp = self.model.trg_vocab.arrays_to_sentences(arrays=hyp,
                                                    cut_at_eos=True)
        decoded_valid_tar = self.model.trg_vocab.arrays_to_sentences(arrays=trg,
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
        rew_distributed = distribute_reward(trg, hyp, blue_batch_score, self.eos_index)

        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target reward: ", decoded_valid_tar)
            print("Eval  reward: ", decoded_valid_hyp)
            print("Target reward: ", trg)
            print("Eval  reward: ", hyp)
            print("Bleu batch reward: ", blue_batch_score)
            print("Rew vector reward: ", rew_distributed)

        return rew_distributed

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
                        state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu()

                    else:
                        state = torch.FloatTensor(prev_att_vector.squeeze(1).detach().cpu().numpy())

                    if batch_i == 0:
                        print('So far: ', output, ' the state[:3] is: ', state[:3])

                    logits = self.eval_net(state)
                    batch_size_aux =  len(logits)
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

                r_total += torch.sum(r)

                if self.dev_network_count == 0:
                    extra_col = self.eos_index *torch.ones(len(batch.trg_input) , dtype=int).view(len(batch.trg_input),-1) 
                    trg_extra_col = torch.cat([batch.trg_input,extra_col ], dim= 1)
                    roptimal = self.Reward(batch.trg_input, trg_extra_col , show = False)
                    #print('roptimal: ', roptimal)
                    roptimal_total += torch.sum(roptimal)
                    #print('roptimal_total: ', roptimal_total)

                all_outputs.extend(stacked_output)
                all_outputs_to_bleu.extend(stacked_output_to_bleu)

                batch_i += 1

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

            print(self.dev_network_count ,' r_total and score: ', r_total , current_valid_score)
            
            self.logger.info("r_total: %.2f ", r_total)
            self.logger.info("bleu score: %.2f", current_valid_score)

            unfreeze_model(self.eval_net)

        return current_valid_score

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