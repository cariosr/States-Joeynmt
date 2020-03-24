

from logging import Logger
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

#from joeynmt.DQN_utils import DQN
from joeynmt.helpers import bpe_postprocess, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint
from joeynmt.data import load_data, make_data_iter
from joeynmt.model import build_model, Model
from joeynmt.vocabulary import Vocabulary
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.batch import Batch
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
from joeynmt.prediction import validate_on_data

import random

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_STATES-10)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization

        self.fc2 = nn.Linear(N_STATES-10, N_STATES-20)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization

        self.out = nn.Linear(N_STATES-20, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


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

        if logger is None:
            logger = make_logger()

        cfg = load_config(cfg_file)

        if "test" not in cfg["data"].keys():
            raise ValueError("Test data must be specified in config.")

        #print(cfg.keys())
        if "dqn" not in cfg.keys():
            raise ValueError("dqn data must be specified in config.")

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


        self.batch_size = 1 #**
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
        self.state_size = cfg["model"]["encoder"]["hidden_size"]*2
        self.actions_size = len(src_vocab)

        self.epochs = cfg["dqn"]["epochs"]

        # Inii the Qnet and Qnet2
        self.eval_net = Net(self.state_size, self.actions_size)
        self.target_net = Net(self.state_size, self.actions_size)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.mem_cap, self.state_size * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters()
                                          , lr=self.lr )
        self.loss_func = nn.MSELoss()

        #others parameters
        self.bos_index = trg_vocab.stoi[BOS_TOKEN]
        self.eos_index = trg_vocab.stoi[EOS_TOKEN]
        self.pad_index = trg_vocab.stoi[PAD_TOKEN]

        self.data_to_train_dqn = {"test": test_data}
        self.data_to_dev = {"dev": dev_data}
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
        
    def Collecting_experiences(self)-> None:
        for epoch_no in range(self.epochs):
            print("EPOCH %d", epoch_no + 1)

            self.beam_qdn = self.beam_min + int(self.beam_max * epoch_no/self.epochs)
            self.egreed = self.egreed_max*(1 - epoch_no/(1.1*self.epochs))
            self.gamma = self.gamma_max*(1 - epoch_no/(2*self.epochs))

            print(' beam_qdn, egreed, gamma: ', self.beam_qdn, self.egreed, self.gamma)
            for data_set_name, data_set in self.data_to_train_dqn.items():
                valid_iter = make_data_iter(
                    dataset=data_set, batch_size=1, batch_type=self.batch_type,
                    shuffle=False, train=False)
                valid_sources_raw = data_set.src
                # disable dropout
                self.model.eval()
                # don't track gradients during validation
                with torch.no_grad():
                    for valid_batch in iter(valid_iter):
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
                        output = []
                        attention_scores = []
                        hidden = self.model.decoder._init_hidden(encoder_hidden)
                        prev_att_vector = None
                        finished = batch.src_mask.new_zeros((batch_size, 1)).byte()

                        exp_list = []
                        # pylint: disable=unused-variable
                        for t in range(self.max_output_length):
                            state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]
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
                            state_ = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]

                            # greedy decoding: choose arg max over vocabulary in each step with egreedy porbability

                            if random.uniform(0, 1) < self.egreed_max:
                                i_ran = random.randint(0,self.beam_qdn-1)
                                next_word = torch.argsort(logits, descending=True)[:, :, i_ran]
                            else:
                                next_word = torch.argmax(logits, dim=-1)  # batch x time=1

                            a = next_word.squeeze(1).detach().cpu().numpy()[0]

                            output.append(next_word.squeeze(1).detach().cpu().numpy())
                            tup = (self.memory_counter, state, a, state_)
                            self.memory_counter += 1
                        
                            exp_list.append(tup)
                            prev_y = next_word
                            # check if previous symbol was <eos>
                            is_eos = torch.eq(next_word, self.eos_index)
                            finished += is_eos
                            # stop predicting if <eos> reached for all elements in batch
                            if (finished >= 1).sum() == batch_size:
                                break
                        
                        #Collecting rewards
                        hyp = np.stack(output, axis=1)  # batch, time
                        r = self.Reward(batch.trg, hyp )  # 1 , time  
                        self.store_transition(exp_list, r)
                        
                        #Learning.....
                        if self.memory_counter > self.mem_cap - self.max_output_length:
                            self.learn()
                            # initialize memory
                            self.memory = np.zeros((self.mem_cap, self.state_size * 2 + 2))   
                            self.memory_counter = 0
                            print('-----------------------------------------------------------------------' )
                            #break

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.nu_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            #testing the preformace of the network
            if self.learn_step_counter != 0:
                self.dev_network()
        self.learn_step_counter += 1

        # Sampling the higgest rewards values
        b_memory = self.memory[np.argsort(-self.memory[:, self.state_size+1])][:self.sample_size]
        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_size:])

        print('We choose the following rewads: ', b_r )

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.sample_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        print(loss.data)
        loss.requires_grad = True
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss.requires_grad = False

        for param in  self.eval_net.parameters():
            print(param.data)

    def store_transition(self, exp_list, rew):
        assert (len(exp_list) == len(rew) )
        for i, ele in enumerate(exp_list):
            index, state, a, state_ = ele
            r = rew[i]
            transition = np.hstack((state, [a, r], state_))
            self.memory[index, :] = transition

    def Reward(self, trg, hyp, show = False):
        """
        Return an array of rewards, based on the current Score.
        From a T predicted sequence. Gives a reward per each T steps.
        Just when the predicted word is on the right place.

        :param trg: target.
        :param hyp: the predicted sequence.
        """

        tar_len = trg.shape[1]
        hyp_len = hyp.shape[1]

        final_rew = -1*np.ones(hyp_len)

        len_temp = 0
        if  tar_len > hyp_len:
            len_temp = hyp_len
        else:
            len_temp = tar_len
        hyp2com = np.zeros([1,tar_len])
        hyp2com[0 ,:len_temp] = hyp[0 ,:len_temp]

        equal = (trg.numpy() == hyp2com)

        ind1, ind2 = np.where(equal == False)

        if len(ind1) != 0:
            equal[ind1[0]:, ind2[0]:] = False

        decoded_valid_tar = self.model.trg_vocab.arrays_to_sentences(arrays=trg ,
                                                cut_at_eos=True)
        decoded_valid_hyp = self.model.trg_vocab.arrays_to_sentences(arrays=hyp ,
                                                cut_at_eos=True)

        if show:
            print('la lista trg-out decodificada: ', decoded_valid_tar)
            print('la lista hypotesis decodificada: ', decoded_valid_hyp)

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

        k = sum(np.arange(tar_len+1))
        a_i = np.arange(1,tar_len+1)/k
        VSa_i = [sum(a_i[:i]) for i in  np.arange(1,tar_len+1, dtype='int')]
        VSa_i = np.multiply(np.asanyarray(VSa_i)
                .reshape([1, tar_len]), equal).reshape([tar_len])

        final_rew[: len_temp] = np.multiply(VSa_i
        , current_valid_score)[: len_temp]

        return final_rew

    def dev_network(self):
        for data_set_name, data_set in self.data_to_dev.items():
            valid_iter = make_data_iter(
                dataset=data_set, batch_size=1, batch_type=self.batch_type,
                shuffle=False, train=False)
            valid_sources_raw = data_set.src

            self.eval_net.eval()
            # don't track gradients during validation
            with torch.no_grad():
                r_total = 0

                for valid_batch in iter(valid_iter):
                    # run as during training to get validation loss (e.g. xent)

                    batch = Batch(valid_batch, self.pad_index, use_cuda=self.use_cuda)

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
                    attention_scores = []
                    #hidden = None
                    hidden = self.model.decoder._init_hidden(encoder_hidden)
                    prev_att_vector = None
                    finished = batch.src_mask.new_zeros((batch_size, 1)).byte()

                    # pylint: disable=unused-variable
                    for t in range(self.max_output_length):
                        state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu()[0]
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
                        logits = self.eval_net(state)
                        logits = logits.reshape([1,1,-1]) 
                        #print(type(logits), logits.shape, logits)
                        next_word = torch.argmax(logits, dim=-1)                        
                        a = next_word.squeeze(1).detach().cpu().numpy()[0]
                        prev_y = next_word
                        
                        output.append(next_word.squeeze(1).detach().cpu().numpy())
                        prev_y = next_word
                        
                        # check if previous symbol was <eos>
                        is_eos = torch.eq(next_word, self.eos_index)
                        finished += is_eos
                        # stop predicting if <eos> reached for all elements in batch
                        if (finished >= 1).sum() == batch_size:
                            break
                    stacked_output = np.stack(output, axis=1)  # batch, time

                    #decode back to symbols
                    decoded_valid_in = self.model.trg_vocab.arrays_to_sentences(arrays=batch.src,
                                                cut_at_eos=True)
                    decoded_valid_out_trg = self.model.trg_vocab.arrays_to_sentences(arrays=batch.trg,
                                                                        cut_at_eos=True)
                    decoded_valid_out = self.model.trg_vocab.arrays_to_sentences(arrays=stacked_output,
                                                cut_at_eos=True)
                    #Final reward?      
                    hyp = stacked_output
                    r = self.Reward(batch.trg, hyp , show = False)
                    r_total += sum(r[np.where(r > 0)])
                    print('Reward: ', r)
                    print('r_total: ', r_total )

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
    cfg_file = "./configs/reverse.yaml" 
    MDQN = QManager(cfg_file, "./reverse_model/best.ckpt")
    MDQN.Collecting_experiences()
