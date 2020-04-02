

from logging import Logger
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

#from joeynmt.DQN_utils import DQN
from joeynmt.helpers import bpe_postprocess, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint
from joeynmt.data import load_data, make_data_iter
from joeynmt.model import build_model
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.batch import Batch
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy

import random
from torch.utils.tensorboard import SummaryWriter
import sacrebleu

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 80)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization

        self.fc2 = nn.Linear(80, 60)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization

        self.out = nn.Linear(60, N_ACTIONS)
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

        if logger is None:
            logger = make_logger()

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
        self.state_type = cfg["dqn"]["state_type"]
        
        if self.state_type == 'hidden':
            self.state_size = cfg["model"]["encoder"]["hidden_size"]*2
        else:
            self.state_size = cfg["model"]["encoder"]["hidden_size"]

        self.actions_size = len(src_vocab)
        self.gamma = None
        
        print("Sample size: ", self.sample_size )
        print("State size: ", self.state_size)
        print("Action size: ", self.actions_size)
        self.epochs = cfg["dqn"]["epochs"]

        # Inii the Qnet and Qnet2
        self.eval_net = Net(self.state_size, self.actions_size)
        self.target_net = Net(self.state_size, self.actions_size)

        #Following the algorithm
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.size_memory1 = self.state_size * 2 + 2 + 1
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
        #self.data_to_dev = {"dev": dev_data}
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

        #others not important parameters
        self.index_fin = None
        path_tensroboard = self.model_dir + "/tensorboard_DQN/"
        self.tb_writer = SummaryWriter( log_dir=path_tensroboard , purge_step=0)
        self.dev_network_count = 0
        print(cfg["dqn"]["reward_type"])
        #Reward funtion related:
        if cfg["dqn"]["reward_type"] == "bleu_diff" : 
            print("You select the reward based on the Bleu score differences")
            self.Reward = self.Reward_bleu_diff
        elif cfg["dqn"]["reward_type"] == "bleu_lin" : 
            print("You select the reward based on the linear Bleu socres, and several punishments")
            self.Reward = self.Reward_lin
        else:
            print("You select the reward based on the final score on the last state ")
            self.Reward = self.Reward_bleu_fin


    def Collecting_experiences(self)-> None:
        """
        Main funtion. Compute all the process.

        :param exp_list: List of experineces. Tuples (memory_counter, state, a, state_, is_eos[0,0])
        :param rew: rewards for every experince. Of lenght of the hypotesis
        """
        for epoch_no in range(self.epochs):
            print("EPOCH %d", epoch_no + 1)
            
            #beam_dqn = self.beam_min + int(self.beam_max * epoch_no/self.epochs)
            #egreed = self.egreed_max*(1 - epoch_no/(1.1*self.epochs))
            #self.gamma = self.gamma_max*(1 - epoch_no/(2*self.epochs))

            beam_dqn = 1
            egreed = 0.5
            #self.gamma = self.gamma_max
            self.gamma = 0.6

            self.tb_writer.add_scalar("parameters/beam_dqn",
                                              beam_dqn, epoch_no)
            self.tb_writer.add_scalar("parameters/egreed",
                                              egreed, epoch_no)
            self.tb_writer.add_scalar("parameters/gamma",
                                              self.gamma, epoch_no)
            if beam_dqn > self.actions_size:
                print("The beam_dqn cannot exceed the action size!")
                print("then the beam_dqn = action size")
                beam_dqn = self.actions_size

            print(' beam_dqn, egreed, gamma: ', beam_dqn, egreed, self.gamma)
            for _, data_set in self.data_to_train_dqn.items():
                
                valid_iter = make_data_iter(
                    dataset=data_set, batch_size=1, batch_type=self.batch_type,
                    shuffle=False, train=False)
                #valid_sources_raw = data_set.src
                # disable dropout
                #self.model.eval()

                i_sample = 0
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
                    output = []
                    hidden = self.model.decoder._init_hidden(encoder_hidden)
                    prev_att_vector = None
                    finished = batch.src_mask.new_zeros((batch_size, 1)).byte()

                    # print("Source_raw: ", batch.src)
                    # print("Target_raw: ", batch.trg_input)
                    # print("y0: ", prev_y)
                    
                    
                    
                    exp_list = []
                    # pylint: disable=unused-variable
                    for t in range(self.max_output_length):
                        if t != 0:
                            if self.state_type == 'hidden':
                                state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]
                            else:
                                if t == 0:
                                    state = hidden[0].squeeze(1).detach().cpu().numpy()[0]
                                else:
                                    state = prev_att_vector.squeeze(1).detach().cpu().numpy()[0]
                                
                    
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
                        if t != 0:
                            if self.state_type == 'hidden':
                                state_ = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0]
                            else:
                                state_ = prev_att_vector.squeeze(1).detach().cpu().numpy()[0]
                        
                        # if t == 0:
                        #     print('states0: ', state, state_)

                        # greedy decoding: choose arg max over vocabulary in each step with egreedy porbability
                        
                        if random.uniform(0, 1) < egreed:
                            i_ran = random.randint(0,beam_dqn-1)
                            next_word = torch.argsort(logits, descending=True)[:, :, i_ran]
                        else:
                            next_word = torch.argmax(logits, dim=-1)  # batch x time=1
                        # if t != 0:
                        a = prev_y.squeeze(1).detach().cpu().numpy()[0]
                        #a = next_word.squeeze(1).detach().cpu().numpy()[0]
                        
                        # print("state ",t," : ", state )
                        # print("state_ ",t," : ", state_ )
                        # print("action ",t," : ", a )
                        # print("__________________________________________")

                        output.append(next_word.squeeze(1).detach().cpu().numpy())

                        #tup = (self.memory_counter, state, a, state_)
                        
                    
                        prev_y = next_word
                        # check if previous symbol was <eos>
                        is_eos = torch.eq(next_word, self.eos_index)
                        finished += is_eos
                        if t != 0:
                            self.memory_counter += 1
                            tup = (self.memory_counter, state, a, state_, 1)
                            exp_list.append(tup)
                        
                        #print(t)
                        # stop predicting if <eos> reached for all elements in batch
                        if (finished >= 1).sum() == batch_size:
                            a = next_word.squeeze(1).detach().cpu().numpy()[0]
                            self.memory_counter += 1
                            #tup = (self.memory_counter, state_, a, np.zeros([self.state_size]) , is_eos[0,0])
                            tup = (self.memory_counter, state_, a, np.zeros([self.state_size]), 0)
                            exp_list.append(tup)
                            #print('break')
                            break
                        if t == self.max_output_length-1:
                            #print("reach the max output")
                            a = 0
                            self.memory_counter += 1
                            #tup = (self.memory_counter, state_, a, np.zeros([self.state_size]) , is_eos[0,0])
                            tup = (self.memory_counter, state_, a, -1*np.ones([self.state_size]), 1)
                            exp_list.append(tup)
                            
                        
                            
                    
                    #Collecting rewards
                    hyp = np.stack(output, axis=1)  # batch, time

                    if epoch_no == 0:
                        if i_sample == 0 or i_sample == 3 or i_sample == 6:
                            #print(i_sample)
                            r = self.Reward(batch.trg, hyp, show=True)  # 1 , time-1 
                        else:
                            r = self.Reward(batch.trg, hyp, show=False)  # 1 , time -1 
                    else:
                        #print("aaaa - ",i_sample)
                        r = self.Reward(batch.trg, hyp, show=False)  # 1 , time -1 
                    
                    # if i_sample == 0 or i_sample == 3 or i_sample == 6:
                    #     print("\n Sample Collected: ", i_sample, "-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
                    #     print("Target: ", batch.trg, decoded_valid_out_trg)
                    #     print("Eval  : ", stacked_output, decoded_valid_out)
                    #     print("Reward: ", r, "\n")
                    
                    i_sample += 1
                    self.store_transition(exp_list, r)
                    
                    #Learning.....
                    if self.memory_counter > self.mem_cap - self.max_output_length:
                        self.learn()
                        
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
            self.target_net.load_state_dict(self.eval_net.state_dict())
            #testing the preformace of the network
            if self.learn_step_counter == 0:
                print('As referece this first test on dev data. Is maded with the Q networks, initialized randomly : ' )
            else:
                print("\n Lets copy the Q-value Net in to Q-target net!. And test the performace on the dev data: ")
            
            current_bleu = self.dev_network()
            print("Current Bleu score is: ", current_bleu)
            
        self.learn_step_counter += 1

        
        long_Batch = self.sample_size*3
        # Sampling the higgest rewards values
        b_memory_big = self.memory[np.argsort(-self.memory[:-self.max_output_length, self.state_size+1])][:long_Batch]
        
        sample_index = np.random.choice(long_Batch, self.sample_size)
        b_memory = b_memory_big[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_size+2: self.state_size+2 + self.state_size])

        b_is_eos = torch.FloatTensor(b_memory[:, self.size_memory1-1:]).view(self.sample_size, 1)
        #print(b_a, b_a.size)
        #print(b_is_eos)
        #Activate the eval_net
        unfreeze_model(self.eval_net)
        
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        #taking the most likely action.
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


        
    def store_transition(self, exp_list, rew):
        """
        Fill/ or refill the memory with experiences.

        :param exp_list: List of experineces. Tuples (memory_counter, state, a, state_, is_eos[0,0])
        :param rew: rewards for every experince. Of lenght of the hypotesis        
        """

        if len(exp_list) != len(rew):
            print(' exp_list: ', exp_list)
            print(' rew: ', rew)
            

        assert (len(exp_list) == len(rew) )
        for i, ele in enumerate(exp_list):
            
            index, state, a, state_, is_eos  = ele
            index = index % self.mem_cap
            
            r = rew[i]
            transition = np.hstack((state, [a, r], state_, is_eos))
            #print(i, a, r, is_eos)
            self.memory[index, :] = transition


    def Reward_bleu_diff(self, trg, hyp, show = False):
        """
        To use as self.Reward funtion.
        Return an array of rewards, based on the differences
        of current Blue Score. As proposed on paper.

        :param trg: target.
        :param hyp: the predicted sequence.
        :param show: Boolean, display the computation of the rewards
        :return: current Bleu score
        """


        smooth = 0.001
        rew = np.zeros([len(hyp[0])])
        # print('aa',len(hyp[0]))
        discount_ini_token = 1
        discount_fin_token = 1
        if trg[0,0] != hyp[0,0]:
            #print(trg, hyp)
            discount_ini_token = 0.5
        if len(hyp[0]) > len(trg[0]):
            discount_ini_token = 0.5


        for t in np.arange(len(hyp[0])-1):
            hyp_sub = hyp[:,:t+1]
            #print(hyp_sub)
            decoded_valid_tar = self.model.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
            decoded_valid_hyp = self.model.trg_vocab.arrays_to_sentences(arrays=hyp_sub,
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
            if valid_references:
                assert len(valid_hypotheses) == len(valid_references)

                
            current_valid_score = sacrebleu.corpus_bleu(valid_hypotheses[0]
                , valid_references[0], smooth_method='floor', smooth_value=smooth
                , use_effective_order=True).score
            if t == 0:
                current_valid_score *= discount_ini_token
            if t > len(trg[0]):
                current_valid_score *= discount_fin_token

            if t > self.max_output_length-1:
                current_valid_score = -10
            # if show:
            #     print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            #     print("Target: ", trg, decoded_valid_tar, valid_references)
            #     print("Eval  : ", hyp_sub, decoded_valid_hyp, valid_hypotheses )
            #     print("Current Reward: ", current_valid_score, "\n")
            

            rew[t+1] = current_valid_score
        if show:
            print('rew: ', rew)

        rew[:-1] = np.diff(rew)
        final_rew = rew  
        # final_rew = np.zeros(len(hyp[0]))
        # final_rew[0] = r_1
        # #print(rew)
        # final_rew[1:] = np.diff(rew)
        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target: ", trg, decoded_valid_tar)
            print("Eval  : ", hyp, decoded_valid_hyp)
            print("Reward: ", final_rew, "\n")

        return final_rew

    def Reward_lin(self, trg, hyp, show = False):
        """
        To use as self.Reward funtion. 
        Return an array of rewards, based on the current Score.
        From a T predicted sequence. Gives a reward per each T steps.
        Just when the predicted word is on the right place.

        :param trg: target.
        :param hyp: the predicted sequence.
        :param show: Boolean, display the computation of the rewards
        :return: current Bleu score
        """

        tar_len = trg.shape[1]
        hyp_len = hyp.shape[1]

        final_rew = -1*np.ones(hyp_len-1)

        len_temp = 0
        if  tar_len > hyp_len:
            len_temp = hyp_len
        else:
            len_temp = tar_len
        hyp2com = np.zeros([1,tar_len])
        hyp2com[0 ,:len_temp] = hyp[0 ,:len_temp]

        equal = (trg.numpy() == hyp2com)

        #equal = np.invert(equal)*np.ones(equal.size)*0.2
        # ind1, ind2 = np.where(equal == False)


        # if len(ind1) != 0:
        #     equal[ind1[0]:, ind2[0]:] = False

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

        k = sum(np.arange(tar_len))
        a_i = np.arange(1,tar_len)/k
        VSa_i = [sum(a_i[:i]) for i in  np.arange(1,tar_len, dtype='int')]
        VSa_i = np.multiply(np.asanyarray(VSa_i)
                .reshape([1, tar_len-1]), equal).reshape([tar_len-1])

        final_rew[: len_temp-1] = np.multiply(VSa_i
        , current_valid_score)[: len_temp]
        
        if show:
            print('Reward is: ' , final_rew)
            print('sum: ', sum(final_rew))
        return final_rew

    def Reward_bleu_fin(self, trg, hyp, show = False):
        """
        To use as self.Reward funtion.
        Return an array of rewards, based on the differences
        of current Blue Score. As proposed on paper.

        :param trg: target.
        :param hyp: the predicted sequence.
        :param show: Boolean, display the computation of the rewards
        :return: current Bleu score
        """
        rew = np.zeros(len(hyp[0]))

        decoded_valid_tar = self.model.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        decoded_valid_hyp = self.model.trg_vocab.arrays_to_sentences(arrays=hyp,
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
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)

            current_valid_score = 0
            if self.eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                #print(' aaa ')
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
            
        rew[-1] = current_valid_score
        
        final_rew = rew[1:]
        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target: ", trg, decoded_valid_tar)
            print("Eval  : ", hyp, decoded_valid_hyp)
            print("Reward: ", final_rew, "\n")

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
                dataset=data_set, batch_size=1, batch_type=self.batch_type,
                shuffle=False, train=False)
            valid_sources_raw = data_set.src

            
            # don't track gradients during validation
            r_total = 0
            roptimal_total = 0
            all_outputs = []
            i_sample = 0

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
                hidden = self.model.decoder._init_hidden(encoder_hidden)
                prev_att_vector = None
                finished = batch.src_mask.new_zeros((batch_size, 1)).byte()

                # pylint: disable=unused-variable
                for t in range(self.max_output_length):
                    

                    # if i_sample == 0 or i_sample == 3 or i_sample == 6:
                    #     print("state on t = ", t, " : " , state)

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
                        state = torch.cat(hidden, dim=2).squeeze(1).detach().cpu()[0]
                    else:
                        state = torch.FloatTensor(prev_att_vector.squeeze(1).detach().cpu().numpy()[0])

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
                
                    

                hyp = stacked_output

                r = self.Reward(batch.trg, hyp , show = False)
                
                if i_sample == 0 or i_sample == 3 or i_sample == 6:
                    print("\n Sample ", i_sample, "-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
                    print("Target: ", batch.trg, decoded_valid_out_trg)
                    print("Eval  : ", stacked_output, decoded_valid_out, "\n")
                    print("Reward: ", r)

                #r = self.Reward1(batch.trg, hyp , show = False)
                r_total += sum(r[np.where(r > 0)])
                if i_sample ==0:
                    roptimal = self.Reward(batch.trg, batch.trg , show = False)
                    roptimal_total += sum(roptimal[np.where(roptimal > 0)])
                
                all_outputs.extend(stacked_output)
                i_sample += 1

            assert len(all_outputs) == len(data_set)

            # decode back to symbols
            decoded_valid = self.model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                                cut_at_eos=True)

            # evaluate with metric on full dataset
            join_char = " " if self.level in ["word", "bpe"] else ""
            valid_sources = [join_char.join(s) for s in data_set.src]
            valid_references = [join_char.join(t) for t in data_set.trg]
            valid_hypotheses = [join_char.join(t) for t in decoded_valid]

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
            self.tb_writer.add_scalar("dev/dev_reward",
                                            r_total, self.dev_network_count)
            self.tb_writer.add_scalar("dev/dev_bleu",
                                            current_valid_score, self.dev_network_count)
            
            print(self.dev_network_count ,' r_total and score: ', r_total , current_valid_score)

            
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

