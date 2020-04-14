from logging import Logger
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

# from joeynmt.DQN_utils import DQN
from joeynmt.helpers import bpe_postprocess, load_config, make_logger, \
    get_latest_checkpoint, load_checkpoint
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


class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 60)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization

        #self.fc2 = nn.Linear(80, 60)
        #self.fc2.weight.data.normal_(0, 0.1)  # initialization

        self.out = nn.Linear(60, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        #x = self.fc2(x)
        #x = F.relu(x)

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

        # print(cfg.keys())
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
                step = ckpt.split(model_dir + "/")[1].split(".ckpt")[0]
            except IndexError:
                step = "best"

        self.batch_size = 1  # **
        self.batch_type = cfg["training"].get(
            "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
        self.use_cuda = cfg["training"].get("use_cuda", False)
        self.level = cfg["data"]["level"]
        self.eval_metric = cfg["training"]["eval_metric"]
        self.max_output_length = cfg["training"].get("max_output_length", None)

        # load the data
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])
        # Loading the DQN parameters:
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

        if self.state_type == 'hidden':
            self.state_size = cfg["model"]["encoder"]["hidden_size"] * 2
        else:
            self.state_size = cfg["model"]["encoder"]["hidden_size"]

        self.actions_size = len(src_vocab)
        self.gamma = None

        print("Sample size: ", self.sample_size)
        print("State size: ", self.state_size)
        print("Action size: ", self.actions_size)
        self.num_episodes = cfg["dqn"]["num_episodes"]

        # Inii the Qnet and Qnet2
        self.eval_net = Net(self.state_size, self.actions_size)
        self.target_net = Net(self.state_size, self.actions_size)

        # Following the algorithm
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.size_memory_dim = self.state_size * 2 + 2 + 1
        self.memory = np.zeros((self.mem_cap, self.size_memory_dim))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters()
                                          , lr=self.lr)
        self.loss_func = nn.MSELoss()

        # others parameters
        self.bos_index = trg_vocab.stoi[BOS_TOKEN]
        self.eos_index = trg_vocab.stoi[EOS_TOKEN]
        self.pad_index = trg_vocab.stoi[PAD_TOKEN]

        self.data_to_train_dqn = {"train": train_data}

        # self.data_to_train_dqn = {"test": test_data}
        # self.data_to_dev = {"dev": dev_data}
        #self.data_to_dev = {"dev": dev_data}
        # self.data_to_train_dqn = {"train": train_data
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

        relevant_hyp = "nu_pretrain=" + str(self.nu_pretrain) + "_" + "reward_type=" + str(self.reward_type)

        # others not important parameters
        self.index_fin = None
        # construct the name of the folder for tensorboard for the test given the date and the relevant_hyp
        path_tensroboard = self.model_dir + "/tensorboard_DQN/" + date + "/" + relevant_hyp + "/"
        self.tb_writer = SummaryWriter(log_dir=path_tensroboard, purge_step=0)
        print(self.reward_type)
        # Reward funtion related:
        if self.reward_type == "bleu_diff":
            print("You select the reward based on the Bleu score differences")
            self.Reward = self.Reward_bleu_diff
        elif self.reward_type == "correct_action":
            print("You select the reward based on the correct action ")
            self.Reward = self.Reward_correct_action

    def train_DQN(self) -> None:
        """
        Function to train the DQN.

        """
        beam_dqn = 3
        egreed = 0.2
        self.gamma = 0.9
        egreed_inc = 2 * (1 - egreed) / self.num_episodes
        correct_actions_episode = 0

        for episode_i in range(self.num_episodes):
            print("EPISODE: ", episode_i + 1)

            # self.tb_writer.add_scalar("parameters/beam_dqn",
            #                           beam_dqn, episode_i)
            self.tb_writer.add_scalar("parameters/egreed",
                                       egreed, episode_i)
            self.tb_writer.add_scalar("parameters/correct_actions_episode",
                                      correct_actions_episode, episode_i)
            # self.tb_writer.add_scalar("parameters/gamma",
            #                           self.gamma, episode_i)
            if beam_dqn > self.actions_size:
                print("The beam_dqn cannot exceed the action size!")
                print("Then the beam_dqn will be set to action size")
                beam_dqn = self.actions_size

            print(' beam_dqn, egreed, gamma: ', beam_dqn, egreed, self.gamma)

            for _, data_set in self.data_to_train_dqn.items():

                valid_iter = make_data_iter(
                    dataset=data_set, batch_size=1, batch_type=self.batch_type,
                    shuffle=False, train=False)

                i_sample = 0
                for valid_batch in iter(valid_iter):

                    batch = Batch(valid_batch
                                  , self.pad_index, use_cuda=self.use_cuda)

                    encoder_output, encoder_hidden = self.model.encode(
                        batch.src, batch.src_lengths,
                        batch.src_mask)

                    # In case of the maximum output length is not globally specified,
                    # adapt it to the double of src len
                    if self.max_output_length is None:
                        self.max_output_length = 2 * np.max(batch.src_lengths.cpu().numpy())

                    batch_size = batch.src_mask.size(0)
                    prev_y = batch.src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                                     dtype=torch.long)
                    output = []
                    hidden = self.model.decoder._init_hidden(encoder_hidden)
                    prev_att_vector = None


                    # print("Source_raw: ", batch.src)
                    # print("Target_raw: ", batch.trg_input)

                    # get the raw vector in order to use it
                    # later in the learning for the true actions
                    trg_input = batch.trg.cpu().detach().numpy().squeeze()

                    # print ("trg numpy: ", trg_input_np)
                    # print("y0: ", prev_y)

                    finished = batch.src_mask.new_zeros((batch_size, 1)).byte()

                    exploration_counter = 0
                    exploitation_counter = 0
                    for t in range(self.max_output_length):
                        if t > 0:
                            if self.state_type == 'hidden':
                                state = torch.FloatTensor(torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0])
                            else:
                                state = torch.FloatTensor(prev_att_vector.squeeze(1).detach().cpu().numpy()[0])
                        else:
                            state = torch.zeros(self.state_size)

                        # decode one single step
                        logits, hidden, att_probs, prev_att_vector = self.model.decoder(
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask,
                            trg_embed=self.model.trg_embed(prev_y),
                            hidden=hidden,
                            prev_att_vector=prev_att_vector,
                            unroll_steps=1)

                        # get the next state
                        if self.state_type == 'hidden':
                            state_ = torch.FloatTensor(torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()[0])
                        else:
                            state_ = torch.FloatTensor(prev_att_vector.squeeze(1).detach().cpu().numpy()[0])

                        # use the e-greedy policy
                        exp_flag = False
                        if random.uniform(0, 1) < egreed: # exploitation
                            logits = self.eval_net(state)
                            logits = logits.reshape([1, 1, -1])
                            next_word = torch.argmax(logits, dim=-1)
                            exploitation_counter += 1
                            exp_flag = True
                        else: # exploration
                            i_ran = random.randint(0, beam_dqn - 1)
                            next_word = torch.argsort(logits, descending=True)[:, :, i_ran]
                            exploration_counter += 1

                        # get the action the model took based on the next_word
                        a = next_word.squeeze(1).detach().cpu().numpy()[0]

                        # append the next_word to the final output
                        output.append(next_word.squeeze(1).detach().cpu().numpy())

                        # save next_word in prev_y to use it when decoding
                        prev_y = next_word


                        # get the reward if the action taken was correct
                        if t >= len(trg_input):
                            r = 0
                        else:
                            if a == trg_input[t]:
                                if exp_flag:
                                    correct_actions_episode += 1
                                if a == self.eos_index:
                                    r = 0.2
                                else:
                                    r = 1
                            else:
                                r = 0

                        # check if previous symbol was <eos>
                        is_eos = torch.eq(next_word, self.eos_index)

                        if is_eos:
                            tup = (self.memory_counter, state, a, r, state_, 1)
                        else:
                            tup = (self.memory_counter, state, a, r, state_, 0)


                        self.memory_counter += 1

                        # store transition
                        self.store_transition(tup)

                        # Learning.....
                        if self.memory_counter >= self.mem_cap:
                            if self.memory_counter == self.mem_cap:
                                print ("Starting learning...")
                            self.learn()

                        finished += is_eos
                        # stop predicting if <eos> reached for all elements in batch
                        if (finished >= 1).sum() == batch_size:
                            break

                    # compare the prediction and the true output
                    # stack the output in the hyp
                    hyp = np.stack(output, axis=1)
                    if i_sample < 10:
                        print ("Sample {} :".format(i_sample))
                        print ("Hyp: ", hyp)
                        print ("Target: ", trg_input)
                        print ("Exploration actions: ", exploration_counter)
                        print ("Exploitation actions: ", exploitation_counter)
                        print ("")

                    i_sample += 1
                if egreed + egreed_inc < 0.98:
                    egreed += egreed_inc

            print ("Correct actions in episode {}: {}".format(episode_i + 1, correct_actions_episode))

        self.tb_writer.close()

    def learn(self):
        """
            Learning process...
        """

        # target parameter update
        if self.learn_step_counter % self.nu_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        long_Batch = self.sample_size * 3
        # Sampling the higgest rewards values
        b_memory_big = self.memory[np.argsort(-self.memory[:-self.max_output_length, self.state_size + 1])][:long_Batch]

        sample_index = np.random.choice(long_Batch, self.sample_size)
        b_memory = b_memory_big[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size + 1:self.state_size + 2])
        b_s_ = torch.FloatTensor(b_memory[:, self.state_size + 2: self.state_size + 2 + self.state_size])

        b_is_eos = torch.FloatTensor(b_memory[:, self.size_memory_dim - 1:]).view(self.sample_size, 1)

        # print(b_a, b_a.size)
        # print(b_is_eos)

        # Activate the eval_net
        unfreeze_model(self.eval_net)

        # q_eval w.r.t the action in experience

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate

        if self.learn_step_counter % 50 == 1:
            print ("learn step counter: ", self.learn_step_counter)

        # take the most likely action
        b_a_ = torch.LongTensor(q_next.max(1)[1].view(self.sample_size, 1).long())

        q_eval_next = self.eval_net(b_s_).gather(1, b_a_)  # shape (batch, 1)

        # If eos then q_target = reward
        q_target = b_r + self.gamma * b_is_eos * q_eval_next.view(self.sample_size, 1)  # shape (batch, 1)



        loss = self.loss_func(q_eval, q_target)

        self.tb_writer.add_scalar("learn/learn_batch_loss",
                                  loss.data, self.learn_step_counter)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # desctivate the eval_net
        freeze_model(self.eval_net)

    def store_transition(self, tup):
        """
        Store transition based on the sent tuple.
        """
        index, state, a, r, state_, is_eos = tup
        index = index % self.mem_cap

        transition = np.hstack((state, [a, r], state_, is_eos))

        self.memory[index, :] = transition

    def Reward_bleu_diff(self, trg, hyp, show=False):
        """
        To use as self.Reward function.
        Return an array of rewards, based on the differences
        of current Blue Score. As proposed on paper.

        :param trg: target.
        :param hyp: the predicted sequence.
        :param show: Boolean, display the computation of the rewards
        :return: current Bleu score
        """

        smooth = 0.001
        rew = np.zeros([len(hyp[0])])
        # print('len(hyp[0]) = ',len(hyp[0]))
        # discount_ini_token = 1
        # discount_fin_token = 1
        # if trg[0,0] != hyp[0,0]:
        # print(trg, hyp)
        # discount_ini_token = 0.5
        # if len(hyp[0]) > len(trg[0]):
        # discount_fin_token = 0.5

        for t in np.arange(len(hyp[0]) - 1):
            hyp_sub = hyp[:, :t + 1]
            # print(hyp_sub)
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
                                                        , valid_references[0], smooth_method='floor',
                                                        smooth_value=smooth
                                                        , use_effective_order=True).score
            rew[t + 1] = current_valid_score
        if show:
            print('rew: ', rew)

        rew[:-1] = np.diff(rew)
        final_rew = rew

        # if len(hyp[0]) == self.max_output_length:
        #    final_rew[-1] = -10

        # final_rew[-1] = final_rew[-1] / len(hyp[0])

        # final_rew = np.zeros(len(hyp[0]))
        # final_rew[0] = r_1
        # #print(rew)
        # final_rew[1:] = np.diff(rew)
        if show:
            print("\n Sample-------------Target vs Eval_net prediction:--Raw---and---Decoded-----")
            print("Target: ", trg, decoded_valid_tar)
            print("Eval  : ", hyp, decoded_valid_hyp)
            print("Reward: ", final_rew, "\n")

        # print ("shape final_rew = ", final_rew.shape)
        return final_rew

    def Reward_correct_action(self, trg, hyp, show=False):
        """
        To use as self.Reward funtion.
        Return an array of rewards, based on the differences
        of current Blue Score. As proposed on paper.

        :param trg: target.
        :param hyp: the predicted sequence.
        :param show: Boolean, display the computation of the rewards
        :return: current Bleu score
        """
        len_hyp = len(hyp[0])
        final_rew = np.arange(1, len_hyp + 1)
        # final_rew = np.ones(len_hyp)
        return final_rew

def dqn(cfg_file, ckpt: str, output_path: str = None) -> None:
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
    # cfg_file = "./reverse_model/config.yaml"
    # cfg_file = "./configs/reverse.yaml"
    # MDQN = QManager(cfg_file, "./reverse_model/best.ckpt")
    MDQN = QManager(cfg_file, ckpt)
    MDQN.train_DQN()

