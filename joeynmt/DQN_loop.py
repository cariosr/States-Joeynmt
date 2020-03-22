

import os
import sys
from typing import List, Optional
from logging import Logger
import numpy as np

import torch
from torchtext.data import Dataset, Field
import torch.nn.functional as F

#from joeynmt.DQN_utils import DQN
from joeynmt.helpers import bpe_postprocess, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint
from joeynmt.data import load_data, make_data_iter, MonoDataset
from joeynmt.model import build_model, Model
from joeynmt.vocabulary import Vocabulary
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.batch import Batch
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
from joeynmt.prediction import validate_on_data

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


        self.batch_size = 1 #** #cfg["training"].get(
            #"eval_batch_size", cfg["training"]["batch_size"])
        self.batch_type = cfg["training"].get(
            "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
        self.use_cuda = cfg["training"].get("use_cuda", False)
        self.level = cfg["data"]["level"]
        self.eval_metric = cfg["training"]["eval_metric"]
        self.max_output_length = cfg["training"].get("max_output_length", None)
        

        #Loading the DQN parameters:
        self.sample_size = cfg["dqn"]["sample_size"]		# Sampling size
        self.lr = cfg["dqn"].get("lr", 0.01)                # learning rate on QL
        self.egreed_max = cfg["dqn"].get("egreed_max", 0.9) # greedy policy, max values
        self.egreed_min = cfg["dqn"].get("egreed_min", 0.01)# greedy policy, min value
        self.gamma_max = cfg["dqn"].get("gamma_max", 0.9)   # reward discount, max value
        self.gamma_min = cfg["dqn"].get("gamma_min", 0.5)   # reward discount, min value
        self.nu_iter = cfg["dqn"]["nu_iter"] 				# target update frequency, on learn.
        self.mem_cap = cfg["dqn"]["mem_cap"]    		    # Number of buffer experience
        self.beam_min = cfg["dqn"]["beam_min"] 				# Beam minimun value
        self.beam_max = cfg["dqn"]["beam_max"] 	            # Beam maximun value

        # load the data
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])
        
        #others parameters
        self.bos_index = trg_vocab.stoi[BOS_TOKEN]
        self.eos_index = trg_vocab.stoi[EOS_TOKEN]
        self.pad_index = trg_vocab.stoi[PAD_TOKEN]

        self.data_to_predict = {"test": test_data}
        #self.data_to_predict = {"dev": dev_data, "test": test_data}
        #self.data_to_predict = {"train": train_data, "dev": dev_data, "test": test_data}

        # load model state from disk
        print(ckpt)
        model_checkpoint = load_checkpoint(ckpt, use_cuda=self.use_cuda)

        # build model and load parameters into it
        self.model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
        self.model.load_state_dict(model_checkpoint["model_state"])

        if self.use_cuda:
            self.model.cuda()

        # whether to use beam search for decoding, 0: greedy decoding
        beam_size = 1
        beam_alpha = -1

        print(cfg["model"]["encoder"]["hidden_size"])
        print(self.gamma_max)

    def Collecting_experinces(self)-> None:
        beam_qdn = self.beam_min  # a modificarrr

        for data_set_name, data_set in self.data_to_predict.items():
            valid_iter = make_data_iter(
                dataset=data_set, batch_size=1, batch_type=self.batch_type,
                shuffle=False, train=False)
            valid_sources_raw = data_set.src
            #pad_index = self.model.src_vocab.stoi[PAD_TOKEN]
            #pad_index = self.model.pad_index
            # disable dropout

            self.model.eval()
            # don't track gradients during validation
            with torch.no_grad():
                all_outputs = []
                valid_attention_scores = []
                total_loss = 0
                total_ntokens = 0
                total_nseqs = 0
        
        
                for valid_batch in iter(valid_iter):
                    # run as during training to get validation loss (e.g. xent)

                    batch = Batch(valid_batch, self.pad_index, use_cuda=self.use_cuda)
                    # sort batch now by src length and keep track of order
                    #sort_reverse_index = batch.sort_by_src_lengths()
                    print('El input es:  ', batch.src)
                    print('El trg es:  ', batch.trg)
                    
                    
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
                    #hidden = None
                    hidden = self.model.decoder._init_hidden(encoder_hidden)
                    prev_att_vector = None
                    finished = batch.src_mask.new_zeros((batch_size, 1)).byte()

                    r = 0
                    r_list = []
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
                        state_ = torch.cat(hidden, dim=2).squeeze(1).detach().cpu().numpy()
                        
                        # greedy decoding: choose arg max over vocabulary in each step
                        # change it.... 
                        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
                        #print('La predicion ' ,t,' es:  ' , next_word)
                        a = next_word.squeeze(1).detach().cpu().numpy()[0]
                        # print('La acion_ ' ,t,' es:  ' , a)
                        
                        output.append(next_word.squeeze(1).detach().cpu().numpy())
                        hyp_t =  np.stack(output, axis=1)   # batch, time, hypotesis so far
                        len_t = hyp_t.shape[1]              # length so far
                        trg_t = batch.trg[0,:len_t].detach().cpu().numpy()   #target so far
                        
                        # print(len_t)
                        # print(trg_t)
                        # print(hyp_t, batch.trg.shape, batch.trg)
                        
                        # print('reward = ', sum(sum(hyp_t == trg_t)))
                        #The reward is computed of how many coincidences are betewn the 
                        #current target and current hypotesis. Basic reward!
                        
                        flag_ = False
                        for ele in r_list:
                            if ele == r:
                                flag_ = True


                        if t < batch.trg.shape[1] :
                            r = sum(sum(hyp_t == trg_t))                            
                        elif  flag_:
                            r = 0

                        r_list.append(r)

                        print(t,'_ Colecting s,a,r,s_ : ', state.shape, a, r, state_.shape )

                        prev_y = next_word
                        attention_scores.append(att_probs.squeeze(1).detach().cpu().numpy())
                        # batch, max_src_lengths

                        # check if previous symbol was <eos>
                        is_eos = torch.eq(next_word, self.eos_index)
                        finished += is_eos
                        # stop predicting if <eos> reached for all elements in batch
                        if (finished >= 1).sum() == batch_size:
                            break

                    stacked_output = np.stack(output, axis=1)  # batch, time
                    stacked_attention_scores = np.stack(attention_scores, axis=1) 

                    print('src shape: ', batch.src.shape)
                    print('trg shape: ', batch.trg.shape)
                    print('hyp shape: ', stacked_output.shape)
                    #decode back to symbols
                    decoded_valid_in = self.model.trg_vocab.arrays_to_sentences(arrays=batch.src,
                                                cut_at_eos=True)
                    decoded_valid_out_trg = self.model.trg_vocab.arrays_to_sentences(arrays=batch.trg,
                                                                        cut_at_eos=True)
                    decoded_valid_out = self.model.trg_vocab.arrays_to_sentences(arrays=stacked_output,
                                                cut_at_eos=True)
                    

                    print('la lista in decodificada: ', decoded_valid_in)
                    print('la lista trg-out decodificada: ', decoded_valid_out_trg)
                    print('la lista out decodificada: ', decoded_valid_out)

                 
                    #Final reward?      
                    hyp = stacked_output
                    print('Reward: ', self.Reward(batch.trg, hyp ))




    def Reward(self, trg, hyp):
        """
        Return an array of rewards, based on the current Score.
        From a T predicted sequence. Gives a reward per each T steps.
        Just when the predicted word is on the right place.

        :param trg: target.
        :param hyp: the predicted sequence.
        """
        
        tar_len = trg.shape[1]
        hyp_len = hyp.shape[1]
        
        len_temp = 0
        if  tar_len > hyp_len:
            len_temp = hyp_len
        else:
            len_temp = tar_len
        hyp2com = np.zeros([1,tar_len])
        hyp2com[0,:len_temp] = hyp[0,:len_temp]
        
        equal = (trg.numpy() == hyp2com)
        #print(equal, '\n')
        
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
        VSa_i = [sum(a_i[:i]) for i in  np.arange(1,tar_len+1, dtype = 'int' )]
        VSa_i = np.multiply(np.asanyarray(VSa_i).reshape([1,tar_len]), equal).reshape([tar_len])
        return np.multiply(VSa_i,current_valid_score)


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
    MDQN.Collecting_experinces()

