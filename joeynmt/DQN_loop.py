

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
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.batch import Batch

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
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])

        self.data_to_predict = {"dev": dev_data, "test": test_data}

        # load model state from disk
        print(ckpt)
        model_checkpoint = load_checkpoint(ckpt, use_cuda=self.use_cuda)

        # build model and load parameters into it
        self.model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
        #model.load_state_dict(model_checkpoint["model_state"])

        if self.use_cuda:
            self.model.cuda()

        # whether to use beam search for decoding, 0: greedy decoding
        beam_size = 1
        beam_alpha = -1

        print(cfg["model"]["encoder"]["hidden_size"])
        print(self.gamma_max)

    def Reward_optimo(self)-> None:
        print('acaa')
        
        for data_set_name, data_set in self.data_to_predict.items():
            valid_iter = make_data_iter(
                dataset=data_set, batch_size=self.batch_size, batch_type=self.batch_type,
                shuffle=False, train=False)
            valid_sources_raw = data_set.src
            pad_index = self.model.src_vocab.stoi[PAD_TOKEN]
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

                    batch = Batch(valid_batch, pad_index, use_cuda=self.use_cuda)
                    # sort batch now by src length and keep track of order
                    sort_reverse_index = batch.sort_by_src_lengths()
                    print( self.model.get_state(batch))
            



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
    cfg_file = "./reverse_model/config.yaml"
    MDQN = QManager(cfg_file, "./reverse_model/best.ckpt")
    MDQN.Reward_optimo()

