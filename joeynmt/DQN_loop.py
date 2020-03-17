

import os
import sys
from typing import List, Optional
from logging import Logger
import numpy as np

import torch
from torchtext.data import Dataset, Field
import torch.nn.functional as F

from DQN_utils import DQN
from joeynmt.helpers import bpe_postprocess, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint
from joeynmt.data import load_data, make_data_iter, MonoDataset
from joeynmt.model import build_model, Model

class Main_Loop_DQN:
    """ Manages the Collecting and learning of the DQN, 
    extract the states from the sequence. 
    Say what is the reward and the next sequence and next state ... etc"""

    def __init__(self, cfg_file,
         ckpt: str,
         output_path: str = None,
         logger: Logger = None, MEMORY_CAPACITY = 2000) -> None:

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


        batch_size = 1 #** #cfg["training"].get(
            #"eval_batch_size", cfg["training"]["batch_size"])
        self.batch_type = cfg["training"].get(
            "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
        self.use_cuda = cfg["training"].get("use_cuda", False)
        self.level = cfg["data"]["level"]
        self.eval_metric = cfg["training"]["eval_metric"]
        self.max_output_length = cfg["training"].get("max_output_length", None)

        # load the data
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"])

        self.data_to_predict = {"dev": dev_data, "test": test_data}

        # load model state from disk
        model_checkpoint = load_checkpoint(ckpt, use_cuda=self.use_cuda)

        # build model and load parameters into it
        self.model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
        #model.load_state_dict(model_checkpoint["model_state"])

        if self.use_cuda:
            self.model.cuda()

        # whether to use beam search for decoding, 0: greedy decoding
        if "testing" in cfg.keys():
            beam_size = cfg["testing"].get("beam_size", 1)
            beam_alpha = cfg["testing"].get("alpha", -1)
        else:
            beam_size = 1
            beam_alpha = -1

        print(cfg["encoder"]["hidden_size"])

    def forward(self) \
        -> None:
        print('acaa')
        # dqn = DQN()
        # for data_set_name, data_set in self.data_to_predict.items():



cfg_file = "./reverse_model/config.yaml"
MDQN = Main_Loop_DQN(cfg_file, "reverse_model")
