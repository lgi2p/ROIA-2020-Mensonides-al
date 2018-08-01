# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Abstract_module(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def _init_weights(self):
        # initialize the weights of the model here
        raise NotImplementedError


    def forward(self,  x, x_lengths, x_max_length, batch_size):
        # x: input, should be of shape [batch_size, x_max_length]
        # x_lengths: int array
        # x_max_length: scalar
        # batch_size: scalar
        raise NotImplementedError

    def get_last_layer_outputs(self, x, x_lengths, x_max_length, batch_size):
        # concat input from previous layers
        raise NotImplementedError

    def compute_w_l2_norm(self, rnn_weight_matrix_reg_coeff=1e-6, classifier_weight_matrix_reg_coeff=1e-5):
        raise NotImplementedError

    def compute_successive_reg_term(self, last_epoch_parameters, sucessive_reg_coeff = 1e-2):
        raise NotImplementedError

