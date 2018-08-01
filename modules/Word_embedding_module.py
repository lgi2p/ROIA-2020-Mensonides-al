# -*- coding: utf-8 -*-


import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import io
import numpy as np


class Word_embedding_module(nn.Module):
    def __init__(self,
                 vocabulary_handler,
                 ):
        super(Word_embedding_module, self).__init__()

        self.vocabulary_handler = vocabulary_handler

        ###############
        # embedding layers
        ###############

        self.word_embeddings = nn.Embedding(
            num_embeddings=len(self.vocabulary_handler.vocabulary),
            embedding_dim=self.vocabulary_handler.embedding_dimension
        )

        self.init_weights()

    def init_weights(self):

        # initialize word embedding manually
        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.vocabulary_handler.embedding_matrix))

    def forward(self, x, x_lengths, x_max_length, batch_size):
        # x: input, should be of shape [batch_size, x_max_length]
        # x_lengths: int array
        # x_max_length: scalar
        # batch_size: scalar

        x_embedded = self.word_embeddings(x)

        return x_embedded

    def do_word_embedding(self, x):
        # return the embedding of x
        return self.word_embeddings(x)  # [batch_size, max_length, embedding_dim]

