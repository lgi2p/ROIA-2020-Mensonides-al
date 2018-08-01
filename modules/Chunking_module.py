# -*- coding: utf-8 -*-

from aaev2.modules.Abstract_module import Abstract_module

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Chunking_module(Abstract_module):

    def __init__(self,
                 pos_module,
                 pos_label_embedding_handler,
                 rnn_hidden_dim,
                 nb_classes,
                 dropout_rate):

        Abstract_module.__init__(self)

        self.pos_module = pos_module
        self.pos_label_embedding_handler = pos_label_embedding_handler
        self.rnn_hidden_dim = rnn_hidden_dim
        self.nb_classes = nb_classes
        self.dropout_rate = dropout_rate

        # to compute the pos weighted label embedding
        self.pos_weighted_label_embedding = nn.Linear(
            in_features=self.pos_module.nb_classes,
            out_features=pos_label_embedding_handler.embedding_dimension,
            bias=False
        )

        rnn_input_size = self.pos_module.word_embedding_module.word_embeddings.embedding_dim \
                         + 2*self.pos_module.rnn_hidden_dim \
                         + self.pos_label_embedding_handler.embedding_dimension

        self.rnn = nn.LSTM(
            input_size= rnn_input_size,
            hidden_size= self.rnn_hidden_dim,
            bias=  True,
            num_layers= 1,
            batch_first= True,
            bidirectional= True
        )

        self.fc = nn.Linear(
            in_features= 2*self.rnn_hidden_dim,
            out_features=2*self.rnn_hidden_dim
        )

        self.logits = nn.Linear(
            in_features= 2*self.rnn_hidden_dim,
            out_features= self.nb_classes
        )

        self._init_weights()

    def _init_weights(self):
        # initialize the weights of the model here

        for param_name, param_value in self.rnn.named_parameters():
            if 'weight' in param_name:
                nn.init.orthogonal_(param_value)
            elif 'bias' in param_name:
                nn.init.constant_(param_value, 0.0)
            else:
                raise ('Error on param type, not a weight nor a bias')

        for param_name, param_value in self.fc.named_parameters():
            if 'weight' in param_name:
                nn.init.kaiming_normal_(param_value)
            elif 'bias' in param_name:
                nn.init.constant_(param_value, 0.0)
            else:
                raise ('Error on param type, not a weight nor a bias')


        for param_name, param_value in self.logits.named_parameters():
            if 'weight' in param_name:
                nn.init.kaiming_normal_(param_value)
            elif 'bias' in param_name:
                nn.init.constant_(param_value, 0.0)
            else:
                raise ('Error on param type, not a weight nor a bias')

        # initialize pos label embedding manually
        self.pos_weighted_label_embedding.weight.data.copy_(torch.from_numpy(np.transpose(self.pos_label_embedding_handler.embedding_matrix)))

    def get_last_layer_outputs(self, x, x_lengths, x_max_length, batch_size):
        llo = self.pos_module.forward(x, x_lengths, x_max_length, batch_size)

        return {
            'x_embedded': llo['x_embedded'],
            'pos_rnn_hidden_states': llo['rnn_hidden_states'],
            'pos_logits': llo['logits']
        }

    def _do_weighted_pos_label_embedding(self, pos_flat_logits, x_max_length, batch_size):

        pos_prob_flat = torch.nn.functional.softmax(pos_flat_logits, dim=1)
        pos_prob = pos_prob_flat.contiguous().view(batch_size, x_max_length, self.pos_module.nb_classes)
        weighted_pos_label_embedding = self.pos_weighted_label_embedding(pos_prob)

        return weighted_pos_label_embedding

    def forward(self, x, x_lengths, x_max_length, batch_size):
        # x: input, should be of shape [batch_size, x_max_length]
        # x_lengths: int array
        # x_max_length: scalar
        # batch_size: scalar

        llo = self.get_last_layer_outputs( x, x_lengths, x_max_length, batch_size)
        x_embedded = llo['x_embedded']
        pos_rnn_hidden_states = llo['pos_rnn_hidden_states']

        weighted_pos_label_embedding = self._do_weighted_pos_label_embedding(
            pos_flat_logits=llo['pos_logits'],
            x_max_length= x_max_length,
            batch_size = batch_size
        )

        #concat inputs
        inputs = torch.cat((x_embedded, pos_rnn_hidden_states, weighted_pos_label_embedding), 2)
        #dropout
        inputs = torch.nn.functional.dropout(inputs, p=self.dropout_rate, training=self.training)

        # rnn
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths=x_lengths, batch_first=True)
        rnn_hidden_states, _ = self.rnn(packed_input)
        rnn_hidden_states, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_hidden_states, batch_first=True, padding_value=0)
        # flatten hidden_states to be shape [batch_size*x_max_length, 2*rnn_hidden_dim]
        flat_hidden_states= rnn_hidden_states.contiguous().view(batch_size*x_max_length, 2*self.rnn_hidden_dim)

        #fc
        #dropout
        flat_hidden_states= torch.nn.functional.dropout(flat_hidden_states, p=self.dropout_rate, training=self.training)
        fc_states = self.fc(flat_hidden_states)
        fc_states = torch.nn.functional.relu(fc_states)


        #logits
        # dropout
        fc_states = torch.nn.functional.dropout(fc_states, p=self.dropout_rate, training=self.training)
        logits= self.logits(fc_states)

        return {
            'logits': logits,
            'rnn_hidden_states': rnn_hidden_states,
            'x_embedded': x_embedded,
            'pos_rnn_hidden_states': pos_rnn_hidden_states,
            'pos_weighted_label_embedding': weighted_pos_label_embedding
        }


    def compute_w_l2_norm(self, rnn_weight_matrix_reg_coeff=1e-6, classifier_weight_matrix_reg_coeff=1e-5):

        l2_norm = self.pos_module.compute_w_l2_norm(rnn_weight_matrix_reg_coeff, classifier_weight_matrix_reg_coeff)

        # self.rnn L2
        for param_name, param_value in self.rnn.named_parameters():
            if 'weight' in param_name:
                l2_norm = l2_norm + rnn_weight_matrix_reg_coeff * param_value.norm(p=2)**2

        # self.fc L2
        for param_name, param_value in self.fc.named_parameters():
            if 'weight' in param_name:
                l2_norm = l2_norm + classifier_weight_matrix_reg_coeff * param_value.norm(p=2) ** 2

        # self.pos_logits L2
        for param_name, param_value in self.logits.named_parameters():
            if 'weight' in param_name:
                l2_norm = l2_norm + classifier_weight_matrix_reg_coeff * param_value.norm(p=2) ** 2

        return l2_norm


    def compute_successive_reg_term(self, last_epoch_parameters, sucessive_reg_coeff = 1e-2):

        successive_norm = Variable(torch.zeros(1), requires_grad=True).cuda()

        for param_name, param_value in self.pos_module.named_parameters():
            new_norm = param_value - Variable(last_epoch_parameters.get('pos_module.'+param_name), requires_grad = False).cuda()
            new_norm = new_norm.norm(p=2)**2
            successive_norm = successive_norm + sucessive_reg_coeff * new_norm

        return successive_norm






