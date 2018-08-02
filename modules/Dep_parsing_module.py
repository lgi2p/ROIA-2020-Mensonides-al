# -*- coding: utf-8 -*-

from modules.Abstract_module import Abstract_module

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Dep_parsing_module(Abstract_module):

    def __init__(self,
                 chunking_module,
                 chunking_label_embedding_handler,
                 rnn_hidden_dim,
                 nb_classes,
                 dropout_rate):

        Abstract_module.__init__(self)

        self.chunking_module = chunking_module
        self.chunking_label_embedding_handler = chunking_label_embedding_handler
        self.rnn_hidden_dim = rnn_hidden_dim
        self.nb_classes = nb_classes
        self.dropout_rate = dropout_rate

        # to compute the chunking weighted label embedding
        self.chunking_weighted_label_embedding = nn.Linear(
            in_features=self.chunking_module.nb_classes,
            out_features=chunking_label_embedding_handler.embedding_dimension,
            bias=False
        )

        rnn_input_size = self.chunking_module.pos_module.word_embedding_module.word_embeddings.embedding_dim \
                         + 2*self.chunking_module.pos_module.rnn_hidden_dim \
                         + self.chunking_module.pos_label_embedding_handler.embedding_dimension\
                         + 2*self.chunking_module.rnn_hidden_dim\
                         + self.chunking_label_embedding_handler.embedding_dimension

        self.rnn = nn.LSTM(
            input_size= rnn_input_size,
            hidden_size= self.rnn_hidden_dim,
            bias=  True,
            num_layers= 1,
            batch_first= True,
            bidirectional= True
        )

        self.linear_head = nn.Linear(
            in_features= 2*self.rnn_hidden_dim,
            out_features= 2*self.rnn_hidden_dim,
            bias=False
        )

        self.root_param_vector = torch.nn.Parameter(torch.rand(2*self.rnn_hidden_dim), requires_grad=True)

        self.fc_relation_heads = nn.Linear(
            in_features= 4*self.rnn_hidden_dim,
            out_features= self.rnn_hidden_dim
        )

        self.logits_relation_heads = nn.Linear(
            in_features= self.rnn_hidden_dim,
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

        for param_name, param_value in self.fc_relation_heads.named_parameters():
            if 'weight' in param_name:
                nn.init.kaiming_normal_(param_value)
            elif 'bias' in param_name:
                nn.init.constant_(param_value, 0.0)
            else:
                raise ('Error on param type, not a weight nor a bias')


        for param_name, param_value in self.logits_relation_heads.named_parameters():
            if 'weight' in param_name:
                nn.init.kaiming_normal_(param_value)
            elif 'bias' in param_name:
                nn.init.constant_(param_value, 0.0)
            else:
                raise ('Error on param type, not a weight nor a bias')

        for param_name, param_value in self.linear_head.named_parameters():
            if 'weight' in param_name:
                nn.init.kaiming_normal_(param_value)
            elif 'bias' in param_name:
                nn.init.constant_(param_value, 0.0)
            else:
                raise ('Error on param type, not a weight nor a bias')



        # initialize chunking label embedding manually
        self.chunking_weighted_label_embedding.weight.data.copy_(torch.from_numpy(np.transpose(self.chunking_label_embedding_handler.embedding_matrix)))

    def get_last_layer_outputs(self, x, x_lengths, x_max_length, batch_size):
        llo = self.chunking_module.forward(x, x_lengths, x_max_length, batch_size)

        return {
            'x_embedded': llo['x_embedded'],
            'pos_rnn_hidden_states': llo['pos_rnn_hidden_states'],
            'pos_weighted_label_embedding': llo['pos_weighted_label_embedding'],
            'chunking_rnn_hidden_states': llo['rnn_hidden_states'],
            'chunking_logits': llo['logits']
        }

    def _do_weighted_chunking_label_embedding(self, chunking_flat_logits, x_max_length, batch_size):

        chunking_prob_flat = torch.nn.functional.softmax(chunking_flat_logits, dim=1)
        chunking_prob = chunking_prob_flat.contiguous().view(batch_size, x_max_length, self.chunking_module.nb_classes)
        weighted_chunking_label_embedding = self.chunking_weighted_label_embedding(chunking_prob)

        return weighted_chunking_label_embedding

    def forward(self, x, x_lengths, x_max_length, batch_size, true_heads_indices):
        # x: input, should be of shape [batch_size, x_max_length]
        # x_lengths: int array
        # x_max_length: scalar
        # batch_size: scalar

        # true head indices is to compute the relation label between each word and its head index. only used in training and for testing the LAS. Not used in a real world env

        llo = self.get_last_layer_outputs( x, x_lengths, x_max_length, batch_size)
        x_embedded = llo['x_embedded']
        pos_rnn_hidden_states = llo['pos_rnn_hidden_states']
        pos_weighted_label_embedding = llo['pos_weighted_label_embedding']
        chunking_rnn_hidden_states = llo['chunking_rnn_hidden_states']

        chunking_weighted_label_embedding = self._do_weighted_chunking_label_embedding(
            chunking_flat_logits= llo['chunking_logits'],
            x_max_length= x_max_length,
            batch_size = batch_size
        )
        chunking_weighted_label_embedding = torch.nn.functional.dropout(chunking_weighted_label_embedding, p=0.4, training=self.training)
        pos_weighted_label_embedding = torch.nn.functional.dropout(pos_weighted_label_embedding, p=0.4, training=self.training)

        #concat inputs
        inputs = torch.cat((x_embedded, pos_rnn_hidden_states, chunking_rnn_hidden_states), 2)
        inputs = torch.nn.functional.dropout(inputs, p=self.dropout_rate, training=self.training)
        inputs = torch.cat((inputs, pos_weighted_label_embedding, chunking_weighted_label_embedding), 2)

        # rnn
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths=x_lengths, batch_first=True)
        rnn_hidden_states, _ = self.rnn(packed_input)
        rnn_hidden_states, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_hidden_states, batch_first=True, padding_value=0) #[bath_size, max_length, 2 * rnn_hidden_dim]


        ###########
        # Comppute heads from here
        ###########

        # flatten hidden_states to be shape [batch_size*x_max_length, 2*rnn_hidden_dim]
        fhs= rnn_hidden_states.contiguous().view(batch_size*x_max_length, 2*self.rnn_hidden_dim)

        # compute wd_hj
        wd_hj = self.linear_head(fhs)
        wd_hj = wd_hj.contiguous().view(batch_size, x_max_length, 2*self.rnn_hidden_dim)

        #compute root and prepend root
        wd_r = self.linear_head(self.root_param_vector)
        wd_r = wd_r.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 2*self.rnn_hidden_dim)
        wd_hj= torch.cat((wd_r, wd_hj), 1)
        #transpose wd_hj
        wd_hj = torch.transpose(wd_hj, 1, 2)

        #unflat fhs
        fhs = fhs.contiguous().view(batch_size, x_max_length, 2*self.rnn_hidden_dim)

        # for each word, compute score for each potential head
        possible_heads_per_word=[]
        for i in range(x_max_length):
            fhs_i = fhs[:, i, :].unsqueeze(1)
            m_ij = torch.matmul(fhs_i, wd_hj) # corresponds to m_tj for t = i and all j. shape = [batch_size, 1, x_max_lenght]. It computes a score for the current i and all possible heads
            m_ij = torch.squeeze(m_ij, 1)
            possible_heads_per_word.append(m_ij)

        heads_logits = torch.stack(possible_heads_per_word, 1)
        flat_heads_logits = heads_logits.contiguous().view(batch_size * x_max_length, x_max_length+1) # the +1 is because of the root


        ##################
        # compute relation labels from here
        ##################

        # catch the correct h_j
        root_to_prepend = self.root_param_vector.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 2*self.rnn_hidden_dim)
        pool_of_states_to_select_from = torch.cat((root_to_prepend, rnn_hidden_states), 1)
        pool_of_states_to_select_from = pool_of_states_to_select_from.contiguous().view(batch_size*(x_max_length+1), 2*self.rnn_hidden_dim)

        # remove -1 index to index select
        true_heads_indices[true_heads_indices == -1] = x_max_length-1
        # change the index to be the total index after flatenning
        a = torch.arange(0, batch_size * (x_max_length+1), x_max_length+1, dtype=torch.long).cuda()
        a = a.unsqueeze(1).expand(batch_size, x_max_length)
        true_heads_indices = true_heads_indices + a
        true_heads_indices = true_heads_indices.contiguous().view(-1)

        correct_hj = torch.index_select(pool_of_states_to_select_from, 0, true_heads_indices)
        correct_hj = correct_hj.contiguous().view(batch_size, x_max_length, 2*self.rnn_hidden_dim)


        ht_hj = torch.cat((rnn_hidden_states, correct_hj), 2)
        ht_hj = ht_hj.contiguous().view(batch_size * x_max_length, 4*self.rnn_hidden_dim)

        #fc
        #dropout
        ht_hj= torch.nn.functional.dropout(ht_hj, p=self.dropout_rate, training=self.training)
        fc_states = self.fc_relation_heads(ht_hj)
        fc_states = torch.nn.functional.relu(fc_states)


        #logits
        # dropout
        fc_states = torch.nn.functional.dropout(fc_states, p=self.dropout_rate, training=self.training)
        logits= self.logits_relation_heads(fc_states)


        return {
            'flat_heads_logits': flat_heads_logits,
            'flat_relation_logits': logits
        }


    def compute_w_l2_norm(self, rnn_weight_matrix_reg_coeff=1e-6, classifier_weight_matrix_reg_coeff=1e-5):

        l2_norm = self.chunking_module.compute_w_l2_norm(rnn_weight_matrix_reg_coeff, classifier_weight_matrix_reg_coeff)

        # self.rnn L2
        for param_name, param_value in self.rnn.named_parameters():
            if 'weight' in param_name:
                l2_norm = l2_norm + rnn_weight_matrix_reg_coeff * param_value.norm(p=2)**2

        # self.fc L2
        for param_name, param_value in self.fc_relation_heads.named_parameters():
            if 'weight' in param_name:
                l2_norm = l2_norm + classifier_weight_matrix_reg_coeff * param_value.norm(p=2) ** 2

        # self.logits L2
        for param_name, param_value in self.logits_relation_heads.named_parameters():
            if 'weight' in param_name:
                l2_norm = l2_norm + classifier_weight_matrix_reg_coeff * param_value.norm(p=2) ** 2

        # linear heads
        for param_name, param_value in self.linear_head.named_parameters():
            if 'weight' in param_name:
                l2_norm = l2_norm + classifier_weight_matrix_reg_coeff * param_value.norm(p=2) ** 2

        return l2_norm


    def compute_successive_reg_term(self, last_epoch_parameters, sucessive_reg_coeff = 1e-2):

        successive_norm = Variable(torch.zeros(1), requires_grad=True).cuda()

        for param_name, param_value in self.chunking_module.named_parameters():
            new_norm = param_value - Variable(last_epoch_parameters.get('chunking_module.'+param_name), requires_grad = False).cuda()
            new_norm = new_norm.norm(p=2)**2
            successive_norm = successive_norm + sucessive_reg_coeff * new_norm

        return successive_norm






