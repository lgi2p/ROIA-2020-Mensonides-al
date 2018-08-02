# -*- coding: utf-8 -*-

import torch
from data_handlers.VocabularyHandler import VocabularyHandler
from data_handlers.POS_data_handler import POS_data_handler
from data_handlers.Chunking_data_handler import Chunking_data_handler
from data_handlers.Dep_parsing_handler import Dep_parsing_handler

from modules.Word_embedding_module import Word_embedding_module
from modules.POS_module import POS_module
from modules.Chunking_module import Chunking_module
from modules.Dep_parsing_module import Dep_parsing_module

import misc_functions as misc_functions
import train_and_test_functions as train_and_test_functions

import random
import numpy as np
import copy
import os
import time



random.seed(10)
np.random.seed(10)
torch.cuda.manual_seed_all(10)
torch.manual_seed(10)




# #####################
# data handlers
# #####################
vocabulary_handler = VocabularyHandler(
    path_to_word_embedding_file = 'data/glove.6B.300d.txt',
    add_empty_and_unkown_tokens = True
)

# vocabulary_handler = VocabularyHandler(
#     path_to_word_embedding_file = 'data/shorten.glove.txt',
#     add_empty_and_unkown_tokens = True
# )

pos_label_embedding_handler = VocabularyHandler(
    path_to_word_embedding_file= 'data/POS/pos_label_embeddings.txt',
    add_empty_and_unkown_tokens= False
)

chunking_label_embedding_handler = VocabularyHandler(
    path_to_word_embedding_file= 'data/chunking/chunking_label_embeddings.txt',
    add_empty_and_unkown_tokens= False
)

train_POS_data_handler = POS_data_handler(
    file_with_data= 'data/POS/train.txt',
    POS_label_to_id_file= 'data/POS/pos_labels_to_id.txt',
    vocabulary_handler = vocabulary_handler,
    batch_size= 200
)

test_POS_data_handler = POS_data_handler(
    file_with_data= 'data/POS/test.txt',
    POS_label_to_id_file= 'data/POS/pos_labels_to_id.txt',
    vocabulary_handler = vocabulary_handler,
    batch_size= 200
)

train_chunking_data_handler = Chunking_data_handler(
    file_with_data='data/chunking/train.txt',
    chunking_label_to_id_file= 'data/chunking/chunking_labels_to_id.txt',
    vocabulary_handler = vocabulary_handler,
    batch_size= 200
)

test_chunking_data_handler = Chunking_data_handler(
    file_with_data='data/chunking/test.txt',
    chunking_label_to_id_file= 'data/chunking/chunking_labels_to_id.txt',
    vocabulary_handler = vocabulary_handler,
    batch_size= 200
)

train_dep_parsing_data_handler = Dep_parsing_handler(
    list_of_files_with_data= ['data/dep_parsing/UD_English-EWT/en_ewt-ud-train.conllu',
                              'data/dep_parsing/UD_English-GUM/en_gum-ud-train.conllu',
                              'data/dep_parsing/UD_English-LinES/en_lines-ud-train.conllu',
                              'data/dep_parsing/UD_English-ParTUT/en_partut-ud-train.conllu',],
    label_to_id_file= 'data/dep_parsing/dep_parsing_labels_to_id.txt',
    vocabulary_handler = vocabulary_handler,
    batch_size= 200
)

# train_dep_parsing_data_handler = Dep_parsing_handler(
#     list_of_files_with_data= ['data/dep_parsing/UD_English-EWT/en_ewt-ud-dev.conllu'],
#     label_to_id_file= 'data/dep_parsing/dep_parsing_labels_to_id.txt',
#     vocabulary_handler = vocabulary_handler,
#     batch_size= 30
# )

test_dep_parsing_data_handler = Dep_parsing_handler(
    list_of_files_with_data= ['data/dep_parsing/UD_English-EWT/en_ewt-ud-test.conllu',
                              'data/dep_parsing/UD_English-GUM/en_gum-ud-test.conllu',
                              'data/dep_parsing/UD_English-LinES/en_lines-ud-test.conllu',
                              'data/dep_parsing/UD_English-ParTUT/en_partut-ud-test.conllu',],
    label_to_id_file= 'data/dep_parsing/dep_parsing_labels_to_id.txt',
    vocabulary_handler = vocabulary_handler,
    batch_size= 200
)

# test_dep_parsing_data_handler = Dep_parsing_handler(
#     list_of_files_with_data= ['data/dep_parsing/UD_English-EWT/en_ewt-ud-test.conllu'],
#     label_to_id_file= 'data/dep_parsing/dep_parsing_labels_to_id.txt',
#     vocabulary_handler = vocabulary_handler,
#     batch_size= 30
# )

##########################
# Modules
##########################

word_embedding_module = Word_embedding_module(
    vocabulary_handler = vocabulary_handler
).cuda()

pos_module = POS_module(
    word_embedding_module= word_embedding_module,
    rnn_hidden_dim= 100,
    nb_classes= 44,
    dropout_rate= 0.2
).cuda()

chunking_module = Chunking_module(
    pos_module = pos_module,
    pos_label_embedding_handler= pos_label_embedding_handler,
    rnn_hidden_dim= 100,
    nb_classes= 23,
    dropout_rate= 0.2
).cuda()

dep_parsing_module = Dep_parsing_module(
    chunking_module = chunking_module,
    chunking_label_embedding_handler=chunking_label_embedding_handler,
    rnn_hidden_dim= 100,
    nb_classes= 37,
    dropout_rate= 0.2
).cuda()

##########################
# optimizers
##########################

pos_optimizer = torch.optim.Adam(pos_module.parameters(),lr=1e-2)
chunking_optimizer = torch.optim.Adam(chunking_module.parameters(), lr=1e-2)
dep_parsing_optimizer = torch.optim.Adam(dep_parsing_module.parameters(), lr=1e-2)

##########################
# train / test  loop
##########################


start_time = str(int(time.time()))
os.mkdir(os.path.join('./logs', start_time))
print ('start training...')

#lr decaying
pos_scheduler = torch.optim.lr_scheduler.LambdaLR(pos_optimizer, lr_lambda= lambda epoch: 0.75 ** epoch)
chunking_scheduler = torch.optim.lr_scheduler.LambdaLR(chunking_optimizer, lr_lambda= lambda epoch: 0.75 ** epoch)
dep_parsing_scheduler = torch.optim.lr_scheduler.LambdaLR(dep_parsing_optimizer, lr_lambda= lambda epoch: 0.75 ** epoch)
for num_epoch in range(1000):


    if num_epoch % 5 == 0:
        pos_scheduler.step()
        chunking_scheduler.step()
        dep_parsing_scheduler.step()
    for param_group in pos_optimizer.param_groups:
        print ('pos')
        print (param_group['lr'])

    for param_group in chunking_optimizer.param_groups:
        print('chunking')
        print (param_group['lr'])

    ############
    # Pos training
    ############
    pos_train_metrics = train_and_test_functions.train_pos_layer(train_POS_data_handler, pos_optimizer, pos_module, grad_clip_max=1.0)
    pos_test_metrics = train_and_test_functions.test_pos_layer(test_POS_data_handler, pos_module)
    print ('##############################')
    print ('POS: end of epoch ' + str(train_POS_data_handler.num_epoch))
    misc_functions.do_print_results('train', pos_train_metrics)
    misc_functions.do_print_results('test', pos_test_metrics)
    misc_functions.save_logs(os.path.join('./logs', start_time), 'pos_train', pos_train_metrics, train_POS_data_handler.num_epoch)
    misc_functions.save_logs(os.path.join('./logs', start_time), 'pos_test', pos_test_metrics, test_POS_data_handler.num_epoch)

    ###########
    # Chunking training
    ###########
    chunking_train_metrics = train_and_test_functions.train_pos_layer(train_chunking_data_handler, chunking_optimizer, chunking_module, grad_clip_max=2.0)
    chunking_test_metrics = train_and_test_functions.test_pos_layer(test_chunking_data_handler, chunking_module)
    print ('##############################')
    print ('chunking: end of epoch ' + str(train_chunking_data_handler.num_epoch))
    misc_functions.do_print_results('train', chunking_train_metrics)
    misc_functions.do_print_results('test', chunking_test_metrics)
    misc_functions.save_logs(os.path.join('./logs', start_time), 'chunking_train', chunking_train_metrics, train_chunking_data_handler.num_epoch)
    misc_functions.save_logs(os.path.join('./logs', start_time), 'chunking_test', chunking_test_metrics, test_chunking_data_handler.num_epoch)

    ###########
    # dep_parsing training
    ###########
    dep_parsing_train_metrics = train_and_test_functions.train_dep_parsing_layer(train_dep_parsing_data_handler, dep_parsing_optimizer, dep_parsing_module, grad_clip_max=3.0)
    dep_parsing_test_metrics = train_and_test_functions.test_dep_parsing_layer(test_dep_parsing_data_handler, dep_parsing_module)
    print ('##############################')
    print ('dep_parsing: end of epoch ' + str(train_dep_parsing_data_handler.num_epoch))
    misc_functions.do_print_results_dep_parsing('train', dep_parsing_train_metrics)
    misc_functions.do_print_results_dep_parsing('test', dep_parsing_test_metrics)
    misc_functions.save_logs_dep_parsing(os.path.join('./logs', start_time), 'dep_parsing_train', dep_parsing_train_metrics, train_dep_parsing_data_handler.num_epoch)
    misc_functions.save_logs_dep_parsing(os.path.join('./logs', start_time), 'dep_parsing_test', dep_parsing_test_metrics, test_dep_parsing_data_handler.num_epoch)

