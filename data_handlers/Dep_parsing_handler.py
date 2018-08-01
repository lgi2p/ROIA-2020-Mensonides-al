# -*- coding: utf-8 -*-
from aaev2.data_handlers.Abstract_data_handler import Abstract_data_handler

import io
import random
import numpy as np


class Dep_parsing_handler(Abstract_data_handler):

    def __init__(self, list_of_files_with_data, label_to_id_file, vocabulary_handler, batch_size):

        Abstract_data_handler.__init__(self)
        self.list_of_files_with_data = list_of_files_with_data
        self.vocabulary_handler = vocabulary_handler
        self.label_to_id_file = label_to_id_file
        self.batch_size = batch_size
        self.label_to_id_dict = None #computed in self.read_data

        self.num_epoch = 0
        self.nb_examples_seen_in_current_epoch = 0
        self.examples = self.read_data()
        self.next_epoch()




    def read_data(self):
        #read the examples in file_with_POS_data
        # return the examples transformed into ids
        # return the labels transformed into ids. ids are taken from self.POS_label_to_id_file
        # return an array of the length of the ids array
        # the whole is returned as a list of tuple (token sequence, label_sequence, sequence_length)

        # first build the label_to_id dictionary

        label_to_id_dict = dict()
        try:
            with io.open(self.label_to_id_file, encoding='utf8') as f:
                lines= f.read().splitlines()
                for line in lines:
                    key = line.split('\t')[0]
                    value = line.split('\t')[1]
                    label_to_id_dict[key] = value
        except Exception as e:
            raise e

        self.label_to_id_dict = label_to_id_dict

        #fetch examples
        examples = []
        num_example=0
        try:
            for file_with_data in self.list_of_files_with_data:
                #one word per line, examples are separated by a blank line in the file
                with io.open(file_with_data, encoding='utf8') as f:
                    lines = f.read().splitlines()

                input_sequence = []
                relation_label_sequence = []
                heads_sequence=[]
                for line in lines:
                    if line == '':
                        examples.append({'words_ids': input_sequence, 'heads_indexes': heads_sequence, 'relation_label_ids': relation_label_sequence, 'num_example': num_example})
                        input_sequence = []
                        relation_label_sequence = []
                        heads_sequence = []
                        num_example +=1
                    elif line.split('\t')[0].isdigit():
                        token = line.split('\t')[1].lower() # all words in lower case
                        head = int(line.split('\t')[6])
                        heads_sequence.append(head)
                        label = line.split('\t')[7].split(':')[0]
                        input_sequence.append(self.vocabulary_handler.word_to_id_dictionnary.get(token, 0))
                        label_token_value = label_to_id_dict.get(label, -1)
                        if label_token_value == -1:
                            raise ('UNKNOWN LABEL')
                        else:
                            relation_label_sequence.append(float(label_token_value))
        except Exception as e:
            raise e

        return examples

    def next_epoch(self):
        random.shuffle(self.examples)
        self.nb_examples_seen_in_current_epoch = 0
        self.num_epoch += 1

    def next_batch(self):
        #return  the next batch of data, sorted by example length of words_ids
        #       if all example were already returned, reshuffle examples and start new epoch

        # gather next examples
        next_examples = self.examples[self.nb_examples_seen_in_current_epoch : min(self.nb_examples_seen_in_current_epoch + self.batch_size, len(self.examples))]
        self.nb_examples_seen_in_current_epoch += len(next_examples)

        #sort by decreasing length of words_ids
        next_examples = sorted(next_examples, key = lambda k: len(k['words_ids']), reverse = True)

        # catch len of each sequence
        x_lengths = [len(x['words_ids']) for x in next_examples]

        # padd words_ids with 2
        x = [x['words_ids'] for x in next_examples]
        x = self.pad_list_of_list_to_np_array(x, 2, np.int32)

        # padd heads with -1
        heads = [x['heads_indexes'] for x in next_examples]
        heads = self.pad_list_of_list_to_np_array(heads, -1, np.float32)

        # padd relation labels with -1
        relation_labels = [x['relation_label_ids'] for x in next_examples]
        relation_labels = self.pad_list_of_list_to_np_array(relation_labels, -1, np.float32)


        # shuffle and go next epoch if needed
        if self.nb_examples_seen_in_current_epoch == len(self.examples):
            self.next_epoch()
        elif self.nb_examples_seen_in_current_epoch > len(self.examples):
            raise ("Too many examples seen error")

        return {
            'x': x,
            'heads': heads,
            'relation_labels': relation_labels,
            'x_lengths': x_lengths,
            'x_max_length': x_lengths[0],
            'nb_y_in_batch': sum(x_lengths),
            'batch_size': len(x_lengths)
        }








