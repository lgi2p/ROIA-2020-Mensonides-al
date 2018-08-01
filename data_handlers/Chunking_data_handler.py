# -*- coding: utf-8 -*-
from aaev2.data_handlers.Abstract_data_handler import Abstract_data_handler

import io
import random
import numpy as np


class Chunking_data_handler(Abstract_data_handler):

    def __init__(self, file_with_data, chunking_label_to_id_file, vocabulary_handler, batch_size):

        Abstract_data_handler.__init__(self)
        self.file_with_data = file_with_data
        self.vocabulary_handler = vocabulary_handler
        self.chunking_label_to_id_file = chunking_label_to_id_file
        self.batch_size = batch_size
        self.label_to_id_dict = None #computed in self.read_data

        self.num_epoch = 0
        self.nb_examples_seen_in_current_epoch = 0
        self.examples = self.read_data()
        self.next_epoch()




    def read_data(self):
        #read the examples in file_with_chunking_data
        # return the examples transformed into ids
        # return the labels transformed into ids. ids are taken from self.chunking_label_to_id_file
        # return an array of the length of the ids array
        # the whole is returned as a list of tuple (token sequence, label_sequence, sequence_length)

        # first build the label_to_id dictionary

        label_to_id_dict = dict()
        try:
            with io.open(self.chunking_label_to_id_file, encoding='utf8') as f:
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
        try:
            #one word per line, examples are separated by a blank line in the file
            with io.open(self.file_with_data, encoding='utf8') as f:
                lines = f.read().splitlines()

            examples = []
            input_sequence = []
            label_sequence = []
            num_example=0
            for line in lines:
                if line == '':
                    examples.append({'words_ids': input_sequence, 'label_ids': label_sequence, 'num_example': num_example})
                    input_sequence = []
                    label_sequence = []
                    num_example +=1
                else:
                    token = line.split(' ')[0].lower() # all words in lower case
                    label = line.split(' ')[2]
                    input_sequence.append(self.vocabulary_handler.word_to_id_dictionnary.get(token, 0))
                    label_token_value = label_to_id_dict.get(label, -1)
                    if label_token_value == -1:
                        raise ('UNKNOWN LABEL')
                    else:
                        label_sequence.append(float(label_token_value))

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

        # padd labels with -1
        y = [x['label_ids'] for x in next_examples]
        y = self.pad_list_of_list_to_np_array(y, -1, np.float32)


        # shuffle and go next epoch if needed
        if self.nb_examples_seen_in_current_epoch == len(self.examples):
            self.next_epoch()
        elif self.nb_examples_seen_in_current_epoch > len(self.examples):
            raise ("Too many examples seen error")

        return {
            'x': x,
            'y': y,
            'x_lengths': x_lengths,
            'x_max_length': x_lengths[0],
            'nb_y_in_batch': sum(x_lengths),
            'batch_size': len(x_lengths)
        }








