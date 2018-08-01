# -*- coding: utf-8 -*-

import random
import numpy as np

class Abstract_data_handler():

    def __init__(self):
        pass

    def read_data(self):
        #read the data
        # to be overwritten in children classes
        raise NotImplementedError

    def next_batch(self):
        # return next batch of data
        raise NotImplementedError

    def pad_list_of_list_to_np_array(self, x, pad_with, dtype):
        # input x = a list of list. each list should be be padded with pad_with
        # pad_with = what should the list be padded with
        # dtype = the dtype of the np array

        # the list should already be sorted by decreasing list

        max_length = len(x[0])
        res = np.zeros([1,max_length], dtype=dtype)

        for l in x:
            tmp = np.array(l, dtype=dtype).reshape(1,len(l))
            tmp = np.pad(tmp, ((0, 0), (0, max_length - len(l))), mode='constant', constant_values=pad_with)

            res = np.concatenate((res, tmp), axis=0)



        # remove the first useless line from input_tokens and input_labels
        res = np.delete(res, 0, axis=0)

        return res