
import torch
import aaev2.misc_functions as misc_functions
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import copy
import numpy as np

def train_pos_layer(data_handler, optimizer, model, grad_clip_max=1.0):
    # train the POS layer for 1 epoch
    model.train()

    criterionTagger = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
    current_epoch = data_handler.num_epoch
    nb_minibatch_done = 0
    loss_total = 0.0

    # deep copy of the model parameters before each epoch; for successive regularization term
    last_epoch_parameters = copy.deepcopy(model.state_dict())

    true_y = [] # array of np array, containing the true y for the whole dataset
    predicted_y = [] #array of np array, containing the predicted y for the whole dataset

    while current_epoch == data_handler.num_epoch:

        input_data = data_handler.next_batch()
        x = input_data['x']
        y = input_data['y']
        x_lengths = input_data['x_lengths']
        x_max_length = input_data['x_max_length']
        batch_size = input_data['batch_size']

        x = misc_functions.numpy_to_Longtensor_variable(x, requires_grad=False)
        y = misc_functions.numpy_to_Longtensor_variable(y, requires_grad=False)

        optimizer.zero_grad()

        flat_logits = model.forward(x, x_lengths, x_max_length, batch_size)['logits']

        y = y.contiguous().view(-1)
        loss = criterionTagger(flat_logits, y)

        w_l2_norm = model.compute_w_l2_norm()
        successive_reg_term = model.compute_successive_reg_term(last_epoch_parameters)
        loss_with_regularizer = loss + w_l2_norm + successive_reg_term

        loss_with_regularizer.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max)
        optimizer.step()

        _, prediction = torch.max(flat_logits, 1)

        prediction = prediction.cpu().data.numpy()
        y = y.cpu().data.numpy()
        indexes_to_remove = np.where(y == -1)
        y = np.vstack((y, prediction))
        y = np.delete(y, indexes_to_remove, axis=1)

        true_y= true_y + y[0].tolist()
        predicted_y = predicted_y + y[1].tolist()

        loss_total += loss_with_regularizer.item()
        nb_minibatch_done += 1


    loss_total = loss_total / float(nb_minibatch_done)
    accuracy = accuracy_score(true_y, predicted_y)
    macro_precision = precision_score(true_y, predicted_y, average='macro')
    macro_recall = recall_score(true_y, predicted_y, average='macro')
    macro_fmeasure = f1_score(true_y, predicted_y, average='macro')


    return (loss_total, accuracy, macro_precision, macro_recall, macro_fmeasure)

def test_pos_layer(data_handler, model):
    # train the POS layer for 1 epoch
    model.eval()

    criterionTagger = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
    current_epoch = data_handler.num_epoch
    nb_minibatch_done = 0
    loss_total = 0.0

    # deep copy of the model parameters before each epoch; for successive regularization term
    last_epoch_parameters = copy.deepcopy(model.state_dict())

    true_y = [] # array of np array, containing the true y for the whole dataset
    predicted_y = [] #array of np array, containing the predicted y for the whole dataset

    while current_epoch == data_handler.num_epoch:

        input_data = data_handler.next_batch()
        x = input_data['x']
        y = input_data['y']
        x_lengths = input_data['x_lengths']
        x_max_length = input_data['x_max_length']
        batch_size = input_data['batch_size']

        x = misc_functions.numpy_to_Longtensor_variable(x, requires_grad=False)
        y = misc_functions.numpy_to_Longtensor_variable(y, requires_grad=False)

        flat_logits = model.forward(x, x_lengths, x_max_length, batch_size)['logits']

        y = y.contiguous().view(-1)
        loss = criterionTagger(flat_logits, y)
        w_l2_norm = model.compute_w_l2_norm()
        successive_reg_term = model.compute_successive_reg_term(last_epoch_parameters)
        loss_with_regularizer = loss + w_l2_norm + successive_reg_term


        _, prediction = torch.max(flat_logits, 1)

        prediction = prediction.cpu().data.numpy()
        y = y.cpu().data.numpy()
        indexes_to_remove = np.where(y == -1)
        y = np.vstack((y, prediction))
        y = np.delete(y, indexes_to_remove, axis=1)

        true_y= true_y + y[0].tolist()
        predicted_y = predicted_y + y[1].tolist()

        loss_total += loss_with_regularizer.item()
        nb_minibatch_done += 1


    loss_total = loss_total / float(nb_minibatch_done)
    accuracy = accuracy_score(true_y, predicted_y)
    macro_precision = precision_score(true_y, predicted_y, average='macro')
    macro_recall = recall_score(true_y, predicted_y, average='macro')
    macro_fmeasure = f1_score(true_y, predicted_y, average='macro')

    print(true_y[:100])
    print(predicted_y[:100])

    return (loss_total, accuracy, macro_precision, macro_recall, macro_fmeasure)


def train_dep_parsing_layer(data_handler, optimizer, model, grad_clip_max=3.0):
    # train the dep parsing layer for 1 epoch
    model.train()

    criterionTagger = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
    current_epoch = data_handler.num_epoch
    nb_minibatch_done = 0
    loss_total = 0.0

    # deep copy of the model parameters before each epoch; for successive regularization term
    last_epoch_parameters = copy.deepcopy(model.state_dict())

    true_heads_list = [] # array of np array, containing the true heads for the whole dataset
    true_relation_label_list = [] # array of np array, containing the true relation label for the whole dataset
    predicted_heads_list = [] #array of np array, containing the predicted heads for the whole dataset
    predicted_relation_label_list = [] #array of np array, containing the predicted relation labels for the whole dataset

    while current_epoch == data_handler.num_epoch:

        input_data = data_handler.next_batch()
        x = input_data['x']
        heads = input_data['heads']
        heads_indexes_for_training = copy.deepcopy((heads))
        relation_labels = input_data['relation_labels']
        x_lengths = input_data['x_lengths']
        x_max_length = input_data['x_max_length']
        batch_size = input_data['batch_size']

        x = misc_functions.numpy_to_Longtensor_variable(x, requires_grad=False)
        heads = misc_functions.numpy_to_Longtensor_variable(heads, requires_grad=False)
        heads_indexes_for_training = misc_functions.numpy_to_Longtensor_variable(heads_indexes_for_training, requires_grad=False)
        relation_labels = misc_functions.numpy_to_Longtensor_variable(relation_labels, requires_grad=False)

        optimizer.zero_grad()

        all_logits = model.forward(x, x_lengths, x_max_length, batch_size, true_heads_indices = heads_indexes_for_training)
        flat_heads_logits = all_logits['flat_heads_logits']
        flat_relation_logits = all_logits['flat_relation_logits']

        heads = heads.contiguous().view(-1)
        relation_labels = relation_labels.contiguous().view(-1)

        loss_heads = criterionTagger(flat_heads_logits, heads)
        loss_relation_heads = criterionTagger(flat_relation_logits, relation_labels)

        w_l2_norm = model.compute_w_l2_norm()
        successive_reg_term = model.compute_successive_reg_term(last_epoch_parameters)
        loss_with_regularizer = loss_heads + loss_relation_heads + w_l2_norm + successive_reg_term

        loss_with_regularizer.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max)
        optimizer.step()

        _, prediction_heads = torch.max(flat_heads_logits, 1)
        _, prediction_relation_label = torch.max(flat_relation_logits, 1)


        prediction_heads = prediction_heads.cpu().data.numpy()
        heads = heads.cpu().data.numpy()
        indexes_to_remove = np.where(heads == -1)
        heads = np.vstack((heads, prediction_heads))
        heads = np.delete(heads, indexes_to_remove, axis=1)

        prediction_relation_label = prediction_relation_label.cpu().data.numpy()
        relation_labels = relation_labels.cpu().data.numpy()
        indexes_to_remove = np.where(relation_labels == -1)
        relation_labels = np.vstack((relation_labels, prediction_relation_label))
        relation_labels = np.delete(relation_labels, indexes_to_remove, axis=1)


        true_heads_list= true_heads_list + heads[0].tolist()
        true_relation_label_list = true_relation_label_list + relation_labels[0].tolist()
        predicted_heads_list = predicted_heads_list + heads[1].tolist()
        predicted_relation_label_list = predicted_relation_label_list + relation_labels[1].tolist()


        loss_total += loss_with_regularizer.item()
        nb_minibatch_done += 1


    loss_total = loss_total / float(nb_minibatch_done)

    # compute UAS
    true_heads_list = np.array(true_heads_list)
    predicted_heads_list = np.array(predicted_heads_list)
    good_heads = np.equal(true_heads_list, predicted_heads_list).astype(int)
    UAS = good_heads.mean()

    # compute LAS
    true_relation_label_list = np.array(true_relation_label_list)
    predicted_relation_label_list = np.array(predicted_relation_label_list)
    good_relation_labels = np.equal(true_relation_label_list, predicted_relation_label_list).astype(int)
    good_heads_and_labels = np.multiply(good_heads, good_relation_labels)
    LAS = good_heads_and_labels.mean()



    return (loss_total, UAS, LAS)

def test_dep_parsing_layer(data_handler, model):
    # test the dep parsing layer for 1 epoch
    model.eval()

    criterionTagger = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
    current_epoch = data_handler.num_epoch
    nb_minibatch_done = 0
    loss_total = 0.0

    # deep copy of the model parameters before each epoch; for successive regularization term
    last_epoch_parameters = copy.deepcopy(model.state_dict())

    true_heads_list = [] # array of np array, containing the true heads for the whole dataset
    true_relation_label_list = [] # array of np array, containing the true relation label for the whole dataset
    predicted_heads_list = [] #array of np array, containing the predicted heads for the whole dataset
    predicted_relation_label_list = [] #array of np array, containing the predicted relation labels for the whole dataset

    while current_epoch == data_handler.num_epoch:

        input_data = data_handler.next_batch()
        x = input_data['x']
        heads = input_data['heads']
        heads_indexes_for_training = copy.deepcopy((heads))
        relation_labels = input_data['relation_labels']
        x_lengths = input_data['x_lengths']
        x_max_length = input_data['x_max_length']
        batch_size = input_data['batch_size']

        x = misc_functions.numpy_to_Longtensor_variable(x, requires_grad=False)
        heads = misc_functions.numpy_to_Longtensor_variable(heads, requires_grad=False)
        heads_indexes_for_training = misc_functions.numpy_to_Longtensor_variable(heads_indexes_for_training, requires_grad=False)
        relation_labels = misc_functions.numpy_to_Longtensor_variable(relation_labels, requires_grad=False)

        all_logits = model.forward(x, x_lengths, x_max_length, batch_size, true_heads_indices = heads_indexes_for_training)
        flat_heads_logits = all_logits['flat_heads_logits']
        flat_relation_logits = all_logits['flat_relation_logits']

        heads = heads.contiguous().view(-1)
        relation_labels = relation_labels.contiguous().view(-1)

        loss_heads = criterionTagger(flat_heads_logits, heads)
        loss_relation_heads = criterionTagger(flat_relation_logits, relation_labels)

        w_l2_norm = model.compute_w_l2_norm()
        successive_reg_term = model.compute_successive_reg_term(last_epoch_parameters)
        loss_with_regularizer = loss_heads + loss_relation_heads + w_l2_norm + successive_reg_term


        _, prediction_heads = torch.max(flat_heads_logits, 1)
        _, prediction_relation_label = torch.max(flat_relation_logits, 1)


        prediction_heads = prediction_heads.cpu().data.numpy()
        heads = heads.cpu().data.numpy()
        indexes_to_remove = np.where(heads == -1)
        heads = np.vstack((heads, prediction_heads))
        heads = np.delete(heads, indexes_to_remove, axis=1)

        prediction_relation_label = prediction_relation_label.cpu().data.numpy()
        relation_labels = relation_labels.cpu().data.numpy()
        indexes_to_remove = np.where(relation_labels == -1)
        relation_labels = np.vstack((relation_labels, prediction_relation_label))
        relation_labels = np.delete(relation_labels, indexes_to_remove, axis=1)


        true_heads_list= true_heads_list + heads[0].tolist()
        true_relation_label_list = true_relation_label_list + relation_labels[0].tolist()
        predicted_heads_list = predicted_heads_list + heads[1].tolist()
        predicted_relation_label_list = predicted_relation_label_list + relation_labels[1].tolist()


        loss_total += loss_with_regularizer.item()
        nb_minibatch_done += 1


    loss_total = loss_total / float(nb_minibatch_done)

    # compute UAS
    true_heads_list = np.array(true_heads_list)
    predicted_heads_list = np.array(predicted_heads_list)
    good_heads = np.equal(true_heads_list, predicted_heads_list).astype(int)
    UAS = good_heads.mean()

    # compute LAS
    true_relation_label_list = np.array(true_relation_label_list)
    predicted_relation_label_list = np.array(predicted_relation_label_list)
    good_relation_labels = np.equal(true_relation_label_list, predicted_relation_label_list).astype(int)
    good_heads_and_labels = np.multiply(good_heads, good_relation_labels)
    LAS = good_heads_and_labels.mean()



    return (loss_total, UAS, LAS)