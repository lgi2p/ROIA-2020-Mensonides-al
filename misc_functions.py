
import torch
import os


###########
# misc functions here
###########


def numpy_to_Longtensor_variable(numpy_ndarray, requires_grad = True, run_on_CUDA=True):
    # transform a numpy variable into a pytorch variable.
    if run_on_CUDA:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor

    tensor = torch.from_numpy(numpy_ndarray)
    tensor = tensor.type(dtype)
    tensor = torch.autograd.Variable(tensor, requires_grad=requires_grad)
    return tensor


def do_print_results(train_or_test, task_metrics):
    # print the results of training/test loop
    # train_or_test = 'train' or 'test'
    # task_metrics = (loss, accuracy, macro_precision, macro_recall, macro_fmeasure)
    print (train_or_test+' loss = ' + str(task_metrics[0]) + '\t'+train_or_test+' accuracy = ' + str(task_metrics[1]))
    print (train_or_test+' precision = ' + str(task_metrics[2]) + '\t'+train_or_test+' recall = ' + str(task_metrics[3]) + '\t'+train_or_test+' f1score = ' + str(task_metrics[4]))

    print ('\n')

def do_print_results_dep_parsing(train_or_test, task_metrics):
    # print the results of training/test loop for dep parsing
    # train_or_test = 'train' or 'test'
    # task_metrics = (loss, UAS, LAS)
    print (train_or_test+' loss = ' + str(task_metrics[0]) + '\t'+train_or_test+' UAS = ' + str(task_metrics[1]) + '\t'+train_or_test+' LAS = '+str(task_metrics[2]))

    print ('\n')



def save_logs(directory, file_name, task_metrics, num_epoch):
    # save log of a training/test loop
    # directory = directory to save logs, eg './logs/start_time'
    # file_name = pos_train, pos_test, chunking_train etc
    # task_metrics: metrics as returned by train_layer

    if not os.path.exists(os.path.join(directory, file_name)):
        with open(os.path.join(directory, file_name), 'w') as f:
            f.write('# file separated by \\t \n')
            f.write('# num_epoch, loss, accuracy, precision, recall, f1score\n')

    with open(os.path.join(directory, file_name), 'a') as f:
        f.write(str(num_epoch)+'\t')
        for i in range(4):
            f.write(str(task_metrics[i])+'\t')
        f.write(str(task_metrics[4])+'\n')

def save_logs_dep_parsing(directory, file_name, task_metrics, num_epoch):
    # save log of a training/test loop for dep parsing
    # directory = directory to save logs, eg './logs/start_time'
    # file_name = pos_train, pos_test, chunking_train etc
    # task_metrics: metrics as returned by train_layer

    if not os.path.exists(os.path.join(directory, file_name)):
        with open(os.path.join(directory, file_name), 'w') as f:
            f.write('# file separated by \\t \n')
            f.write('# num_epoch, loss, UAS, LAS\n')

    with open(os.path.join(directory, file_name), 'a') as f:
        f.write(str(num_epoch)+'\t')
        for i in range(2):
            f.write(str(task_metrics[i])+'\t')
        f.write(str(task_metrics[2])+'\n')



