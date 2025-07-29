import os
import logging
import torch
import shutil

def set_gpu(args):
    if args.device == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.device.split(',')]
        print('use gpu:', gpu_list)
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    return gpu_list.__len__()


def allocate_tensors():
    """
    init data tensors
    :return: data tensors
    """
    tensors = dict()
    tensors['support_data'] = torch.FloatTensor()
    tensors['support_label'] = torch.LongTensor()
    tensors['query_data'] = torch.FloatTensor()
    tensors['query_label'] = torch.LongTensor()
    return tensors


def set_logging_config(logdir):
    """
    set logging configuration
    :param logdir: directory put logs
    :return: None
    """
    if not os.path.exists(logdir): os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s", level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')), logging.StreamHandler(os.sys.stdout)])


def save_checkpoint(state, is_best, exp_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'), os.path.join('{}'.format(exp_name), 'model_best.pth.tar'))


def adjust_learning_rate(optimizers, lr, iteration, dec_lr_step, lr_adj_base):
    """
    adjust learning rate after some iterations
    :param optimizers: the optimizers
    :param lr: learning rate
    :param iteration: current iteration
    :param dec_lr_step: decrease learning rate in how many step
    :return: None
    """
    new_lr = lr * (lr_adj_base ** (int(iteration / dec_lr_step)))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def label2edge(label, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    # get size
    num_samples = label.size(1)
    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)
    # compute edge
    edge = torch.eq(label_i, label_j).float().to(device)
    return edge


def one_hot_encode(num_classes, class_idx, device):
    """
    one-hot encode the ground truth
    :param num_classes: number of total class
    :param class_idx: belonging class's index
    :param device: the gpu device that holds the one-hot encoded ground truth label
    :return: one-hot encoded ground truth label
    """
    class_idx = class_idx.to(device)  
    eye_tensor = torch.eye(num_classes, device=device)
    return eye_tensor[class_idx]


def preprocessing(num_ways, num_shots, num_queries, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for evaluation
    """
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots                   
    num_samples = num_supports + num_queries * num_ways  

    # set edge mask (to distinguish support and query edges)
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device)
    
    return num_supports, num_samples, query_edge_mask, evaluation_mask


def set_tensors(tensors, batch):
    """
    set data to initialized tensors
    :param tensors: initialized data tensors
    :param batch: current batch of data
    :return: None
    """
    support_data, support_label, query_data, query_label = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    tensors['query_label'].resize_(query_label.size()).copy_(query_label)


def initialize_nodes_edges(batch, num_supports, tensors, batch_size, num_queries, num_ways, device):
    """
    :param batch: data batch
    :param num_supports: number of samples in support set
    :param tensors: initialized tensors for holding data
    :param batch_size: how many tasks per batch
    :param num_queries: number of samples in query set
    :param num_ways: number of classes for each few-shot task
    :param device: the gpu device that holds all data

    :return: data of support set, label of support set,
             data of query set,label of query set,
             data of support and query set, label of support and query set,
             initialized node features of distribution graph (Vd_(0)),
             initialized edge features of point graph (Ep_(0)),
             initialized edge_features_of distribution graph (Ed_(0))
    """
    
    # allocate data in this batch to specific variables
    set_tensors(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)    
    support_label = tensors['support_label'].squeeze(0)  
    query_data = tensors['query_data'].squeeze(0)        
    query_label = tensors['query_label'].squeeze(0) 

    all_data = torch.cat([support_data, query_data], 1)    
    all_label = torch.cat([support_label, query_label], 1) 
    all_label_in_edge = label2edge(all_label, device)      
    light_edge_feature = all_label_in_edge.clone()         

    light_edge_feature[:, num_supports:, :num_supports] = 1. / num_supports
    light_edge_feature[:, :num_supports, num_supports:] = 1. / num_supports
    light_edge_feature[:, num_supports:, num_supports:] = 0
    for i in range(num_ways * num_queries): 
        light_edge_feature[:, num_supports + i, num_supports + i] = 1

    color_edge_feature = light_edge_feature.clone() 
    
    return support_data, support_label, query_data, query_label, \
           all_data, all_label_in_edge, light_edge_feature, color_edge_feature


def backbone_two_stage_initialization(full_data, encoder):
    """
    :param full_data: raw data
    :param encoder: backbone network
    :return: last_layer_data: last layer logits from backbone network
             second_last_layer_data: second last layer logits from backbone network
    """
    last_layer_data_temp = []
    second_last_layer_data_temp = []
    
    for data in full_data.chunk(full_data.size(1), dim=1):
        encoded_result = encoder(data.squeeze(1))
        last_layer_data_temp.append(encoded_result[0])
        second_last_layer_data_temp.append(encoded_result[1])
    
    last_layer_data = torch.stack(last_layer_data_temp, dim=1)
    second_last_layer_data = torch.stack(second_last_layer_data_temp, dim=1)
    
    return last_layer_data, second_last_layer_data