# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2025
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, random, logging, argparse, time
from tqdm import tqdm
import importlib.util
from metalab_dataloader import Places365, Stanford_Car, CropDisease, EuroSAT, DataLoader
from metalab_utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, \
            preprocessing, initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode
from metalab.metalab_LabNet import LabNet
from metalab.metalab_LabGNN import LabGNN


class MetaLab_Trainer(object):
    def __init__(self, enc_module, gnn_module, data_loader, log, arg, config, best_step):
        """ Trainer of MetaLab
        :param enc_module: LabNet
        :param gnn_module: LabGNN
        """
        self.arg = arg
        self.config = config
        self.test_opt = config['test_config']
        self.log = log
        self.data_loader = data_loader
        
        print(f'Using devices: {self.arg.device}')
        self.arg.device = torch.device(f'cuda:{self.arg.device}')

        self.tensors = allocate_tensors()
        for key, tensor in self.tensors.items(): 
            self.tensors[key] = tensor.to(self.arg.device)
            
        self.enc_module = enc_module.to(arg.device)
        self.gnn_module = gnn_module.to(arg.device)
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    def eval(self, partition='test', log_flag=True):
        """ evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        """
        if partition=='test': 
            iteration= self.test_opt['iteration']        
            num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
                self.test_opt['num_ways'],
                self.test_opt['num_shots'],
                self.test_opt['num_queries'],
                self.test_opt['batch_size'],
                self.arg.device)
        
        query_edge_loss_generations = []
        query_node_cls_acc_generations = []

        # main training loop, batch size is the number of tasks
        for current_iteration, batch in tqdm(enumerate(self.data_loader[partition]()), desc=f"Testing on {partition}({iteration}it)"):
            # initialize nodes and edges for light and color graphs
            support_data, support_label, query_data, query_label, all_data, all_label_in_edge, \
            light_edge_feature, color_edge_feature = initialize_nodes_edges(batch,
                                                                            num_supports,
                                                                            self.tensors,
                                                                            self.test_opt['batch_size'],
                                                                            self.test_opt['num_queries'],
                                                                            self.test_opt['num_ways'],
                                                                            self.arg.device)
            self.enc_module.eval()
            self.gnn_module.eval()
            
            # LabNet: two-tiered feature embedding
            last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)
            
            # LabGNN: graph classification
            light_edge_similarity, _, _ = self.gnn_module(second_last_layer_data, last_layer_data,
                                                          light_edge_feature, color_edge_feature)
            
            # Prediction and Loss
            query_node_cls_acc_generations, query_edge_loss_generations = \
                                        self.compute_eval_loss_pred(query_edge_loss_generations, query_node_cls_acc_generations,
                                                                    all_label_in_edge, light_edge_similarity,
                                                                    query_edge_mask,evaluation_mask,
                                                                    num_supports,support_label,query_label)
        
        if log_flag:
            self.log.info('------------------------------------')
            self.log.info('step : {}  {}_edge_loss : {}  {}_node_acc : {}'.format(
                self.global_step, partition, np.array(query_edge_loss_generations).mean(),
                partition, np.array(query_node_cls_acc_generations).mean()))
            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                          (current_iteration,
                           np.array(query_node_cls_acc_generations).mean() * 100,
                           np.array(query_node_cls_acc_generations).std() * 100,
                           1.96 * np.array(query_node_cls_acc_generations).std() / np.sqrt(float(len(np.array(query_node_cls_acc_generations)))) * 100))
            self.log.info('------------------------------------')

        return np.array(query_node_cls_acc_generations).mean()


    def compute_eval_loss_pred(self, query_edge_losses, query_node_accs, all_label_in_edge,
                               light_edge_similarity, query_edge_mask, evaluation_mask,
                               num_supports, support_label, query_label):
        """
        compute the query loss and query accuracy
        :param query_edge_losses: container for losses of queries' edges
        :param query_node_accs: container for classification accuracy of queries
        :param light_edge_similarity: prediction edges of light graph
        :param query_edge_mask: edge mask for queries
        :param evaluation_mask: mask for evaluation
        :param num_supports: number of samples in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :return: query loss, query accuracy
        """
        edge_similarity = light_edge_similarity[-1]        
        query_node_pred = torch.bmm(
            edge_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.test_opt['num_ways'], support_label.long(), self.arg.device))
        
        query_node_acc = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()
        query_node_accs += [query_node_acc.item()]
        full_edge_loss = self.edge_loss(1 - edge_similarity, 1 - all_label_in_edge)
        pos_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * all_label_in_edge * evaluation_mask) / torch.sum(
            query_edge_mask * all_label_in_edge * evaluation_mask)
        neg_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * (1 - all_label_in_edge) * evaluation_mask) / torch.sum(
            query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)        
        query_edge_loss = pos_query_edge_loss + neg_query_edge_loss
        query_edge_losses += [query_edge_loss.item()]

        return query_node_accs, query_edge_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='device ID of gpu')
    parser.add_argument('--dataset_root', type=str, default='/home/ssdData/qcfData/Benchmark_MetaLab', help='root directory of dataset')
    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', '5way_1shot_resnet12_mini-imagenet.py'),
                        help='config file with parameters of the experiment. It is assumed that the config file is placed under the directory ./config')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'Result_checkpoints'),
                        help='path that checkpoint will be saved and loaded. It is assumed that the checkpoint file is placed under the directory ./checkpoints')
    parser.add_argument('--display_step', type=int, default=100, help='display training information in how many step')
    parser.add_argument('--log_step', type=int, default=5, help='log information in how many steps')
    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'Result_logs'), 
                        help='path that log will be saved. It is assumed that the checkpoint file is placed under the directory ./logs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    args_opt = parser.parse_args()
    config_file = args_opt.config
    
    print(f'Using devices: {args_opt.device}')
    device = torch.device(f'cuda:{args_opt.device}')
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    test_opt = config['test_config']
    args_opt.exp_name = '{}_{}way_{}shot_{}query_{}'.format('metaLab', test_opt['num_ways'], test_opt['num_shots'], test_opt['num_queries'], config['dataset_name'])
    set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name))
    logger = logging.getLogger('main')
    logger.info('Launching experiment from: {}'.format(config_file))
    logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
    print()
    logger.info('-------------command line arguments-------------')
    logger.info(args_opt)
    print()
    logger.info('-------------configs-------------')
    logger.info(config)

    # set random seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if config['dataset_name'] == 'places':
        dataset = Places365
        print('Dataset: Places365')
    elif config['dataset_name'] == 'cars':
        dataset = Stanford_Car
        print('Dataset: Stanford_Car')
    elif config['dataset_name'] == 'CropDisease':
        dataset = CropDisease
        print('Dataset: CropDisease')
    elif config['dataset_name'] == 'EuroSAT':
        dataset = EuroSAT
        print('Dataset: EuroSAT')        
    else:
        logger.info('Invalid dataset: {}, please specify a dataset from places, cars, \
        CropDisease and EuroSAT.'.format(config['dataset_name']))
        exit()
    
    dataset_test = dataset(root=args_opt.dataset_root, partition='test')
    test_loader = DataLoader(dataset_test,
                             num_tasks=test_opt['batch_size'],
                             num_ways=test_opt['num_ways'],
                             num_shots=test_opt['num_shots'],
                             num_queries=test_opt['num_queries'],
                             epoch_size=test_opt['iteration'])
    data_loader = {'test': test_loader}

    encoder_flag = True if args_opt.exp_name.__contains__('cifar') or args_opt.exp_name.__contains__('fc100') else False    
    enc_module = LabNet(emb_size=config['emb_size'], encoder_flag=encoder_flag).to(device)
    gnn_module = LabGNN(config['emb_size'], config['num_generation'], test_opt['dropout'],
                      test_opt['num_ways'] * test_opt['num_shots'],
                      test_opt['num_ways'] * test_opt['num_shots'] + test_opt['num_ways'] * test_opt['num_queries'],
                      test_opt['loss_indicator'],
                      config['light_distance_metric'], config['color_distance_metric']).to(device)
    
    if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)):
        os.makedirs(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name))
        logger.info('no checkpoint for model: {}, make a new one at {}'.format(args_opt.exp_name, os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
        best_step = 0
    else:
        if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar')): best_step = 0
        else:
            logger.info('find a checkpoint, loading checkpoint from {}'.format(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
            best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar'))
            logger.info('best model pack loaded')
            best_step = best_checkpoint['iteration']
            enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
            gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])
            logger.info('current best validation accuracy is: {}, at step: {}'.format(best_checkpoint['test_acc'], best_step))

    trainer = MetaLab_Trainer(enc_module=enc_module, gnn_module=gnn_module,
                              data_loader=data_loader, log=logger,arg=args_opt, config=config, best_step=best_step)

    if args_opt.mode == 'eval': trainer.eval()
    else:
        print('wrong mode')
        exit()


if __name__ == '__main__':
    main()