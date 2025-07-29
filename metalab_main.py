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
from metalab_dataloader import MiniImagenet, TieredImagenet, Cifar, FC100, CUB200, Aircraft, Meta_iNat, Tiered_Meta_iNat, DataLoader
from metalab_utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, \
            preprocessing, initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode
from metalab.metalab_LabNet import LabNet
from metalab.metalab_LabGNN import LabGNN
# torch.autograd.set_detect_anomaly(True)

class MetaLab_Trainer(object):
    def __init__(self, enc_module, gnn_module, data_loader, log, arg, config, best_step):
        """ Trainer of MetaLab model
        :param enc_module: LabNet
        :param gnn_module: LabGNN
        """
        self.arg = arg
        self.config = config
        self.train_opt = config['train_config']
        self.eval_opt = config['eval_config']
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
        
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())
        self.optimizer = optim.Adam(params=self.module_params, lr=self.train_opt['lr'], weight_decay=self.train_opt['weight_decay'])
        
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        self.global_step = best_step
        self.best_step = best_step
        self.val_acc, self.test_acc = 0, 0


    def train(self):
        # Task initialization
        num_supports, num_samples, query_edge_mask, evaluation_mask = \
        preprocessing(self.train_opt['num_ways'], self.train_opt['num_shots'], self.train_opt['num_queries'],  self.train_opt['batch_size'], self.arg.device)

        is_all_ones = torch.all(evaluation_mask == 1).item()
        print('evaluation_mask:', is_all_ones)  # True 或 False


        # meta-training and meta-evaluation loop
        for iteration, batch in tqdm(enumerate(self.data_loader['train']()), desc=f"Training"):
            #########################
            # 1) meta-training mode
            #########################
            self.optimizer.zero_grad() 
            self.global_step += 1
            
            # initialize nodes and edges for light and color graphs
            support_data, support_label, query_data, query_label, all_data, all_label_in_edge, \
            light_edge_feature, color_edge_feature = initialize_nodes_edges(batch, num_supports,
                                                                        self.tensors,
                                                                        self.train_opt['batch_size'],
                                                                        self.train_opt['num_queries'],
                                                                        self.train_opt['num_ways'],
                                                                        self.arg.device)

            self.enc_module.train()
            self.gnn_module.train()

            # LabNet: two-tiered feature embedding
            second_last_layer_data, last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

            # LabGNN: graph classification
            light_edge_similarity, light_node_similarity, color_edge_similarities = self.gnn_module(second_last_layer_data,
                                                                                              last_layer_data,
                                                                                              light_edge_feature,
                                                                                              color_edge_feature)
            # Prediction and Loss
            total_loss, query_node_cls_acc_generations, query_edge_loss_generations = \
                                          self.compute_train_loss_pred(all_label_in_edge, light_edge_similarity, color_edge_similarities, 
                                          light_node_similarity, query_edge_mask, evaluation_mask, num_supports, support_label, query_label)

            total_loss.backward()
            self.optimizer.step()            
            adjust_learning_rate(optimizers=[self.optimizer],  lr=self.train_opt['lr'], iteration=self.global_step,
                                 dec_lr_step=self.train_opt['dec_lr'], lr_adj_base =self.train_opt['lr_adj_base'])
            
            if self.global_step % self.arg.log_step == 0:
                self.log.info('step : {}  train_edge_loss : {}  node_acc : {}'.format(self.global_step,
                                                                                      query_edge_loss_generations[-1],
                                                                                      query_node_cls_acc_generations[-1]))
            #########################
            # 2) meta-evaluation mode
            #########################
            if self.global_step % self.eval_opt['interval'] == 0:
                is_best = 0
                test_acc = self.eval(partition='val')
                if test_acc > self.test_acc:
                    is_best = 1
                    self.test_acc = test_acc
                    self.best_step = self.global_step
                self.log.info('test_acc : {}         step : {} '.format(test_acc, self.global_step))
                self.log.info('test_best_acc : {}    step : {}'.format( self.test_acc, self.best_step))
                save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'gnn_module_state_dict': self.gnn_module.state_dict(),
                    'test_acc': self.test_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, os.path.join(self.arg.checkpoint_dir, self.arg.exp_name))


    def eval(self, partition='test', log_flag=True):
        """ evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        """
        if partition=='val': 
            iteration= self.eval_opt['iteration']
            num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
                self.eval_opt['num_ways'],
                self.eval_opt['num_shots'],
                self.eval_opt['num_queries'],
                self.eval_opt['batch_size'],
                self.arg.device)
        elif partition=='test': 
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
                                                                            self.eval_opt['batch_size'],
                                                                            self.eval_opt['num_queries'],
                                                                            self.eval_opt['num_ways'],
                                                                            self.arg.device)

            self.enc_module.eval()
            self.gnn_module.eval()
            
            # LabNet: two-tiered feature embedding
            last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

            # LabGNN: graph classification
            light_edge_similarity, _, _ = self.gnn_module(second_last_layer_data, last_layer_data,
                                                          light_edge_feature, color_edge_feature)

            # Predition and Loss
            query_node_cls_acc_generations, query_edge_loss_generations = \
                                        self.compute_eval_loss_pred(query_edge_loss_generations,query_node_cls_acc_generations,
                                                                    all_label_in_edge,light_edge_similarity,
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


    def compute_train_loss_pred(self, all_label_in_edge, light_edge_similarity, color_edge_similarities,light_node_similarities,
                                query_edge_mask, evaluation_mask, num_supports, support_label, query_label):
        """ compute the total loss, query loss and query accuracy
        :param all_label_in_edge: ground truth label in edge form of light graph
        :param light_edge_similarity: prediction edges of light graph
        :param light_node_similarities: node similarities of light graph
        :param query_edge_mask: mask for queries
        :param evaluation_mask: mask for evaluation (for unsupervised setting)
        :param num_supports: number of samples in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :param color_edge_similarities: color edge similarities
        :return: total loss, query accuracy, query loss
        """
        # is_all_ones = torch.all(evaluation_mask == 1).item()
        # print('evaluation_mask:', is_all_ones)  # True 或 False
        
        # compute total edge loss of generations
        total_edge_loss_generations_light = [self.edge_loss((1 - edge_similarity), (1 - all_label_in_edge))
                                            for edge_similarity in light_edge_similarity]
        total_edge_loss_generations_color = [self.edge_loss((1 - color_similarity), (1 - all_label_in_edge))
                                            for color_similarity in color_edge_similarities]
        color_loss_coeff = 0.1
        total_edge_loss_generations = [total_edge_loss_instance + color_loss_coeff * total_edge_loss_color
            for (total_edge_loss_instance, total_edge_loss_color) in zip(total_edge_loss_generations_light, total_edge_loss_generations_color)]
        # compute query edge loss of generations 
        pos_query_edge_loss_generations = [
            torch.sum(total_edge_loss_generation * query_edge_mask * all_label_in_edge)
            / torch.sum(query_edge_mask * all_label_in_edge)
            for total_edge_loss_generation in total_edge_loss_generations]
        neg_query_edge_loss_generations = [
            torch.sum(total_edge_loss_generation * query_edge_mask * (1 - all_label_in_edge))
            / torch.sum(query_edge_mask * (1 - all_label_in_edge))
            for total_edge_loss_generation in total_edge_loss_generations]
        # print('pos_query_edge_loss_generations:', pos_query_edge_loss_generations)
        # print('neg_query_edge_loss_generations:', neg_query_edge_loss_generations)

        query_edge_loss_generations = [
            pos_query_edge_loss_generation + neg_query_edge_loss_generation
            for (pos_query_edge_loss_generation, neg_query_edge_loss_generation) in zip(pos_query_edge_loss_generations, neg_query_edge_loss_generations)]
        # compute query node acc of generations
        query_node_pred_generations = [
            torch.bmm(edge_similarity[:, num_supports:, :num_supports], one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device))
            for edge_similarity in light_edge_similarity]        
        query_node_acc_generations = [
            torch.eq(torch.max(query_node_pred_generation, -1)[1], query_label.long()).float().mean()
            for query_node_pred_generation in query_node_pred_generations]
        # compute query node pred loss of generations
        query_node_pred_generations_ = [
            torch.bmm(node_similarity[:, num_supports:, :num_supports], one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device))
            for node_similarity in light_node_similarities]
        query_node_pred_loss = [
            self.pred_loss(query_node_pred_generation.view(-1, query_node_pred_generation.size(-1)), query_label.view(-1))
            for query_node_pred_generation in query_node_pred_generations_ ]
        # compute loss of generations
        total_loss_generations = [
            query_edge_loss_generation + 0.1 * query_node_pred_loss_
            for (query_edge_loss_generation, query_node_pred_loss_) in zip(query_edge_loss_generations, query_node_pred_loss)]

        # compute total loss
        total_loss = []
        num_loss = self.config['num_loss_generation']
        for l in range(num_loss - 1): total_loss += [total_loss_generations[l].view(-1) * self.config['generation_weight']]
        total_loss += [total_loss_generations[-1].view(-1) * 1.0]
        total_loss = torch.mean(torch.cat(total_loss, 0))
        return total_loss, query_node_acc_generations, query_edge_loss_generations


    def compute_eval_loss_pred(self, query_edge_losses, query_node_accs, all_label_in_edge, light_edge_similarity,
                               query_edge_mask, evaluation_mask, num_supports, support_label, query_label):
        """ compute the query loss and query accuracy
        :param query_edge_losses: container for losses of queries' edges
        :param query_node_accs: container for node accuracy of queries
        :param all_label_in_edge: ground truth label in edge form of light graph
        :param light_edge_similarity: prediction edges of light graph
        :param query_edge_mask: edge mask for queries
        :param evaluation_mask: mask for evaluation
        :param num_supports: samples number in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :return: query loss, query accuracy
        """
        # is_all_ones = torch.all(evaluation_mask == 1).item()
        # print('evaluation_mask:', is_all_ones)  # True 或 False
        
        edge_similarity = light_edge_similarity[-1]
        query_node_pred = torch.bmm(
            edge_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.eval_opt['num_ways'], support_label.long(), self.arg.device))        
        query_node_acc = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()
        query_node_accs += [query_node_acc.item()]
        full_edge_loss = self.edge_loss(1 - edge_similarity, 1 - all_label_in_edge)
        pos_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * all_label_in_edge) / torch.sum(
            query_edge_mask * all_label_in_edge)
        neg_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * (1 - all_label_in_edge)) / torch.sum(
            query_edge_mask * (1 - all_label_in_edge))
        
        # weighted loss for balancing pos/neg
        query_edge_loss = pos_query_edge_loss + neg_query_edge_loss
        query_edge_losses += [query_edge_loss.item()]

        return query_node_accs, query_edge_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='GPU ID')
    # parser.add_argument('--dataset_root', type=str, default='/home/ssdData/qcfData/Benchmark_MetaLab', help='Dataset root directory')
    parser.add_argument('--dataset_root', type=str, default='/home/ssdData/qcfData/Benchmark_FewShot', help='Dataset root directory')

    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', '5way_1shot_resnet12_mini-imagenet.py'),
                        help='config file with parameters of experiments, which is placed under ./metalab_config')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'Result_checkpoints'),
                        help='path that checkpoint will be saved, loaded, and placed under ./checkpoints')
    parser.add_argument('--display_step', type=int, default=100, help='display training information')
    parser.add_argument('--log_step', type=int, default=5, help='log information in how many steps')
    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'Result_logs'), 
                        help='path that log will be saved, where checkpoint file is placed under ./logs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    args_opt = parser.parse_args()
    config_file = args_opt.config

    # Set train and test datasets and the corresponding data loaders
    print(f'Using devices: {args_opt.device}')
    device = torch.device(f'cuda:{args_opt.device}')
    
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.config
    train_opt = config['train_config']
    eval_opt = config['eval_config']
    test_opt = config['test_config']
    args_opt.exp_name = '{}_{}way_{}shot_{}query_{}'.format('metaLab', train_opt['num_ways'], 
                                                    train_opt['num_shots'], train_opt['num_queries'], config['dataset_name'])
    set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name))
    logger = logging.getLogger('main')

    # Load the configuration params of the experiment
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
    
    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImagenet
        print('Dataset: MiniImagenet')
    elif config['dataset_name'] == 'tiered-imagenet':
        dataset = TieredImagenet
        print('Dataset: TieredImagenet')
    elif config['dataset_name'] == 'cifar-fs':
        dataset = Cifar
        print('Dataset: Cifar')
    elif config['dataset_name'] == 'fc100':
        dataset = FC100
        print('Dataset: FC100')
    elif config['dataset_name'] == 'cub-200-2011':
        dataset = CUB200
        print('Dataset: CUB200-200-2011')
    elif config['dataset_name'] == 'aircraft-fs':
        dataset = Aircraft
        print('Dataset: Aircraft-Fewshot')
    elif config['dataset_name'] == 'meta-iNat':
        dataset = Meta_iNat
        print('Dataset: Meta-iNat')
    elif config['dataset_name'] == 'tiered-meta-iNat':
        dataset = Tiered_Meta_iNat
        print('Dataset: Tiered-Meta-iNat')
    else:
        logger.info('Invalid dataset: {}, please specify a dataset from mini-imagenet, tiered-imagenet, \
        cifar-fs, cub-200-2011, meta-iNat and tiered-meta-iNat.'.format(config['dataset_name']))
        exit()
    dataset_train = dataset(root=args_opt.dataset_root, partition='train')
    dataset_valid = dataset(root=args_opt.dataset_root, partition='val')
    dataset_test = dataset(root=args_opt.dataset_root, partition='test')
    train_loader = DataLoader(dataset_train,
                              num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'])
    valid_loader = DataLoader(dataset_valid,
                              num_tasks=eval_opt['batch_size'],
                              num_ways=eval_opt['num_ways'],
                              num_shots=eval_opt['num_shots'],
                              num_queries=eval_opt['num_queries'],
                              epoch_size=eval_opt['iteration'])
    test_loader = DataLoader(dataset_test,
                             num_tasks=test_opt['batch_size'],
                             num_ways=test_opt['num_ways'],
                             num_shots=test_opt['num_shots'],
                             num_queries=test_opt['num_queries'],
                             epoch_size=test_opt['iteration'])
    data_loader = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
    encoder_flag = True if args_opt.exp_name.__contains__('cifar') or args_opt.exp_name.__contains__('fc100') else False
    
    
    if config['backbone'] == 'labnet':
        enc_module = LabNet(emb_size=config['emb_size'], encoder_flag=encoder_flag).to(device)
        print('Backbone: LabNet')
    else:
        logger.info('Invalid backbone: {}, please specify a backbone model'.format(config['backbone']))
        exit()

    gnn_module = LabGNN(config['emb_size'], config['num_generation'], train_opt['dropout'],
                      train_opt['num_ways'] * train_opt['num_shots'],
                      train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'] * train_opt['num_queries'],
                      train_opt['loss_indicator'],
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

    if args_opt.mode == 'train': trainer.train()
    elif args_opt.mode == 'eval': trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()