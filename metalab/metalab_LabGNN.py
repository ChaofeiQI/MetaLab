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
import torch.nn.functional as F

class LightSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: gpu device stores tensors
        :param dropout: dropout rate
        """
        super(LightSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        
        # Interactor construction
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.light_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vl_last_gen, el_last_gen, distance_metric):
        """ 
        :param vl_last_gen: last generation's node feature of Light graph
        :param el_last_gen: last generation's edge feature of Light graph
        :param distance_metric: metric for distance
        :return: current generation's edge feature of Light graph, light node similarity
        """
        vl_i = vl_last_gen.unsqueeze(2)             
        vl_j = torch.transpose(vl_i, 1, 2)          
        if distance_metric == 'l2':   vl_similarity = (vl_i - vl_j)**2         
        elif distance_metric == 'l1': vl_similarity = torch.abs(vl_i - vl_j)   
        light_node_similarity = -torch.sum(vl_similarity, 3)                
        
        trans_similarity = torch.transpose(vl_similarity, 1, 3)
        el_ij = torch.sigmoid(self.light_sim_transform(trans_similarity))
        
        diagonal_mask = 1.0 - torch.eye(vl_last_gen.size(1)).unsqueeze(0).repeat(vl_last_gen.size(0), 1, 1).to(el_last_gen.get_device())
        el_last_gen *= diagonal_mask
        el_last_gen_sum = torch.sum(el_last_gen, -1, True)

        try: el_ij = F.normalize(el_ij.squeeze(1).clone() * el_last_gen.clone(), p=1, dim=-1) * el_last_gen_sum
        except Exception as e: print(f"Error during computation: {e}")

        diagonal_reverse_mask = torch.eye(vl_last_gen.size(1)).unsqueeze(0).to(el_last_gen.get_device())
        el_ij += (diagonal_reverse_mask + 1e-6)
        el_ij /= torch.sum(el_ij, dim=2).unsqueeze(-1)

        return el_ij, light_node_similarity


class ColorSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: gpu device stores tensors
        :param dropout: dropout rate
        """
        super(ColorSimilarity, self).__init__()
        self.in_c = in_c         
        self.base_c = base_c     
        self.dropout = dropout
        
        # Interactor construction
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.color_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vc_last_gen, ec_last_gen, distance_metric):
        """
        :param vc_last_gen: last generation's node feature of Color graph
        :param ec_last_gen: last generation's edge feature of Color graph
        :param distance_metric: metric for distance
        :return: current generation's edge feature of Color graph, color node similarity
        """
        vc_i = vc_last_gen.unsqueeze(2)    
        vc_j = torch.transpose(vc_i, 1, 2) 

        if distance_metric == 'l2':   vc_similarity = (vc_i - vc_j)**2      
        elif distance_metric == 'l1': vc_similarity = torch.abs(vc_i - vc_j)
        color_node_similarity = -torch.sum(vc_similarity, 3)             
        
        trans_similarity = torch.transpose(vc_similarity, 1, 3)
        ec_ij = torch.sigmoid(self.color_sim_transform(trans_similarity))
        
        diagonal_mask = 1.0 - torch.eye(vc_last_gen.size(1)).unsqueeze(0).repeat(vc_last_gen.size(0), 1, 1).to(ec_last_gen.get_device())
        ec_last_gen *= diagonal_mask
        ec_last_gen_sum = torch.sum(ec_last_gen, -1, True)

        try: ec_ij = F.normalize(ec_ij.squeeze(1).clone() * ec_last_gen.clone(), p=1, dim=-1) * ec_last_gen_sum
        except Exception as e: print(f"Error during computation: {e}")

        diagonal_reverse_mask = torch.eye(vc_last_gen.size(1)).unsqueeze(0).to(ec_last_gen.get_device())
        ec_ij += (diagonal_reverse_mask + 1e-6)
        ec_ij /= torch.sum(ec_ij, dim=2).unsqueeze(-1)

        return ec_ij, color_node_similarity


class Color_layering(nn.Module):
    def __init__(self, in_c, out_c):
        """
        :param in_c: number of input channel for fc layer
        :param out_c:number of output channel for fc layer
        """
        super(Color_layering, self).__init__()
        self.color_node_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True),
                                                 nn.LeakyReLU()])
        self.out_c = out_c

    def forward(self, light_edge, color_node):
        """
        :param light_edge: current generation's edge feature of Light graph
        :param color_node: last generation's node feature of Color graph
        :return: current generation's node feature of Color graph
        """
        meta_batch = light_edge.size(0)
        num_sample = light_edge.size(1)

        color_node = torch.cat([light_edge[:, :, :self.out_c], color_node], dim=2)
        color_node = color_node.view(meta_batch*num_sample, -1)
        
        color_node = self.color_node_transform(color_node)
        color_node = color_node.view(meta_batch, num_sample, -1)
        return color_node


class Light_gradient(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: gpu device stores tensors
        :param dropout: dropout rate
        """
        super(Light_gradient, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        
        # Interactor construction
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]
        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        self.light_node_transform = nn.Sequential(*layer_list)

    def forward(self, color_edge, light_node):
        """
        :param color_edge: current generation's edge feature of Color graph
        :param light_node: last generation's node feature of light graph
        :return: current generation's node feature of light graph
        """
        meta_batch = light_node.size(0) 
        num_sample = light_node.size(1)

        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(color_edge.get_device())
        edge_feat = F.normalize(color_edge * diag_mask, p=1, dim=-1)

        # color attention and aggregate
        aggr_feat = torch.bmm(edge_feat, light_node)
        node_feat = torch.cat([light_node, aggr_feat], -1).transpose(1, 2)
        
        # interactor
        node_feat = self.light_node_transform(node_feat.unsqueeze(-1))
        light_node = node_feat.transpose(1, 2).squeeze(-1)
     
        return light_node


#########################################
# LabGNN
#########################################
class LabGNN(nn.Module):
    def __init__(self, emb_size, num_generations, dropout, num_support_sample, num_sample, loss_indicator, light_metric, color_metric):
        """
        :param num_generations: number of total generations
        :param dropout: dropout rate
        :param num_support_sample: number of support sample
        :param num_sample: number of sample
        :param loss_indicator: indicator of what losses are using
        :param light_metric: metric for distance in light graph
        :param color_metric: metric for distance in color graph
        """
        super(LabGNN, self).__init__()
        self.emb_size = emb_size
        self.generation = num_generations
        self.dropout = dropout
        self.num_support_sample = num_support_sample
        self.num_sample = num_sample
        self.loss_indicator = loss_indicator
        self.light_metric = light_metric
        self.color_metric = color_metric
        
        # Light nodes & edges update
        L_edge_init = LightSimilarity(self.emb_size, self.emb_size, dropout=self.dropout)
        self.add_module('Light_edge_initial', L_edge_init)
        
        # Color nodes & edges update
        AB_edge_init = ColorSimilarity(self.emb_size, self.emb_size, dropout=self.dropout)
        self.add_module('Color_edge_initial', AB_edge_init)
        
        # Construct Graph Fearure Learners
        for l in range(self.generation):
            L_edge = LightSimilarity(self.emb_size, self.emb_size, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('Light_edge_generation_{}'.format(l), L_edge)
            L2AB_node = Color_layering(self.num_sample+self.emb_size, self.emb_size)
            self.add_module('L2AB_node_generation_{}'.format(l), L2AB_node)
            AB_edge = ColorSimilarity(self.emb_size, self.num_sample, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('Color_edge_generation_{}'.format(l), AB_edge)
            AB2L_node = Light_gradient(self.emb_size*2, self.emb_size, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('AB2L_node_generation_{}'.format(l), AB2L_node)
  
    def forward(self, init_second_node, init_final_node, mask_light_edge, mask_color_edge):
        """ 
        :param init_middle_node: feature extracted from second last layer of Embedding Network
        :param init_light_node: feature extracted from last layer of Embedding Network
        :param init_light_edge: initialized edge of light graph
        :param init_color_edge: initialized edges of Color graph
        :return: classification result, light_similarity, color_similarity
        """
        init_light_second_node=init_second_node[:, :, 0:int(init_second_node.shape[2]/2)] 
        init_color_second_node=init_second_node[:, :, int(init_second_node.shape[2]/2):]
        init_light_final_node =init_final_node[:, :, 0:int(init_final_node.shape[2]/2)] 
        init_color_final_node =init_final_node[:, :, int(init_final_node.shape[2]/2):]
        light_edge_similarities, light_node_similarities = [], [] 
        color_edge_similarities, color_node_similarities = [], [] 
        
        # edges initialization
        init_light_edge, _ = self._modules['Light_edge_initial'](init_light_second_node, mask_light_edge, self.light_metric)
        init_color_edge, _ = self._modules['Color_edge_initial'](init_color_second_node, mask_color_edge, self.color_metric)
        light_edge, color_edge = init_light_edge, init_color_edge
        
        # nodes initialization
        light_node, color_node = init_light_final_node, init_color_final_node
        
        # Message Passing
        for l in range(self.generation): 
            # 1)start: L(l)
            light_edge, light_node_similarity = self._modules['Light_edge_generation_{}'.format(l)](light_node, light_edge, self.light_metric)
            # 2)color layering
            color_node = self._modules['L2AB_node_generation_{}'.format(l)](light_edge, color_node)
            # 3)C(l)
            color_edge, color_node_similarity = self._modules['Color_edge_generation_{}'.format(l)](color_node, color_edge, self.color_metric)
            # 4)light gradient
            light_node = self._modules['AB2L_node_generation_{}'.format(l)](color_edge, light_node)
            # 5)save similarities
            light_edge_similarities.append(light_edge * self.loss_indicator[0])            
            light_node_similarities.append(light_node_similarity * self.loss_indicator[1]) 
            color_edge_similarities.append(color_edge * self.loss_indicator[2])            
            color_node_similarities.append(color_node_similarity * self.loss_indicator[3]) 
        
        return light_edge_similarities, light_node_similarities, color_edge_similarities